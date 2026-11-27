"""Slice minibatches and physically weighted boundary supervision for native five-slice CT."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from physical_unet25d import slice_stack
from physical_resunet import supervised_loss


def shape_batches(samples, batch_size, seed):
    if batch_size < 1:
        raise ValueError('Positive slice batch size required')
    groups = {}
    for index, sample in enumerate(samples):
        shape = tuple(sample['image'].shape[2:4])
        groups.setdefault(shape, []).extend((index,z) for z in range(sample['image'].shape[-1]))
    generator = np.random.default_rng(seed)
    batches = []
    for shape in sorted(groups):
        entries = groups[shape]
        generator.shuffle(entries)
        batches.extend(entries[i:i+batch_size] for i in range(0,len(entries),batch_size))
    generator.shuffle(batches)
    return batches


def collate_slices(samples, entries, seed, augment=True):
    shapes = {tuple(samples[i]['image'].shape[2:4]) for i,_ in entries}
    if len(shapes) != 1:
        raise ValueError('Slice minibatches must share native XY shape; padding changes GroupNorm and inference')
    rng = np.random.default_rng(seed)
    planes, masks, spacings = [], [], []
    for sample_index, center in entries:
        sample = samples[sample_index]
        image = slice_stack(sample['image'], [center], 5)[0].clone()
        mask = sample['target'][0, :, :, :, center].clone()
        if augment:
            if rng.random() < .5:
                image, mask = image.flip(-2), mask.flip(-2)
            if rng.random() < .3:
                image = image * float(rng.uniform(.9,1.1)) + float(rng.uniform(-.1,.1))
        planes.append(image); masks.append(mask)
        spacings.append(sample['native']['spacing'][:2])
    height = max(p.shape[-2] for p in planes)
    width = max(p.shape[-1] for p in planes)
    padded, targets, supports = [], [], []
    for image, mask in zip(planes, masks):
        padding = (0, width-image.shape[-1], 0, height-image.shape[-2])
        padded.append(F.pad(image, padding))
        targets.append(F.pad(mask, padding))
        supports.append(F.pad(torch.ones_like(mask, dtype=torch.bool), padding))
    return torch.stack(padded), torch.stack(targets), torch.stack(supports), spacings


def boundary_weights(target, valid, spacings, radius_mm=1.):
    weights = torch.ones_like(target, dtype=torch.float32)
    for i, spacing in enumerate(spacings):
        if len(spacing) != 2 or any(not math.isfinite(s) or s <= 0 for s in spacing):
            raise ValueError('Invalid in-plane physical spacing')
        radii = [math.ceil(radius_mm / s) for s in spacing]
        kernel = tuple(2*r+1 for r in radii)
        mask = target[i:i+1].float()
        dilated = F.max_pool2d(mask, kernel, 1, tuple(radii))
        eroded = -F.max_pool2d(-mask, kernel, 1, tuple(radii))
        weights[i:i+1] += 5 * (dilated-eroded).clamp(0,1)
    # Padding has no loss or gradient; weights are normalized inside the loss.
    return weights.masked_fill(~valid.bool(), 0)


def slice_loss(outputs, target, valid, spacings, arm):
    if arm == 'control':
        return supervised_loss(tuple(p.unsqueeze(-1) for p in outputs), target.unsqueeze(-1), valid.unsqueeze(-1))
    if arm != 'focal_boundary':
        raise ValueError('Unknown slice loss arm')
    weights = boundary_weights(target, valid, spacings)
    truth = target.float()
    support = valid.float()
    axes = (1,2,3)
    losses = []
    with torch.autocast(device_type=target.device.type, enabled=False):
        for logits in outputs:
            logits = logits.float()
            ce = F.binary_cross_entropy_with_logits(logits, truth, reduction='none')
            pt = torch.exp(-ce)
            focal = ((1-pt).square() * ce * support).sum(axes) / support.sum(axes).clamp_min(1)
            probability = logits.sigmoid() * support
            dice = 1 - (2*(probability*truth).sum(axes)+1e-5) / (probability.sum(axes)+(truth*support).sum(axes)+1e-5)
            edge = (ce * weights).sum(axes) / weights.sum(axes).clamp_min(1)
            losses.append((.5*focal + 2*dice + .5*edge).mean())
    return sum(weight*loss for weight,loss in zip((1.,.3,.15), losses)) / 1.45


def train_slices(model, samples, optimizer, scaler, device, precision, epoch, args):
    from train_private_long import accumulation_window
    batches = shape_batches(samples, args.slice_batch_size, args.seed + epoch)
    entry_count = sum(len(batch) for batch in batches)
    if not entry_count:
        raise ValueError('No training slices')
    model.train(); optimizer.zero_grad(set_to_none=True)
    total, updates, skipped, positive = 0., 0, 0, 0
    norms = []
    for step, batch in enumerate(batches):
        image, target, valid, spacings = collate_slices(samples, batch, args.seed + epoch*100000+step)
        positive += int(target.flatten(1).any(1).sum())
        image,target,valid = [t.to(device) for t in (image,target,valid)]
        with torch.autocast(device_type=device.type, enabled=device.type=='cuda', dtype=precision):
            outputs = model.plane(image)
        loss = slice_loss(outputs,target,valid,spacings,args.slice_loss)
        if not loss.isfinite():
            raise ValueError('Nonfinite slice loss')
        window, update = accumulation_window(step,len(batches),args.accumulation)
        scaler.scale(loss/window).backward()
        total += float(loss.detach())*len(batch)
        if update:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),12.)
            if not norm.isfinite() and not scaler.is_enabled():
                raise ValueError('Nonfinite slice gradients')
            norms.append(float(norm))
            previous=scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            success=scaler.get_scale()>=previous
            updates+=int(success); skipped+=int(not success)
            optimizer.zero_grad(set_to_none=True)
        if step%25==0 or step==len(batches)-1:
            print(f'Brain-recipe={args.slice_loss} epoch={epoch+1}/{args.epochs} batch={step+1}/{len(batches)} loss={float(loss):.5f} updates={updates} skipped={skipped}',flush=True)
    if updates+skipped != math.ceil(len(batches)/args.accumulation):
        raise ValueError('Slice optimizer accounting differs')
    return dict(loss=total/entry_count, optimizer_updates=updates, amp_skipped_updates=skipped,
                slice_samples=entry_count, positive_slices=positive, negative_slices=entry_count-positive,
                gradient_norm_mean=float(np.mean(norms)), loss_arm=args.slice_loss,
                batch_shape_groups=len({tuple(s['image'].shape[2:4]) for s in samples}),
                extra_collate_padding_fraction=0.)
