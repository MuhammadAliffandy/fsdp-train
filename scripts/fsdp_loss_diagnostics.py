"""Counterfactual loss/gradient measurements on an identical five-click forward."""
import math

import torch
from monai.losses import DiceLoss
from torch.nn import functional as F


class RecordedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, squared_pred=True)
        self.terms = []

    def forward(self, logits, target):
        dice = self.dice(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        fraction = target.mean().detach()
        weight = torch.sqrt((1 - fraction) / fraction.clamp_min(1e-8)).clamp(1., 8.)
        balanced = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
        balanced = balanced / (1 + (weight - 1) * fraction)
        self.terms.append(dict(dice=dice, bce=bce, balanced=balanced, logits=logits,
                               target=target, positive_weight=float(weight)))
        return dice + bce


def add(first, second, scale=1.):
    return [a + scale * b for a, b in zip(first, second)]


def norm(values):
    return math.sqrt(sum(float(v.double().square().sum()) for v in values))


def cosine(first, second):
    denominator = norm(first) * norm(second)
    return sum(float((a.double() * b.double()).sum()) for a, b in zip(first, second)) / denominator if denominator else None


def group(name):
    if name.startswith('sam.image_encoder.'):
        return 'image_encoder'
    if name.startswith('sam.mask_decoder.'):
        return 'mask_decoder'
    return 'detail_cnn'


def grouped_norm(names, gradients):
    return {key: norm([g for name, g in zip(names, gradients) if group(name) == key])
            for key in ('image_encoder', 'mask_decoder', 'detail_cnn')}


def inspect_forward(model, criterion):
    terms = criterion.terms
    if len(terms) != 10:
        raise ValueError('Expected detail/context terms at each of five clicks')
    detail, context = terms[::2], terms[1::2]
    mean = lambda rows, key: sum(row[key] for row in rows) / len(rows)
    losses = {'detail_dice': mean(detail, 'dice'), 'detail_bce': mean(detail, 'bce'),
              'context_dice': mean(context, 'dice'), 'context_bce': mean(context, 'bce'),
              'detail_balanced_bce': mean(detail, 'balanced')}
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    names, parameters = zip(*named)
    gradients, logit = {}, {}
    for key, loss in losses.items():
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss diagnostic')
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        gradients[key] = [torch.zeros_like(p, device='cpu') if g is None else g.detach().float().cpu()
                          for p, g in zip(parameters, values)]
        rows = context if key.startswith('context') else detail
        values = torch.autograd.grad(loss, [row['logits'] for row in rows], retain_graph=True)
        fg, bg = 0., 0.
        for row, value in zip(rows, values):
            fg += float(value[row['target'].bool()].abs().sum())
            bg += float(value[~row['target'].bool()].abs().sum())
        logit[key] = {'foreground_abs_sum': fg, 'background_abs_sum': bg}
    detail_g = add(gradients['detail_dice'], gradients['detail_bce'])
    context_g = add(gradients['context_dice'], gradients['context_bce'])
    candidates = {'control': add(detail_g, context_g, .25),
                  'auxiliary_0_1': add(detail_g, context_g, .1),
                  'moderate_balanced_bce': add(add(gradients['detail_dice'], gradients['detail_balanced_bce']), context_g, .25)}
    for values in (*gradients.values(), *candidates.values()):
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError('Nonfinite diagnostic gradient')
    report = {'component_loss': {k: float(v.detach()) for k, v in losses.items()},
              'component_parameter_gradient_l2': {k: grouped_norm(names, v) for k, v in gradients.items()},
              'component_logit_gradients': logit,
              'detail_context_cosine': {key: cosine(
                  [g for name, g in zip(names, detail_g) if group(name) == key],
                  [g for name, g in zip(names, context_g) if group(name) == key])
                  for key in ('image_encoder', 'mask_decoder', 'detail_cnn')},
              'detail_foreground_fraction': float(detail[0]['target'].mean()),
              'context_foreground_fraction': float(context[0]['target'].mean()),
              'detail_positive_weight_candidate': detail[0]['positive_weight'], 'candidate_gradients': {}}
    for key, values in candidates.items():
        length = norm(values)
        report['candidate_gradients'][key] = {'unclipped_l2': length, 'clip_factor_at_1': min(1., 1. / (length + 1e-6)),
                                              'groups': grouped_norm(names, values)}
    probability = detail[-1]['logits'].detach().sigmoid()
    foreground = detail[-1]['target'].bool()
    report['last_click_detail_probability'] = {
        'foreground_mean': float(probability[foreground].mean()) if foreground.any() else None,
        'background_mean': float(probability[~foreground].mean()) if (~foreground).any() else None}
    return report, names, candidates


def virtual_adamw(model, names, gradients, saved_optimizer):
    """One hypothetical per-ROI update with saved moments; parameters/state stay untouched."""
    all_named = list(model.named_parameters())
    groups = [[(n, p) for n, p in all_named if n.startswith('sam.')],
              [(n, p) for n, p in all_named if not n.startswith('sam.')]]
    if len(saved_optimizer['param_groups']) != 2:
        raise ValueError('Expected original SAM/CNN AdamW parameter groups')
    lookup = {}
    for pairs, settings in zip(groups, saved_optimizer['param_groups']):
        if len(pairs) != len(settings['params']) or settings.get('amsgrad', False) or settings.get('maximize', False):
            raise ValueError('Unsupported optimizer layout')
        for (name, parameter), pid in zip(pairs, settings['params']):
            lookup[name] = (parameter, settings, saved_optimizer['state'].get(pid))
    scale = min(1., 1. / (norm(gradients) + 1e-6))
    totals = {key: [0., 0., 0] for key in ('image_encoder', 'mask_decoder', 'detail_cnn')}
    for name, gradient in zip(names, gradients):
        parameter, settings, state = lookup[name]
        weight = parameter.detach().float().cpu()
        gradient = gradient * scale
        beta1, beta2 = settings['betas']
        if state:
            step = float(state['step']) + 1
            first = state['exp_avg'].float().cpu() * beta1 + gradient * (1 - beta1)
            second = state['exp_avg_sq'].float().cpu() * beta2 + gradient.square() * (1 - beta2)
        else:
            step, first, second = 1., gradient * (1 - beta1), gradient.square() * (1 - beta2)
        delta = -settings['lr'] * (settings['weight_decay'] * weight
                + (first / (1 - beta1 ** step)) / ((second / (1 - beta2 ** step)).sqrt() + settings['eps']))
        totals[group(name)][0] += float(delta.double().square().sum())
        totals[group(name)][1] += float(weight.double().square().sum())
        totals[group(name)][2] += int(state is not None)
    return {key: {'delta_l2': math.sqrt(a), 'parameter_l2': math.sqrt(b),
                  'update_to_weight_ratio': math.sqrt(a / b) if b else None, 'tensors_with_saved_moments': count}
            for key, (a, b, count) in totals.items()}
