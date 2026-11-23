"""Static scientific PNGs for train-only segmentation results."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def finish(fig, path, caption):
    fig.text(.03, .015, caption, fontsize=8, color='#444444', va='bottom')
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)


def metric_comparison(threshold_overview, path):
    latest = threshold_overview['checkpoints']['unet_latest']['amp']
    at_half, chosen = latest['at_threshold_0_5'], latest['best_train_only_candidate']
    fields = ['mean_bilateral_patient_dice', 'minimum_side_dice', 'mean_precision', 'mean_recall']
    labels = ['Mean Dice\nper pasien', 'Dice sisi\nterendah', 'Mean precision\nper pasien', 'Mean recall\nper pasien']
    fig, ax = plt.subplots(figsize=(10, 6))
    x, width = np.arange(4), .34
    series = [(-width / 2, at_half, '#888888'), (width / 2, chosen, '#15735f')] if chosen['threshold'] != .5 else [(0, chosen, '#15735f')]
    for shift, row, color in series:
        values = [row[key] * 100 for key in fields]
        bars = ax.bar(x + shift, values, width, label=f"Threshold {row['threshold']:g}", color=color)
        ax.bar_label(bars, labels=[f'{value:.2f}%' for value in values], padding=4, fontsize=10)
    ax.set(xticks=x, xticklabels=labels, ylabel='Skor (%)', ylim=(0, 108),
           title='U-Net checkpoint terakhir: pengaruh threshold pada data train')
    ax.legend(loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.08, .94, '2 pasien train | 4 ROI sisi | bobot sama | Dice pada grid native', fontsize=11)
    fig.subplots_adjust(bottom=.2, top=.85, left=.08, right=.97)
    finish(fig, path, 'Sumber: threshold_overview.json, AMP. Threshold dipilih pada dua pasien train ini; validation/test belum dievaluasi.')


def dice_matrix(report, level, path):
    selected = report['checkpoints']['unet_latest']['amp']['best_train_only_candidate']['threshold']
    thresholds = sorted(set((.5, selected)), reverse=True)
    columns = [(name, threshold) for name in ('unet_best', 'unet_latest') for threshold in thresholds]
    lookup, aliases = {}, set()
    for name, threshold in columns:
        row = next(row for row in report['checkpoints'][name]['amp']['curve'] if row['threshold'] == threshold)
        for item in row['per_side' if level == 'side' else 'per_patient']:
            alias = f"{item['case']} / {item['side']}" if level == 'side' else item['case']
            aliases.add(alias)
            lookup[(alias, name, threshold)] = item['dice' if level == 'side' else 'dice_full_target']
    aliases = sorted(aliases)
    matrix = np.array([[lookup[(alias, name, threshold)] * 100 for name, threshold in columns] for alias in aliases])
    fig, ax = plt.subplots(figsize=(10, max(4, len(aliases) * .7 + 2)))
    view = ax.imshow(matrix, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
    for row in range(len(aliases)):
        for column in range(len(columns)):
            ax.text(column, row, f'{matrix[row, column]:.2f}%', ha='center', va='center',
                    fontsize=12, color='white' if matrix[row, column] > 60 else 'black')
    ax.set(yticks=range(len(aliases)), yticklabels=aliases,
           xticks=range(len(columns)), xticklabels=[f'{name}\nt={threshold:g}' for name, threshold in columns],
           xlabel='Checkpoint dan threshold sigmoid', ylabel='Sisi train' if level == 'side' else 'Pasien train',
           title='Dice full-target pada grid native per ' + ('sisi' if level == 'side' else 'pasien'))
    fig.colorbar(view, ax=ax, label='Dice (%)')
    fig.subplots_adjust(left=.23, bottom=.24, top=.88, right=.9)
    finish(fig, path, 'Sumber: thresholds.json, AMP. Label pasien disamarkan. Hasil held-in train; bukan hasil validation.')


def threshold_curves(report, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, color in [('unet_best', '#777777'), ('unet_latest', '#15735f')]:
        curve = report['checkpoints'][name]['amp']['curve']
        x = [row['threshold'] for row in curve]
        for ax, field in zip(axes, ('mean_bilateral_patient_dice', 'minimum_diagnostic_dice')):
            ax.plot(x, [row[field] * 100 for row in curve], 'o-', color=color, label=name)
    for ax, title in zip(axes, ('Mean Dice bilateral per pasien', 'Minimum Dice seluruh sisi dan pasien')):
        ax.axvline(.5, color='#999999', linestyle=':', label='Threshold awal 0.5')
        ax.set(title=title, xlabel='Threshold probabilitas', ylabel='Dice native (%)', ylim=(-2, 102), xlim=(0, 1))
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Pemeriksaan threshold hanya pada dua pasien train', fontsize=14)
    fig.subplots_adjust(bottom=.2, top=.8, wspace=.3)
    finish(fig, path, 'Sumber: thresholds.json, AMP; titik-titik adalah threshold yang benar-benar diuji. Pemilihan threshold bersifat train-only.')


def training_curves(history, path, sparse=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    loss_rows = [row for row in history if row.get('loss') is not None]
    measured = [row for row in history if row.get('mean_bilateral_patient_dice') is not None]
    style = 'o' if sparse else '-'
    axes[0].plot([row['epoch'] for row in loss_rows], [row['loss'] for row in loss_rows], style, color='#15735f')
    axes[0].set(title='Objective training', xlabel='Epoch', ylabel='DiceCE loss')
    axes[1].plot([row['epoch'] for row in measured], [row['mean_bilateral_patient_dice'] * 100 for row in measured],
                 'o' if sparse else 'o-', color='#15735f', label='Frozen native mean bilateral Dice')
    moving_key = 'moving_model_grid_train_dice'
    moving = [row for row in history if row.get(moving_key) is not None]
    axes[1].plot([row['epoch'] for row in moving], [row[moving_key] * 100 for row in moving],
                 'x' if sparse else '--', color='#777777', label='Moving model-grid mean side Dice')
    axes[1].set(title='Skor training pada threshold 0.5', xlabel='Epoch', ylabel='Dice (%)')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Riwayat run U-Net tiny: 2 pasien train', fontsize=14)
    fig.subplots_adjust(bottom=.2, top=.82, wspace=.3)
    caption = 'Tiga titik dari log pengguna; tidak merekonstruksi kurva lengkap.' if sparse else 'Sumber: history.csv. Skor moving memakai bobot yang berubah; frozen native memakai checkpoint epoch tersebut.'
    finish(fig, path, caption)


def confusion_matrices(rows, path):
    latest = [row for row in rows if row['checkpoint'] == 'unet_latest']
    cases = sorted({row['case'] for row in latest})
    thresholds = sorted({row['threshold'] for row in latest}, reverse=True)
    fig, axes = plt.subplots(len(cases), len(thresholds), figsize=(10, 8), squeeze=False)
    for r, alias in enumerate(cases):
        for c, threshold in enumerate(thresholds):
            row = next(row for row in latest if row['case'] == alias and row['threshold'] == threshold)
            counts = np.array([[row['tn'], row['fp']], [row['fn'], row['tp']]], dtype=np.int64)
            fractions = counts / np.maximum(1, counts.sum(axis=1, keepdims=True))
            ax = axes[r, c]
            ax.imshow(fractions, vmin=0, vmax=1, cmap='Blues')
            for y in (0, 1):
                for x in (0, 1):
                    ax.text(x, y, f'{counts[y, x]:,}\n{fractions[y, x]:.2%}', ha='center', va='center',
                            color='white' if fractions[y, x] > .6 else 'black', fontsize=10)
            ax.set(title=f'{alias} | threshold {threshold:g}', xticks=(0, 1), yticks=(0, 1),
                   xticklabels=('Background', 'FSDP'), yticklabels=('Background', 'FSDP'),
                   xlabel='Prediksi', ylabel='Ground truth')
    fig.suptitle('Confusion matrix native: checkpoint terakhir, dua pasien train', fontsize=13)
    fig.subplots_adjust(bottom=.14, top=.87, hspace=.48, wspace=.38)
    finish(fig, path, 'Angka = voxel; warna/persentase dinormalisasi per baris GT. Domain = union ROI native valid + semua voxel target; bukan seluruh CT.')


def overlay_grid(sample, predictions, threshold, path, checkpoint, epoch):
    raw, gt = sample['raw'], sample['native_target']
    spacing = np.asarray(sample['metadata']['spacing_mm'])
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for r, axis in enumerate((2, 1, 0)):
        remaining = [i for i in range(3) if i != axis]
        index = int(np.argmax(gt.sum(axis=tuple(remaining))))
        ct = np.take(raw, index, axis=axis).T
        target = np.take(gt, index, axis=axis).T
        half = np.take(predictions[.5], index, axis=axis).T
        selected = np.take(predictions[threshold], index, axis=axis).T
        extent = (0, ct.shape[1] * spacing[remaining[0]], 0, ct.shape[0] * spacing[remaining[1]])
        for c, ax in enumerate(axes[r]):
            ax.imshow(ct, cmap='gray', vmin=-1000, vmax=1000, origin='lower', extent=extent)
            mask, color = (target, '#00ced1') if c == 0 else (half if c == 1 else selected, '#d82782')
            if c < 3 and mask.any():
                ax.imshow(np.ma.masked_where(~mask, np.ones_like(mask)), cmap=ListedColormap([color]),
                          vmin=0, vmax=1, origin='lower', extent=extent, alpha=.4)
                if not mask.all():
                    ax.contour(mask, levels=[.5], colors=[color], linewidths=.7, origin='lower', extent=extent)
            if c == 3:
                categories = np.zeros(target.shape, np.uint8)
                categories[target & selected] = 1
                categories[~target & selected] = 2
                categories[target & ~selected] = 3
                ax.imshow(np.ma.masked_where(categories == 0, categories),
                          cmap=ListedColormap(['none', '#2a9d51', '#d82782', '#efb21b']), vmin=0, vmax=3,
                          origin='lower', extent=extent, alpha=.65)
            ax.set_title(['GT (cyan)', 'Prediksi t=0.5', f'Prediksi t={threshold:g}', 'TP hijau / FP magenta / FN kuning'][c], fontsize=9)
            ax.set_xlabel(f'Bidang native {"XYZ"[axis]}, slice {index + sample["metadata"]["crop_offset_xyz"][axis]}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f'{sample["case"]} / {sample["side"]} | {checkpoint}, epoch {epoch}\n'
                 'Data train; irisan dipilih pada massa GT terbesar per sumbu', fontsize=13)
    fig.subplots_adjust(left=.02, right=.98, bottom=.08, top=.89, hspace=.3, wspace=.12)
    finish(fig, path, 'GT = anotasi final. Ukuran bidang mempertahankan spacing native; label sumbu native tidak menyatakan orientasi anatomi standar.')


def slice_montages(sample, prediction, threshold, directory):
    gt, raw = sample['native_target'], sample['raw']
    indices = np.flatnonzero((gt | prediction).any(axis=(0, 1)))
    files = []
    for page, start in enumerate(range(0, len(indices), 24), 1):
        chosen = indices[start:start + 24]
        rows = int(np.ceil(len(chosen) / 6))
        fig, axes = plt.subplots(rows, 6, figsize=(14, rows * 2.3 + 1.2), squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, index in zip(axes.ravel(), chosen):
            ct, target, pred = raw[:, :, index].T, gt[:, :, index].T, prediction[:, :, index].T
            spacing = sample['metadata']['spacing_mm']
            extent = (0, ct.shape[1] * spacing[0], 0, ct.shape[0] * spacing[1])
            ax.imshow(ct, cmap='gray', vmin=-1000, vmax=1000, origin='lower', extent=extent)
            errors = np.zeros(target.shape, np.uint8)
            errors[target & pred], errors[~target & pred], errors[target & ~pred] = 1, 2, 3
            ax.imshow(np.ma.masked_where(errors == 0, errors), vmin=0, vmax=3,
                      cmap=ListedColormap(['none', '#2a9d51', '#d82782', '#efb21b']),
                      alpha=.65, origin='lower', extent=extent)
            ax.set_title(f'Z native {int(index) + sample["metadata"]["crop_offset_xyz"][2]}', fontsize=8)
        fig.suptitle(f'{sample["case"]} / {sample["side"]} | latest t={threshold:g} | halaman {page}', fontsize=12)
        fig.subplots_adjust(top=.84, bottom=.1, wspace=.1, hspace=.25)
        filename = f'latest_{sample["name"]}_slices_z_{page:02d}.png'
        finish(fig, directory / filename, 'Semua slice Z berisi GT atau prediksi ditampilkan. TP hijau, FP magenta, FN kuning. Held-in train; bukan validation.')
        files.append(filename)
    return files
