# Baseline FSDP otomatis dari CT utuh

Workflow ini membandingkan nnU-Net 2D dan 3D full resolution pada 53 pasien train yang valid. Kedua arm menjalankan `nnUNetTrainer_100epochs` sampai epoch 100. Sepuluh pasien validation berada di luar dataset nnU-Net saat fingerprint, planning, normalisasi, dan training. Prediksi memakai CT utuh tanpa ROI ground truth dan tanpa klik. Dice dihitung pada seluruh volume native. Perbandingan dengan skor SAM interaktif sebelumnya tidak setara karena SAM menggunakan ROI ground truth dan lima klik oracle per sisi.

Ekspor dibuat dari `hybrid_data.pt` yang hash-nya cocok dengan foundation. CT sumber diverifikasi terhadap SHA256 cache. Target bilateral direkonstruksi dari dua native target cache, lalu semua indeks foreground dibandingkan dengan target utuh pasien. `train_28` dikecualikan karena citra sumbernya berisi label map. File sumber dan cache tidak diubah. Arsip diagnosis lokal tidak memuat CT/checkpoint, sehingga workflow ini dijalankan di DGX. Kedua arm memakai normalisasi z-score per pasien pada nnU-Net. Kebijakan ini menjaga variasi intensitas di atas batas clipping foreground yang terukur pada model lama, tetapi dampaknya terhadap Dice belum diketahui.

Persyaratan satu kali pada environment DGX: `python -m pip install nnunetv2` jika `python -c 'import nnunetv2'` gagal. Periksa kapasitas disk karena ekspor dan preprocessing menggandakan data; script menolak persiapan bila ruang bebas kurang dari tiga kali ukuran CT ditambah 20 GiB. nnU-Net 2D dan 3D dijalankan berurutan pada satu GPU. Jumlah epoch sama, tetapi jumlah update optimizer, arsitektur, dan biaya komputasi berbeda. Workflow ini adalah pembanding otomatis, bukan ablasi satu variabel.

```bash
cd ~/raid/Clara/fsdp-train
git fetch origin codex/exclude-mislabeled-train28
git merge --ff-only origin/codex/exclude-mislabeled-train28
conda activate fsdp-med3d
GPU_ID=auto FSDP_SOURCE_ROOT="$HOME/raid/normal case_86" \
  FSDP_FOUNDATION_DIR="$HOME/raid/Clara/fsdp-train/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292" \
  bash scripts/launch_fsdp_auto_nnunet_nohup.sh
tail -f logs/fsdp_auto_nnunet/latest.log
```

Jika push branch belum berhasil, salin `outputs/private/fsdp_auto_nnunet_code.tar.gz` dari Mac ke direktori proyek DGX melalui `scp`, lalu ekstrak dengan `tar -xzf fsdp_auto_nnunet_code.tar.gz` sebelum menjalankan launcher. Arsip ini hanya berisi kode dan panduan, bukan CT, label, atau checkpoint.

Launcher memeriksa ketersediaan GPU, paket, dan sumber, lalu berjalan lewat nohup. Lokasi run tersimpan di `logs/fsdp_auto_nnunet/latest_run.txt`. Setelah kedua arm selesai, baca `comparison.json`, `predictions_2d/summary.json`, `predictions_3d_fullres/summary.json`, dan CSV per pasien. Prediksi NIfTI juga tersedia pada folder tiap arm untuk review anatomi.

Pilihan model berikutnya harus didasarkan pada Dice per pasien, precision/recall, serta kasus dengan target terlewat. Jika kedua arm gagal menemukan struktur kecil pada CT utuh, lanjutkan localizer kasar dengan crop prediksi, lalu segmenter detail. Evaluasi lokalizer harus menghitung target di luar crop sebagai false negative. Test tetap belum dibaca.
