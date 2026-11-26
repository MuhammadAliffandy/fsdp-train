# SAM 49% dengan augmentasi fisik

Kelanjutan memakai checkpoint loss control dengan Dice 49,47%, CNN32/Turbo8 dan riwayat prompt. Dua arm berurutan dimulai dari checkpoint yang sama: control memakai light augmentation dan translasi lama; affine menambahkan rotasi ±5° dan skala 0,95–1,05 dengan probabilitas 50%. Transformasi fisik berpusat sama diterapkan pada CT detail, konteks SAM, target dan support. Target memakai nearest-neighbor. Transformasi ditolak bila target kosong, menyentuh tepi, kehilangan support atau jumlah voxel menyimpang >15% dari volume yang diharapkan setelah skala. Ini pemeriksaan operasional; overlay anatomi tetap perlu ditinjau.

Model dan loss mengikuti history asli. CT intensitas tetap memakai light gain/shift terdahulu. Klik disampling setelah transformasi agar sesuai mask. Tidak ada sharpening yang berpotensi mengubah batas anatomi. Validasi tidak diaugmentasi. train_28 dikeluarkan dari update dan kalibrasi lanjutan kedua arm. Reproduksi checkpoint sebelum update masih memakai54pasien sesuai protokol sumber; bobot sumber pernah melihat train_28. Angka awal49,47% dan skor kelanjutan setelah eksklusi tidak merupakan ablasi eksklusi. Kedua arm baru dapat dibandingkan langsung karena sama-sama memakai53pasien.

Masing-masing maksimal100epoch tambahan dengan early stopping30epoch tanpa peningkatan, LR multiplier0,2, resep dan accumulation dari fondasi. Perubahan dibanding eksperimen sebelumnya mencakup eksklusi dan kelanjutan; hanya affine berbeda antara dua arm baru. Hasil tetap segmentasi interaktif dengan GT ROI dan lima klik oracle, belum sistem otomatis dan tidak menjamin80%.

Di DGX:

```bash
cd ~/raid/Clara/fsdp-train
git pull origin codex/exclude-mislabeled-train28
conda activate fsdp-med3d
ALLOW_GPU_SHARING=1 GPU_ID=auto bash scripts/launch_fsdp_sam_augmentation.sh
tail -f logs/fsdp_sam_augmentation/latest.log
```

Default sumber: `runs/fsdp_loss_cohort_20261125T070213Z_3564637/control`. Jika dipindah, set `FSDP_AUG_SOURCE_DIR` ke direktori checkpoint yang sama. Trainer memeriksa hash, kecocokan implementasi, dan reproduksi Dice sumber sebelum update. Ringkasan akhir berada di `comparison.json` pada run yang tercatat dalam `logs/fsdp_sam_augmentation/latest_run.txt`. GPU terpilih memerlukan48GiBkosong; batas allocator PyTorch24GiB bukan reservasi GPU. Proses lain tidak dihentikan. Kode lokal belum dapat membuktikan peningkatan Dice karena checkpoint dan CT lengkap ada di DGX.

## Kompatibilitas checkpoint sumber

Pemeriksaan sumber mengenali dua revisi `train_sam_refine32.py` yang telah ditinjau (050cf6f dan14e6ecc) terhadap hash versi sekarang. Perubahan mencakup laporan serta hook opsional eksklusi dan full encoder; jalur default memakai partial encoder dan tidak mengecualikan pasien saat reproduksi sumber. Pengecualian hash hanya berlaku untuk pasangan hash sumber/versi sekarang yang dicatat dalam kode. Seluruh file model/data/metric lainnya harus identik. Rekaman kompatibilitas disimpan dalam config run baru, lalu hash checkpoint, strict state load dan reproduksi Dice train/val1/5klik serta fixed0,5tetap diperiksa sebelum update. Perubahan yang belum dikenal menghasilkan pesan dengan nama file dan kedua hash; tidak ada opsi melewati seluruh pemeriksaan.
