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

Launcher memeriksa paket lebih dulu, lalu berjalan lewat nohup. Bila belum ada GPU tanpa proses komputasi dengan setidaknya 48 GiB kosong, proses antrean memeriksa ulang setiap 120 detik sampai 24 jam tanpa menghentikan job lain. Antrean dapat dihentikan dengan PID pada `logs/fsdp_auto_nnunet/latest.pid`; pemeriksaan GPU hanya menjamin kondisi saat pemilihan, bukan reservasi sistem. Lokasi run tersimpan di `logs/fsdp_auto_nnunet/latest_run.txt`. Setelah kedua arm selesai, baca `comparison.json`, `predictions_2d/summary.json`, `predictions_3d_fullres/summary.json`, dan CSV per pasien. Prediksi NIfTI juga tersedia pada folder tiap arm untuk review anatomi.

Jika antrean lama, pilihan eksplisit `AUTO_GPU_ALLOW_SHARING=1` menerima GPU yang sudah punya proses komputasi asalkan saat dipilih tersedia minimal 48 GiB. Pilihan ini menjalankan train dan predict melalui pembungkus yang mengatur batas *PyTorch caching allocator* 24 GiB per proses dan meminta 16 GiB cadangan pada pemeriksaan awal. Batas allocator bukan reservasi GPU atau batas bagi semua alokasi CUDA, sehingga job lain yang tumbuh dapat tetap menyebabkan OOM atau memperlambat training. Tidak ada proses lain yang dihentikan. Jangan jalankan dua launcher otomatis bersamaan; hentikan antrean lama setelah memastikan PID-nya masih menjalankan `wait_fsdp_auto_gpu.sh`, lalu jalankan ulang:

```bash
cd ~/raid/Clara/fsdp-train
old_pid=$(cat logs/fsdp_auto_nnunet/latest.pid)
ps -p "$old_pid" -o args= | grep -F 'wait_fsdp_auto_gpu.sh' || exit 1
kill "$old_pid"
sleep 2
AUTO_GPU_ALLOW_SHARING=1 AUTO_GPU_MEMORY_BUDGET_GIB=24 GPU_ID=auto \
  bash scripts/launch_fsdp_auto_nnunet_nohup.sh
tail -f logs/fsdp_auto_nnunet/latest.log
```

Jika pemeriksaan PID tidak cocok, jangan kirim sinyal; baca log terbaru karena proses mungkin sudah masuk fase training. GPU yang dipilih dan PID aktif tercatat pada berkas `logs/fsdp_auto_nnunet/*.gpu.json`. Opsi tanpa sharing tetap menjadi default.

Pilihan model berikutnya harus didasarkan pada Dice per pasien, precision/recall, serta kasus dengan target terlewat. Jika kedua arm gagal menemukan struktur kecil pada CT utuh, lanjutkan localizer kasar dengan crop prediksi, lalu segmenter detail. Evaluasi lokalizer harus menghitung target di luar crop sebagai false negative. Test tetap belum dibaca.
