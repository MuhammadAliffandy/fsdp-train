# Transfer sampling dan loss dari Brain CT Seg

Tujuan adalah meningkatkan volume Dice FSDP menuju 80%. Hasil belum tersedia. Pipeline ini menguji bagian pertama transfer metode Brain: minibatch slice dan loss focal–Dice–boundary. Backbone tetap residual U-Net lima slice yang sudah tersedia, dengan bobot baru acak. Konvolusi C8 menjadi percobaan berikutnya setelah hasil sampling/loss dapat diukur. Tidak ada bobot Brain atau SAM yang dimuat ke model ini.

## Dua arm yang dibandingkan

Control dan focal_boundary sama-sama memakai 53 pasien training, 10 validasi, native GT ROI, lima slice input, seed dan inisialisasi model yang sama. train_28 dikeluarkan. Semua center slice training diacak lintas pasien setiap epoch, termasuk slice tanpa foreground. Minibatch default empat stack, accumulation dua; target adalah mask slice tengah. Bentuk native beragam dipad dan padding tidak berkontribusi pada loss. Tidak ada pemilihan slice berdasarkan mask validasi. Prediksi validasi tetap mencakup semua slice dan direkonstruksi pada volume native.

Control memakai Dice dan balanced BCE terdahulu pada minibatch slice baru. Arm focal_boundary memakai 0,5 focal gamma 2 + 2 per-slice Dice + 0,5 edge-weighted BCE. Edge weight 1 + 5 * boundary dinormalisasi berdasarkan jumlah weight valid; tidak memakai bobot positif 10× dari Brain. Boundary dibentuk dengan dilasi/erosi berbentuk persegi panjang, setengah lebar sekurangnya 1 mm per sumbu sesuai spacing XY. Bentuk ini bukan jarak Euclidean isotropik; resampling dan resolusi voxel membatasi ketepatannya. Deep supervision tetap 1/0,3/0,15. Augmentasi per stack berupa flip selaras dan gain/shift ringan. Normalisasi native ROI z-score sama pada kedua arm.

Ini membandingkan loss dalam resep baru. Dibanding hasil lama 27,05%, sampling, loss per slice dan LR berubah; tidak dapat mengatribusikan perubahan terhadap satu komponen. Jumlah update per epoch jauh berbeda dari training berbasis ROI lama dan dicatat eksplisit bersama jumlah slice positif/negatif. Lima slice tidak berarti lima pasien tambahan.

## Anggaran dan metrik

Maksimal 100 epoch per arm, patience 30, LR 0,0001, warmup/cosine sama, validasi epoch 1 dan setiap lima epoch. Checkpoint dipilih berdasarkan mean bilateral patient volume Dice. Threshold dipilih hanya pada training. Saat validasi, log menyimpan volume Dice per pasien, global Dice gabungan voxel, surface Dice @0,5/1 mm, dan median HD95 yang terdefinisi. Global Dice membantu membandingkan bentuk agregasi dengan Brain, tetapi tidak menggantikan macro per pasien. Surface Dice tinggi tidak menyatakan target volume Dice 80% tercapai. validation sudah ditinjau berulang, test belum dibaca.

Native raw CT, rekonstruksi mask, hash cache, split dan pemeriksaan ekspor mengikuti pipeline 2,5D. Protokol masih memakai GT ROI dan nol klik; belum otomatis menemukan FSDP pada CT utuh. SAM sekitar 50% tetap benchmark interaktif dengan lima klik, sehingga bukan pembanding satu variabel.

## Menjalankan di DGX

```bash
cd ~/raid/Clara/fsdp-train
git pull origin codex/exclude-mislabeled-train28
conda activate fsdp-med3d
ALLOW_GPU_SHARING=1 GPU_ID=auto bash scripts/launch_fsdp_brain_recipe_nohup.sh
tail -f logs/fsdp_brain_recipe/latest.log
```

Launcher membutuhkan GPU dengan 48 GiB kosong dan membatasi allocator PyTorch pada 24 GiB; pemeriksaan awal bukan reservasi. Kedua arm berjalan berurutan, tanpa menghentikan job lain. Lokasi run tercatat pada logs/fsdp_brain_recipe/latest_run.txt. comparison.json menampilkan semua metrik, selisih terhadap control, dan apakah target 0,8 tercapai. summary, history, checkpoint dan overlay tersedia per arm. BRAIN_EPOCHS, BRAIN_PATIENCE dan BRAIN_SLICE_BATCH dapat ditetapkan sebelum launcher; kedua arm memakai nilai yang sama.

Tes CPU lokal memeriksa center target, urutan tetangga, padding/support, gradien loss, dan slice kosong. Training lengkap memerlukan MONAI serta data/checkpoint DGX. Belum ada bukti peningkatan dari pipeline baru ini.
