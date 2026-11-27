# Transfer sampling dan loss dari Brain CT Seg

Tujuan adalah meningkatkan volume Dice FSDP menuju 80%. Hasil belum tersedia. Pipeline ini menguji bagian pertama transfer metode Brain: minibatch slice dan loss focal–Dice–boundary. Backbone tetap residual U-Net lima slice yang sudah tersedia, dengan bobot baru acak. Konvolusi C8 menjadi percobaan berikutnya setelah hasil sampling/loss dapat diukur. Tidak ada bobot Brain atau SAM yang dimuat ke model ini.

## Dua arm yang dibandingkan

Control dan focal_boundary sama-sama memakai 53 pasien training, 10 validasi, native GT ROI, lima slice input, seed dan inisialisasi model yang sama. train_28 dikeluarkan. Semua center slice training dikelompokkan menurut ukuran XY native yang persis sama, lalu urutan slice dan minibatch diacak setiap epoch, termasuk slice tanpa foreground. Minibatch default empat stack, accumulation dua; target adalah mask slice tengah. Ukuran batch terakhir tiap kelompok bisa lebih kecil. Tidak ada padding tambahan untuk mencampur ukuran pasien yang berbeda; padding internal model ke kelipatan delapan sama saat training dan inference. Tidak ada pemilihan slice berdasarkan mask validasi. Prediksi validasi tetap mencakup semua slice dan direkonstruksi pada volume native.

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


## Hasil pertama dan koreksi batching

Run fsdp_brain_recipe_20261127T022501Z_3624684 selesai dengan control volume Dice 23,696% dan focal_boundary 22,540% (selisih −1,156 poin). Focal_boundary mencapai train Dice 35,188%, precision validasi 20,589%, recall 25,534%; kedua arm best epoch 15 dan berhenti pada 45. Surface Dice @1 mm sedikit naik dari 55,984% ke 57,256%, tetapi median HD95 memburuk dari 4,766 ke 5,067 mm. Percobaan ini tidak mendukung manfaat loss baru dalam pipeline yang dijalankan.

Ditemukan ketidaksesuaian implementasi: versi pertama mencampur ukuran XY native dengan zero padding ke ukuran terbesar dalam batch. Walau padding tidak masuk loss, konvolusi dan GroupNorm tetap membaca padding; inference per ROI tidak menyertakan padding tersebut. Uji sintetis model random kecil, seed 2026, input 24×32 dipad menjadi 48×64 mengubah rerata logit absolut 0,422 dan 19,531% mask pada threshold 0,5. Angka sintetis ini bukan estimasi dampak pada pasien atau bukti bahwa seluruh penurunan disebabkan padding.

Koreksi menggunakan exact-shape groups dan menolak campuran bentuk pada collator. Lima tes lokal lulus, termasuk kesamaan prediksi minibatch dan inference satu per satu tanpa augmentasi, cakupan setiap slice tepat sekali, serta gradien loss. Jumlah update dapat berubah karena sisa batch per kelompok; log mencatatnya. Run lama disimpan. Perbaikan hanya berlaku untuk run baru; resume dengan config lama akan ditolak oleh pemeriksaan config/code. Performa koreksi belum diukur di DGX. Kedua arm perlu dibandingkan pada pipeline yang sudah dikoreksi sebelum kesimpulan tentang loss atau C8.
