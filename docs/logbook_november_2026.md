# Logbook FSDP 3D Segmentation — November 2026

Catatan harian eksperimen, diagnosis, dan keputusan untuk meningkatkan Dice segmentasi FSDP. Struktur tabel mengikuti logbook ColonoMind November 2026. Disusun pada 26 November dari laporan DGX yang diberikan pengguna, catatan proyek, dan pembacaan kode lokal. Waktu mulai run mengikuti timestamp UTC pada nama run, sedangkan tanggal log memakai Asia/Jakarta.

Ukuran keberhasilan utama adalah mean bilateral patient Dice pada grid asli. Evaluasi SAM memakai ROI yang ditentukan dari ground truth dan titik petunjuk yang dibentuk dari ground truth (oracle clicks). Satu atau lima klik diberikan per sisi, sehingga lima klik per sisi berarti sepuluh titik per pasien. Hasil tersebut merupakan evaluasi segmentasi interaktif dengan bantuan lokasi.

**Keputusan produk, 26 November.** Pengguna mengonfirmasi klien memerlukan segmentasi otomatis dari CT hingga mask FSDP. Evaluasi utama harus memakai nol klik pengguna/oracle dan ROI yang ditentukan sistem dari citra. Seluruh kasus validasi masuk perhitungan, termasuk kegagalan deteksi/lokalisasi. Skor interaktif historis dipakai sebagai benchmark tambahan dengan protokolnya dicantumkan.

## Log 23 November 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 23 November 2026. Jam kegiatan tidak tercatat lengkap. |
| **Rencana Harian** | Memeriksa hasil U-Net, mengukur karakteristik target kecil, dan mempertahankan detail melalui ROI fisik serta gabungan SAM-Med3D Turbo dan CNN. |
| **Aktivitas yang Dilakukan** | Mengaudit optimizer, gradien, AMP/FP32, retensi target, label roundtrip, dan threshold pada run kecil. Memeriksa 54 pasien train dengan 108 ROI sisi. Menyiapkan grid fisik dari data train serta normalisasi bersama. Mengembangkan hybrid Turbo dengan konteks 128³, token 8³, CNN fusion 32³, serta fitur full/half resolution. Menyelidiki kegagalan roundtrip dan fase origin resampling. |
| **Hasil/Capaian** | Run tiny U-Net terakhir mempunyai Dice 0 pada threshold 0,5 meskipun loss turun, sedangkan confusion matrix pada threshold 0,3 menunjukkan TP 613 untuk train_01 dan TP 391 untuk train_02. Angka 83,40% dan 87,47% pada baris foreground merupakan recall. Audit train menemukan spacing Z 0,25–2,5 mm dan target yang sangat jarang. Grid fisik terpilih 96×160×208 pada spacing 0,25 mm, dengan fase origin bersama 0,5. Foundation hybrid yang kemudian dipakai untuk continuation mencapai **45,63% validasi Dice@5 per sisi**, checkpoint terbaik epoch 30. |
| **Kendala/Masalah** | Downsampling, variasi geometri, foreground kecil, serta ketergantungan pada threshold perlu diukur. Pada train_26, fase origin awal menghasilkan roundtrip sekitar 91%, sedangkan fase alternatif memperbaikinya hingga lebih dari 99% pada diagnosis kasus tersebut. |
| **Solusi/Tindak Lanjut** | Memakai retensi target dan roundtrip sebagai pemeriksaan representasi. Melanjutkan dari checkpoint hybrid yang terverifikasi dan menilai prediksi setelah dikembalikan ke grid asli. |
| **Dokumentasi (Link/Ref)** | [PROJECT.md](../PROJECT.md), [physical_fsdp.py](../scripts/physical_fsdp.py), [sam_refine32.py](../scripts/sam_refine32.py). Foundation DGX `fsdp_private_sam_refine32_20261123T144159Z_3745292`. |

## Log 24 November 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 24 November 2026. Sebagian hasil dibaca pada 25 November. |
| **Rencana Harian** | Membandingkan perubahan detail/token dan menguji history prompt, boundary loss, serta refiner pada pipeline riset. |
| **Aktivitas yang Dilakukan** | Menyiapkan studi resolution baseline/detail/tokens/combined serta screening research control/history/boundary/refiner. Memeriksa ketersediaan GPU, proses anak, sumber data, dan cache. Beralih ke target biner cache yang hash dan integritasnya diverifikasi setelah folder anotasi asli dilaporkan kosong. |
| **Hasil/Capaian** | Screening research selesai 20 epoch tambahan per arm dengan Dice validasi control **47,75%**, history **49,18%**, boundary **47,51%**, dan refiner **47,20%**. Resolution baseline mencapai **46,51%**, selesai pada 70/100 epoch tambahan. Detail sempat tercatat 3 epoch dengan best sementara **38,04%**. Eksperimen detail64 multiclass yang dilaporkan mempunyai best macro sekitar **28,52%**, dengan FSDP epoch 110 sekitar **20,12%**. |
| **Kendala/Masalah** | Beberapa proses research/resolution berakhir dengan SIGTERM (-15). Pengirim sinyal belum diketahui. Percobaan multiclass/detail64 mengubah tugas dan protokol sekaligus, sehingga penurunan FSDP belum mengisolasi efek grid64. |
| **Solusi/Tindak Lanjut** | Mencatat kegagalan proses terpisah dari hasil model. Menyimpan baseline, memverifikasi sumber dan cache, serta memakai pembandingan dengan satu perubahan per arm untuk diagnosis berikutnya. |
| **Dokumentasi (Link/Ref)** | [Riset 24 November](FSDP_RESEARCH_2026-11-24.md), [PROJECT.md](../PROJECT.md). Screening DGX `fsdp_research_gpu7_20261124T132503Z_3250370`. |

## Log 25 November 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 25 November 2026. Run pengecualian train_28 dimulai sekitar 22.56 WIB berdasarkan timestamp UTC nama run. |
| **Rencana Harian** | Meningkatkan hasil cohort melalui history continuation, augmentasi ROI, diagnosis loss/gradien, serta baseline 3D otomatis. Menelusuri kasus dengan prediksi sangat buruk. |
| **Aktivitas yang Dilakukan** | Melanjutkan history dengan cap 100 epoch tambahan. Menguji jitter translasi fisik train-only dan kontrol sebanding. Menjalankan tiny-fit dua pasien pada loss control/balanced/auxiliary, kemudian full-cohort control dan balanced. Melatih Physical ResUNet dari inisialisasi acak dengan nol klik. Memeriksa sumber CT, cache, statistik intensitas, dan header. Menemukan input train_28 berisi label map dan menambahkan pengecualian eksplisit pada continuation. |
| **Hasil/Capaian** | History continuation berhenti pada 25/100 epoch tambahan dengan best epoch 0 dan Dice **49,18%**. Jitter mencapai **49,38%**, sekitar **+0,20 poin persentase** terhadap matched control. Tiny-fit loss mencapai mean Dice **86,69–87,41%** pada dua pasien train. Full-cohort loss control memilih **49,47%**, sedangkan balanced memilih checkpoint sumber **49,38%**. Pada pengukuran terakhir epoch 30, train/val control **59,00%/42,74%** dan balanced **59,11%/45,19%**. ResUNet mencapai train **65,51%**, validasi **33,27%**, best epoch 120, berhenti pada 180/300. |
| **Kendala/Masalah** | Perbaikan tiny-fit belum terbawa ke cohort. Train membaik sementara validasi melemah pada beberapa continuation. CT `6220173.nii.gz` (train_28) memiliki nilai 0–14 dengan hanya 10 nilai unik. Seluruh voxel RFSDP bernilai 1 dan LFSDP bernilai 9. Salinan processed imagesTr memiliki hash identik. Ini mendukung diagnosis label map yang masuk sebagai citra. Anotasi asli labelsTr/ColorTable belum tersedia untuk pemeriksaan semantik independen. |
| **Solusi/Tindak Lanjut** | Menyiapkan continuation 53 pasien dengan mengecualikan train_28 atas dasar input salah. Validasi tetap pada 10 pasien. Mencatat bahwa checkpoint warm start dan statistik preprocessing masih berasal dari cohort 54 pasien. Melakukan pencarian terbatas pada folder dataset dan proyek untuk CT pengganti, dengan hasil yang tersedia hanya salinan label map tersebut. |
| **Dokumentasi (Link/Ref)** | [Diagnosis sistem](FSDP_DIAGNOSIS_AND_DATASET_V1.md), [PROJECT.md](../PROJECT.md). ResUNet `physical_resunet_20261125T130044Z_641541`. Implementasi pengecualian commit `e940fa0`, branch `codex/exclude-mislabeled-train28`. |

## Log 26 November 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 26 November 2026, pembacaan hasil continuation 53 pasien dan penyusunan diagnosis lanjutan. |
| **Rencana Harian** | Mengevaluasi dampak continuation tanpa input train_28 dan menentukan percobaan berikutnya berdasarkan input, trainability encoder, dan kualitas anotasi. |
| **Aktivitas yang Dilakukan** | Membaca summary DGX, membandingkan skor dengan foundation, memeriksa kembali normalisasi dan set_stage pada kode, serta menyusun matriks diagnosis di bawah. |
| **Hasil/Capaian** | Run selesai pada **55/100 epoch tambahan** melalui early stopping. Best epoch tambahan **5**, setara epoch sumber 30 + 5. Threshold train-only **0,45**. Dice validasi bilateral **48,09%** dengan lima klik/sisi dan **39,16%** dengan satu klik/sisi. Dice train checkpoint terpilih **53,25%**. Validasi fixed threshold 0,5 **47,43%**, precision pada threshold terpilih **46,34%**, recall **52,08%**. Minimum bilateral patient Dice **36,06%**, minimum side Dice **28,97%**. Selisih train–val checkpoint terpilih **5,16 poin persentase**. |
| **Kendala/Masalah** | Validasi naik **2,46 poin** terhadap foundation 45,63%, dan fixed-0,5 naik **4,35 poin** dari 43,08%. Hasil sekitar **1,38 poin** di bawah benchmark loss control 49,47% yang memakai resep/cohort berbeda. Pengecualian dan training tambahan terjadi bersama, sehingga kontribusi pengecualian belum terisolasi. Summary checkpoint terbaik belum menunjukkan perjalanan Dice train hingga epoch 55. |
| **Solusi/Tindak Lanjut** | Memprioritaskan diagnosis D1 normalisasi, D2 trainability encoder, D3 konsistensi target. Menentukan input dan kasus yang perlu diperbaiki sebelum menetapkan training berikutnya. |
| **Dokumentasi (Link/Ref)** | Laporan DGX `fsdp_private_foundation_20261125T155656Z_1188354/summary.json`, checkpoint SHA256 `762fb86726ec216f4dab65ad992160ffa9f7197c94d2c130057925084ecaf945`. [Trainer foundation](../scripts/train_fsdp_foundation.py), [trainer hybrid](../scripts/train_sam_refine32.py). |

Lokasi artefak DGX hasil terbaru:

```text
/raid/D13K48009/Clara/fsdp-train-train28fix/runs/fsdp_private_foundation_20261125T155656Z_1188354/
  summary.json
  history.csv
  preflight.json
  hybrid_best.pth
  best_exports/train_5_patients.csv
  best_exports/train_5_sides.csv
  best_exports/val_5_patients.csv
  best_exports/val_5_sides.csv
  best_exports/overlays/
  results.tar.gz
```

## Diagnosis untuk menentukan percobaan berikutnya

Status per 26 November: bukti laporan dan kode sudah ditinjau. Pengukuran D1/D2/D3 yang memerlukan tensor, checkpoint, atau pemeriksaan anatomis di DGX masih menunggu pelaksanaan. Catatan ini menyusun diagnosis dan rancangan pembandingan.

### D1 — Informasi konteks berkurang akibat normalisasi

**Pertanyaan.** Apakah clipping berbasis intensitas foreground membuang variasi konteks yang dibutuhkan untuk membedakan FSDP dari area sekitarnya?

**Temuan yang sudah ada.** physical_policy memakai clip_lower −1343, clip_upper 136,84, mean −700,42, std 297,14, ditaksir dari target train 54 pasien. Nilai di atas 136,84 dipetakan ke nilai maksimum yang sama. Pada train_01, median input valid sekitar 2,48 dan p95/max sekitar 2,82, sedangkan raw context p95 sekitar 912–933. Pada train_53 raw context p95 sekitar 1050 dan input p95 juga mencapai maksimum. Saturasi input telah terlihat, tetapi pengaruhnya terhadap Dice belum diisolasi. Kalibrasi HU fisik perlu dikonfirmasi sebelum menetapkan jendela HU sebagai kebijakan.

**Pengukuran yang diperlukan.**

1. Hitung fraksi clipping bawah/atas untuk foreground dan background pada detail serta SAM context, per pasien/sisi. Mulai dari 53 train yang valid, lalu terapkan kebijakan train pada 10 val.
2. Catat sebaran intensitas raw, satuan/scaling sumber, spacing, dan korelasinya dengan TP/FP/FN serta Dice. Korelasi digunakan untuk mencari kasus pemeriksaan.
3. Periksa distribusi aktivasi patch embedding dan beberapa blok encoder pada citra yang sama dengan normalisasi lama dan kandidat.
4. Bila membuat kandidat baru, hitung statistik dari 53 train. Samakan ROI, target, grid, support, dan koordinat klik dengan kontrol. Regenerasikan kedua tampilan CT dari sumber raw yang terverifikasi.

**Rancangan percobaan.** Kontrol A memakai input lama pada 53 train. Kandidat B memakai satu kebijakan clipping/normalisasi yang dipilih dari audit train. Keduanya memakai checkpoint awal, encoder aktif, loss, seed, optimizer, jadwal dan budget yang sama. Evaluasi epoch 0 mengukur dampak langsung perubahan input, lalu evaluasi setelah adaptasi mengukur hasil fine-tuning. Riwayat klik oracle adaptif boleh berbeda karena prediksi berbeda, dengan aturan sampling dan seed tetap sama.

**Keputusan.** Prioritaskan kandidat bila validasi bilateral meningkat terhadap matched control, dengan evaluasi fixed-0,5, TP/FP/FN, dan konsistensi perubahan pasien. Ukur ulang konfigurasi yang menjanjikan pada seed tambahan sebelum menyatakan perbaikan stabil. Median train dan kasus terburuk membantu membaca kegagalan.

**Kebutuhan implementasi.** [diagnose_fsdp_system.py](../scripts/diagnose_fsdp_system.py) sudah memprofilkan clipping dan menyediakan kandidat konteks −1000 sampai 2000. Pembentuk kandidat tersebut masih mengolah cache penuh, termasuk train_28, sehingga statistik kandidat 53 pasien perlu dukungan exclusion terlebih dahulu. Kandidat itu belum layak dipromosikan langsung sebagai input run 53 pasien.

### D2 — Ruang adaptasi encoder Turbo terbatas

**Pertanyaan.** Apakah pembekuan sebagian besar encoder membuat fitur terlalu sulit beradaptasi terhadap citra dan target FSDP?

**Temuan yang sudah ada.** set_stage di train_sam_refine32.py melatih image_encoder.blocks.10/11 dan mask_decoder setelah warmup. Prompt encoder dan blok encoder sebelumnya dibekukan. Modul CNN/refiner di luar SAM dilatih. Token SAM tetap 8³, dengan fusion CNN32 dan skip full/half resolution. Pretraining telah dimuat secara strict. Kecukupan fitur awal untuk target ini belum diukur.

**Pengukuran yang diperlukan.**

1. Catat requires_grad dan jumlah parameter aktif setiap kelompok dari model runtime. Periksa mode train/eval per modul.
2. Ukur variasi spasial dan nilai finite aktivasi patch embedding, blok awal/tengah/akhir, neck, coarse mask, dan keluaran refiner pada pasien mudah/sulit.
3. Ukur norma gradien, frekuensi clipping, dan perubahan bobot relatif setelah update pada kelompok aktif. Parameter beku memang diharapkan tidak memiliki gradien parameter.
4. Bandingkan besaran gradien loss detail dan auxiliary context memakai laporan diagnosis yang sudah ada. Komponen coarse pada click trace hybrid hanya merupakan pembacaan jalur internal model.

**Rancangan percobaan.** Setelah kebijakan input ditetapkan, kontrol A melatih blok 10/11 + mask decoder + CNN. Kandidat B membuka satu kelompok tambahan, misalnya blok 8/9, dengan LR encoder yang sama untuk blok yang sejenis. Arsitektur/token, input, loss, klik, inisialisasi dan budget update dipasangkan. Tahap ini menguji cakupan encoder aktif. Jika LR perlu diteliti, lakukan pembandingan LR berikutnya secara tersendiri.

**Keputusan.** Kandidat layak diteruskan bila kelompok tambahan menerima update finite dan ada kenaikan Dice validasi yang konsisten terhadap kontrol. Bila Dice train naik dan validasi turun, tinjau regularisasi, jadwal, dan kestabilan pembandingan. Efek membuka encoder belum terukur dalam run terakhir.

### D3 — Variasi citra dan konsistensi batas anotasi

**Pertanyaan.** Apakah kasus sulit memiliki karakteristik citra tertentu atau batas target yang diterapkan berbeda antar pasien?

**Temuan yang sudah ada.** train_28 merupakan satu input salah yang terkonfirmasi. Source imagesTr dan processed imagesTr memiliki SHA256 `14a0b2e7e93cbc8d5f6d64fedf7364bd56074f37a57d382907e1f3b91c8744d1`. Keutuhan binary cache telah diperiksa, sedangkan original label dan ColorTable belum dapat ditinjau ulang. Uji tiny yang berhasil memakai train_01/02, sehingga bukti tersebut mencakup dua pasien. Spacing Z seluruh train bervariasi 0,25–2,5 mm. Roundtrip tinggi mengukur keutuhan representasi label, sementara ketepatan anatomis memerlukan pemeriksaan ahli.

**Pengukuran yang diperlukan.**

1. Gabungkan skor per pasien/sisi dengan native spacing, ketebalan irisan, volume target, clipping dan volume prediksi. Tinjau kasus terburuk serta kasus baik sebagai pembanding.
2. Bandingkan skor train_01/02 pada tiny-fit dan cohort menggunakan model, input, jumlah klik, dan threshold yang disejajarkan. Selisih protokol harus dicatat sebelum menginterpretasikan penurunan.
3. Buat overlay axial/coronal/sagittal pada native CT untuk memeriksa lokasi, orientasi, batas, target terlewat, dan area prediksi berlebih. Gunakan jarak permukaan dalam mm sebagai pelengkap Dice volume. Surface Dice@1 mm dan volumetric Dice dicatat sebagai metrik berbeda.
4. Minta peninjau anatomi memberi alasan untuk kasus yang dicurigai. Bila memungkinkan, dua peninjau menandai ulang subset train yang sudah ditetapkan untuk mengukur kesepakatan batas. Dataset hasil revisi memerlukan versi dan provenance baru.

**Rancangan percobaan.** Jika ditemukan kesalahan sumber atau anotasi yang terkonfirmasi, siapkan versi data dengan koreksi tersebut dan dokumentasikan dampak terhadap pembandingan. Jika kasus sulit mengelompok menurut spacing/volume dan anotasi dinilai konsisten, uji satu perubahan sampling per pasien/target pada 53 train dengan input dan arsitektur terpilih. Keputusan sampling memakai karakteristik train. Temuan validation dilaporkan sebagai analisis eksploratif.

**Keputusan.** Bukti anatomi menentukan revisi target. Statistik kesulitan train menentukan kandidat sampling. Skor model yang rendah sendiri belum cukup untuk menetapkan bahwa label perlu diubah.

## Urutan dan catatan keputusan

| Langkah | Kegiatan | Keluaran yang diperlukan | Status 26 November |
|---|---|---|---|
| 1 | Baca history, per-pasien, dan overlay dari run 53 pasien serta benchmark | Train/val pada epoch yang sama, TP/FP/FN, kasus turun/naik, konfigurasi evaluasi | Summary tersedia, rincian artefak DGX perlu dibaca |
| 2 | Jalankan audit D1 dan siapkan review D3 | Tabel clipping, spacing, volume, overlay, unit intensitas | Bukti awal ada, pengukuran lengkap masih diperlukan |
| 3 | Tentukan satu kandidat input dan jalankan matched control | Kontrol 53 train dengan provenance serta kandidat input 53 train | Rancangan dicatat, training belum dimulai |
| 4 | Ukur D2 pada input terpilih dan uji tambahan blok encoder | Aktivasi, gradien/update per kelompok, hasil pasangan | Rancangan dicatat, training belum dimulai |
| 5 | Tindak lanjut hasil review D3 | Revisi data yang terkonfirmasi atau satu kandidat sampling | Menunggu pemeriksaan kasus dan anotasi |

Untuk setiap percobaan, catat run ID, checkpoint awal, hash sumber/cache/manifest, daftar pasien aktif, kebijakan input, jumlah klik, parameter aktif, seed, jumlah update optimizer, jadwal LR, early stopping, dan checkpoint terbaik. Laporkan Dice train/val pada checkpoint yang sama, fixed threshold 0,5, threshold yang dipilih dari train, hasil per pasien, serta perbandingan terhadap kontrol yang sebanding.

Target klien sekitar 80% Dice dicatat sebagai sasaran. Bukti terbaru berada pada 48,09% untuk continuation 53 pasien dan 49,47% untuk benchmark resep sebelumnya. Protokol penggunaan klien perlu menetapkan ROI dan titik petunjuk yang tersedia saat pemakaian. Validation 10 pasien sudah dipakai berulang selama pengembangan, sehingga hasil eksperimen perlu disertai variasi per pasien dan keterbatasan pemilihan model. Laporan terbaru menyatakan test_images_read=false. Pengujian akhir mengikuti protokol yang ditetapkan setelah pemilihan resep.

## Cara menambah catatan harian

Tambahkan entri tanggal baru dengan tujuh baris tabel yang sama. Nyatakan status aktivitas secara konkret, misalnya dirancang, diimplementasikan, dijalankan di DGX, terhenti, atau selesai. Isi Hasil/Capaian dari artefak run yang tersedia dan hubungkan keputusan dengan hasil tersebut. Tambahkan hasil D1/D2/D3 pada tanggal penerimaan hasil agar hipotesis awal dan hasil diagnosis dapat ditelusuri.

## Pembaruan 26 November — implementasi adaptasi seluruh image encoder

Pengguna memprioritaskan penyesuaian representasi pretrained Turbo terhadap dataset FSDP. Opsi `--full-image-encoder` pada trainer foundation membuka seluruh image encoder, termasuk patch embedding, positional embedding, semua blok transformer, dan neck. Mask decoder serta CNN refiner ikut dilatih, dengan prompt encoder beku. Launcher meneruskan opsi melalui `FSDP_FULL_IMAGE_ENCODER=1`.

Pembandingan dimulai dari foundation asli 45,63% yang juga menjadi sumber continuation 53 pasien. Kebijakan input, token8/CNN32, loss, klik, seed, dan jadwal mengikuti kontrol continuation tersebut. LR dasar SAM mengikuti multiplier 0,2 dari resep sumber, umumnya 2e-6, dan LR CNN 2e-5, dengan warmup/cosine yang sudah ada. Ini merupakan percobaan D2 terhadap kontrol dengan input sama. Saturasi input D1 masih memerlukan diagnosis dan pembentukan kandidat 53 pasien.

Run full encoder mencatat `encoder_trainability.json` untuk jumlah parameter aktif per kelompok. Dua update pertama pada epoch awal continuation mencatat gradien patch embedding, blok encoder, neck, serta kelompok SAM/CNN. Tahap di history bernama `full_image_encoder_and_refiner`. Pengecualian train_28 tetap diberikan secara eksplisit.

Pemeriksaan lokal memakai fungsi produksi set_stage dan gradient_groups pada jaringan PyTorch kecil. Mode partial dan full lulus pemeriksaan parameter aktif, train/eval, backward finite, serta update AdamW pada parameter yang seharusnya dilatih. Pemeriksaan juga memastikan prompt tetap beku, positional embedding menerima gradien pada mode full, semua 12 blok mempunyai gradien, dan pengecualian alias bekerja. Sintaks Python/shell dan whitespace diperiksa. Eksekusi model Turbo penuh, kebutuhan VRAM H100, reproduksi checkpoint pada CT asli, dan perubahan Dice belum diukur. Lingkungan lokal belum memiliki MONAI sehingga pemeriksaan lokal ini terbatas pada fungsi training terkait.

Command DGX berikut memakai worktree yang sudah dibuat sebelumnya. GPU_ID harus menunjuk GPU yang tersedia saat dijalankan.

```bash
cd ~/raid/Clara/fsdp-train-train28fix
git fetch origin codex/exclude-mislabeled-train28
git merge --ff-only origin/codex/exclude-mislabeled-train28
conda activate fsdp-med3d

GPU_ID=7 \
FSDP_FOUNDATION_DIR="/raid/D13K48009/Clara/fsdp-train/runs/fsdp_private_sam_refine32_20261123T144159Z_3745292" \
FSDP_SOURCE_ROOT="$HOME/raid/normal case_86" \
FSDP_VERIFIED_CACHE=1 \
FSDP_EXCLUDE_TRAIN_ALIASES=train_28 \
FSDP_FULL_IMAGE_ENCODER=1 \
ADDITIONAL_EPOCHS=100 \
bash scripts/launch_fsdp_foundation_nohup.sh

tail -n 80 -f logs/fsdp_foundation/latest.log
```

## Pembaruan 26 November — hasil adaptasi seluruh image encoder

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 26 November 2026. Nama run mencatat mulai 03.22 UTC, setara 10.22 WIB. Waktu selesai belum dicatat pada laporan yang diberikan. |
| **Rencana Harian** | Menguji apakah pembukaan seluruh image encoder meningkatkan pembelajaran dan validasi pada cohort FSDP 53 pasien. |
| **Aktivitas yang Dilakukan** | Pengguna menjalankan full_image_encoder dengan train_28 dikecualikan, warm start foundation asli 45,63%, normalisasi lama, LR SAM 2e-6, dan LR CNN 2e-5. Seluruh image encoder, mask decoder dan CNN dilatih dengan prompt encoder beku. |
| **Hasil/Capaian** | Run complete=true, early stopping pada **60/100 epoch tambahan**, best epoch tambahan **25**, threshold train-only **0,5**. Dice train@5 **57,98%**, validasi@5 **46,45%**, validasi@1 **38,68%**. Precision validasi **47,30%**, recall **47,28%**, minimum bilateral patient Dice **32,44%**, minimum side Dice **27,25%**. Selisih train–validation checkpoint terbaik **11,53 poin persentase**. |
| **Kendala/Masalah** | Terhadap kontrol partial encoder 53 pasien, Dice train checkpoint terbaik naik **4,73 poin**, sedangkan validasi turun **1,64 poin**. Pada threshold tetap 0,5, validasi turun **0,98 poin**. Dua checkpoint dipilih pada epoch berbeda. Hasil satu seed ini mendukung kecenderungan peningkatan fitting train dengan generalisasi lebih lemah pada resep yang diuji. Kurva train/validation lengkap dan perubahan tiap pasien masih perlu diperiksa sebelum menetapkan pola sepanjang training. |
| **Solusi/Tindak Lanjut** | Hipotesis bahwa membuka encoder saja cukup untuk meningkatkan Dice validasi belum didukung. Prioritaskan audit input D1 dan review spasial D3. Pertahankan kontrol partial encoder dalam pembandingan kandidat normalisasi 53 pasien. Catat penetapan checkpoint, LR, gradien awal, clipping, dan prediksi per pasien untuk menentukan percobaan berikutnya. |
| **Dokumentasi (Link/Ref)** | DGX `fsdp_private_foundation_20261126T032200Z_3059519/summary.json`. Checkpoint SHA256 `9a8515abeb3ad828b87e463611beb43093bafd84c717f9bdaed8d563a836727d`. Artefak pada `/raid/D13K48009/Clara/fsdp-train-train28fix/runs/fsdp_private_foundation_20261126T032200Z_3059519`. |

| Ukuran pada checkpoint terpilih | Partial encoder, 53 train | Full image encoder, 53 train | Selisih full − partial |
|---|---|---|---|
| Train Dice@5 | 53,25% | 57,98% | +4,73 poin |
| Validation Dice@5, threshold train-only | 48,09% | 46,45% | −1,64 poin |
| Validation Dice@5, threshold tetap 0,5 | 47,43% | 46,45% | −0,98 poin |
| Validation Dice@1 | 39,16% | 38,68% | −0,48 poin |
| Selisih train–validation@5 | 5,16 poin | 11,53 poin | +6,37 poin |
| Best epoch tambahan | 5 | 25 | Checkpoint berbeda |

Normalisasi pada laporan full encoder tetap clip −1343 hingga 136,84 dengan mean/std sumber 54 pasien. Dampak saturation D1 dan variasi batas D3 terhadap hasil belum diisolasi oleh percobaan ini. Coarse SAM pada trace klik hybrid mencapai 25,37% validasi@5, tetapi nilai ini memakai interaksi yang diarahkan keluaran hybrid dan belum merupakan pembandingan SAM mandiri. Bukti gradien/parameter aktif DGX belum disertakan dalam summary yang diberikan. Target 80% belum dapat dijanjikan berdasarkan hasil ini.

## Pembaruan 26 November — sasaran segmentasi otomatis dari CT

Pengguna menetapkan segmentasi otomatis sebagai kebutuhan klien. Tujuan operasional adalah CT utuh menjadi mask bilateral FSDP pada geometri asli tanpa titik pengguna maupun titik yang diturunkan dari ground truth. ROI saat evaluasi ditentukan dari citra dan keluaran model. Label tersedia untuk supervisi train dan penilaian prediksi.

Penyesuaian interpretasi eksperimen terdahulu:

- Benchmark SAM 45–49% memakai bantuan ROI ground truth dan lima klik oracle per sisi. Hasil ini mengukur segmentasi interaktif dengan bantuan lokasi.
- Baseline Physical ResUNet 33,27% memakai nol klik dengan ROI ground truth. Kinerja deteksi lokasi pada CT utuh belum diukur oleh run itu.
- Target 80% selanjutnya harus dilaporkan sebagai Dice pada prediksi otomatis CT utuh, dengan denominator kasus dan protokol pemilihan model dinyatakan.

Rancangan berikutnya memakai dua tahap yang perlu diimplementasikan dan diuji secara terpisah sebelum penilaian gabungan:

1. **Lokalisasi dari CT.** Model pada citra beresolusi lebih rendah memprediksi wilayah kandidat kanan/kiri. Target supervisi train dapat disusun dari binary native cache bila cakupan target dan geometri seluruh CT berhasil diverifikasi. Laporkan retensi target dalam crop prediksi, kegagalan deteksi per sisi/pasien, dan volume crop pada validasi. Ukuran/margin/fallback ditentukan dari train.
2. **Segmentasi detail otomatis.** Model detail dilatih pada crop train yang mencakup variasi posisi dan kesalahan lokalisasi. Jika crop prediksi train dipakai, prediksi out-of-fold dapat mengurangi perbedaan tingkat kesulitan dengan deployment. SAM dapat dieksplorasi dengan prompt prediksi dari localizer, atau model nol klik dapat dipakai. Pemilihan decoder mengikuti hasil pembandingan pada input yang sama.
3. **Penilaian CT utuh.** Prediksi dikembalikan ke native CT dan dibandingkan dengan seluruh target. Target di luar crop ikut menjadi false negative. Kegagalan lokalisasi dicatat dan masuk rata-rata semua pasien. Penilaian segmenter dengan ROI ground truth membantu diagnosis kemampuan segmentasi.

Kebutuhan sebelum implementasi training berikutnya adalah artefak partial/full encoder (config, history, preflight, ekspor skor dan overlay), laporan intensitas D1, serta akses CT train/validation dan binary target cache terverifikasi pada DGX. Masalah normalisasi dan anotasi D1/D3 tetap ditinjau. Data sumber 53 train yang valid dan 10 validation tersedia menurut laporan sebelumnya, sedangkan target cache perlu diverifikasi mampu merepresentasikan seluruh target pada volume CT. Original labels/ColorTable belum tersedia untuk validasi semantik independen.

Status saat keputusan dicatat: rancangan otomatis disusun, localizer dan evaluasi gabungan CT utuh belum dijalankan. Pemilihan backbone, jendela intensitas, resolusi localizer dan crop detail masih menunggu audit train. Protokol akhir ini menjadi acuan interpretasi target klien.

## Pembaruan 26 November — pembacaan paket diagnosis lokal

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 26 November 2026, setelah pengguna mengunduh dan mengekstrak fsdp_diagnostic_bundle.tar.gz ke proyek. |
| **Rencana Harian** | Menguji hipotesis D1/D2/D3 menggunakan history lengkap, skor per pasien/sisi, parameter aktif, probe gradien, metadata input dan overlay. |
| **Aktivitas yang Dilakukan** | Membaca dua run 53 train/10 validation, menghitung ulang Dice dari TP/volume, memverifikasi pasangan sisi dan pasien, membandingkan resep serta kurva lengkap. Meninjau diagnosis 128 ROI/64 header dan delapan PNG. Pengguna menegaskan target adalah seluruh volume anatomi FSDP. Membuat analisis JSON/CSV dan grafik lokal. |
| **Hasil/Capaian** | Full encoder akhir epoch 60 mencapai train **63,65%**, validasi **41,89%**. Partial akhir epoch 55 train **60,31%**, validasi **42,34%**. Seluruh 12 blok menerima gradien finite pada dua probe awal full. Median clipping detail **19,82% train/21,16% validation**, SAM context **15,21%/15,63%**. Perbandingan best meningkatkan 5 pasien dan menurunkan 5 pasien, dengan val_06 turun **14,18 poin**. |
| **Kendala/Masalah** | Kurva mendukung overfitting pada kedua resep. Beberapa prediksi memiliki volume hampir benar tetapi overlap rendah, sehingga kalibrasi volume belum cukup. Overlay memperlihatkan bagian tipis yang terlewat atau posisi/bentuk yang tidak sesuai pada irisan yang dilihat. Kandidat normalisasi lama memakai 54 pasien dan batas bawah −1000 meskipun median raw target train_01 berada di bawah −1000. Bukti kesetaraan grid kandidat lama mencatat counter 0. Paket belum memuat CT, checkpoint, ataupun mask volumetrik. |
| **Solusi/Tindak Lanjut** | Menentukan ulang kandidat normalisasi dari 53 train, memverifikasi seluruh grid/target/support, serta melengkapi error spasial native di DGX. Memprioritaskan pipeline otomatis localizer–segmenter sesuai kebutuhan klien. Model interaktif tetap menjadi benchmark diagnosis. Definisi volume pengguna menjadi acuan review anotasi. |
| **Dokumentasi (Link/Ref)** | [Diagnosis paket DGX](FSDP_BUNDLE_DIAGNOSIS_2026-11-26.md). Artefak lokal di outputs/private/diagnostic_review_20261126/. |

## Pembaruan 26 November — baseline otomatis dengan epoch sebanding

| Parameter | Deskripsi |
|---|---|
| **Rencana** | Membandingkan segmentasi FSDP otomatis dari CT utuh pada 2D dan 3D dengan 100 epoch per arm. |
| **Implementasi** | Menambahkan ekspor CT dan target full volume dari cache terverifikasi, pengecualian train_28, pemeriksaan hash/geometri/foreground, training nnU-Net 2D dan 3D berurutan, inferensi sepuluh CT validasi utuh, dan skor native per pasien. |
| **Protokol** | Kedua arm memakai 53 train yang sama, 10 validation eksternal, epoch 100, normalisasi z-score per pasien, tanpa GT ROI maupun klik saat inferensi. Jumlah update optimizer dan komputasi antar arm belum dipasangkan. Test tidak dipakai. |
| **Status** | Kode dan launcher disiapkan lokal. Eksekusi DGX, hasil Dice, kebutuhan VRAM/disk aktual, dan kapasitas target 80% belum diketahui. Paket diagnosis lokal tidak memuat CT maupun checkpoint untuk menjalankan workflow di laptop. |
| **Dokumentasi** | [Panduan baseline otomatis](FSDP_AUTOMATIC_BASELINE_2026-11-26.md), [ekspor dan skor](../scripts/fsdp_auto_nnunet.py), [launcher](../scripts/launch_fsdp_auto_nnunet_nohup.sh). |

GPU DGX belum tersedia saat percobaan peluncuran. GPU 2 melaporkan 73,5 GiB bebas tetapi memiliki PID komputasi aktif, dan pemilih tidak menemukan GPU yang memenuhi syarat tanpa berbagi. Launcher kemudian diperbarui agar mengantre melalui nohup, memeriksa ulang tiap 120 detik sampai 24 jam, serta mulai sendiri bila GPU tanpa proses komputasi dengan 48 GiB bebas tersedia. Tidak ada job lain dihentikan; training belum tercatat mulai pada saat pembaruan ini.

## 26 November 2026 — augmentasi dari benchmark SAM49,47%

Disiapkan kelanjutan SAM CNN32/Turbo8 history dari loss-control terbaik, dengan control dan affine terpisah pada sumber yang sama. Tambahan affine berupa rotasi±5° dan skala0,95–1,05 pada kedua view CT/mask/support dalam koordinat fisik. train_28 dikeluarkan dari kedua arm setelah reproduksi checkpoint sumber. Maksimal100epoch tambahan/arm, patience30. Evaluasi tetap GT-assisted1/5klik,10pasienval,testbelumdibaca. Status: implementasi lokal; training dan hasil perbaikan belum dijalankan dari sesi ini. Panduan: FSDP_SAM_AUGMENTATION_2026-11-26.md.

Perbaikan startup augmentasi: pemeriksaan checkpoint lama menolak hash trainer yang berubah sejak eksperimen sumber. Ditambahkan kompatibilitas terbatas untuk dua revisi trainer yang ditinjau, dengan hash versi saat ini juga dikunci; file model/data/metric lainnya tetap harus sama. Tiga pengujian isolasi pemeriksaan hash lulus. Suite continuation penuh belum bisa dijalankan pada interpreter lokal karena MONAI tidak tersedia. Reproduksi checkpoint/data tetap merupakan pemeriksaan DGX sebelum update pertama.

Hasil augmentasi yang dilaporkan: control49,592%, affine49,983%, kenaikan+0,391poin terhadap control; affine berhenti35epoch, bestepoch1. Belum menunjukkan peningkatan besar. Disiapkan `audit_fsdp_sam_results.py` untuk membandingkan FP/FN, rasio volume, metrik permukaan dan skor per pasien/sisi dari ekspor yang sudah selesai. Dua tes CPU dengan fixture sintetis lulus; hasil audit pasien sebenarnya masih perlu diambil dari DGX. Surface Dice1mm88,11% tidak disamakan dengan volume Dice80% atau diagnosis penyebab.

## 26 November 2026 — pencatatan dua Dice dan percobaan 2,5D

SAM affine mencatat volume Dice **49,983%**, surface Dice @0,5 mm **78,568%**, surface Dice @1 mm **88,110%**, dan median HD95 **2,414 mm**. Control mencatat volume Dice **49,592%**. Surface Dice tidak disamakan dengan target volume Dice 80%.

Ditambahkan percobaan satu versus lima slice pada ROI raw_native: 53 pasien training, 10 validasi, nol klik, bobot 2D baru, maksimal 100 epoch per arm, patience 30. Setiap evaluasi validasi mencatat volume Dice, surface Dice @0,5/1 mm dan HD95. Tiga tes model CPU lulus; training DGX belum dijalankan dari sesi ini. Ini studi konteks slice, dengan GT ROI. Rincian: FSDP_25D_2026-11-26.md.

## 27 November 2026 — menelaah Brain CTC Seg untuk transfer metode

Sumber lokal yang dibaca: `../brain-ctc-seg/training/train_se2_by_dataset.py`, `training/train_ablation.py`, `paper_evalute/evaluate_boundary_metrics.py`, ringkasan CT/CTC dan arsip log Context_5D. Tidak ada file proyek Brain yang diubah atau checkpoint yang dijalankan.

**Bukti hasil:** ringkasan `data_summaries/ctc_summary.csv` mencatat best Dice CTC 97,30% pada epoch 140; CT biasa mencatat 82,04%. Arsip `Context_5D_ct_20260924_213531.txt` memuat Val Dice 91,76%, sesuai baris Context_5D pada ablation_metrics.csv. Ini bukti hasil yang dilaporkan, belum reproduksi checkpoint. CSV ablation lama memiliki HD95 486,001 dan ASSD 255,5084 yang sama pada semua baris; kolom jarak tersebut tidak dipakai sebagai dasar transfer. Skrip boundary lama memakai spacing default 1 dan toleransi 2, sehingga tidak setara dengan evaluasi fisik FSDP.

**Teknologi:** model Brain memakai U-Net dengan ESCNN R2Conv dan grup rotasi C8 (`N=8`), regular representations, empat tahap downsampling dan skip decoder. Loader menyusun tiga atau lima slice dan memprediksi mask slice tengah; inisialisasi random. Loss ablation menggabungkan 0,5 focal + 2 Dice + 0,5 edge-weighted cross entropy. Edge mask berasal dari dilasi/erosi kernel 5, cross entropy dibobot 1 + 5 * edge; class weights [1,10]. Normalisasi min-max per stack; training memakai minibatch slice, augmentasi rotasi dan sebagian konfigurasi CLAHE. Varian lima slice menonaktifkan CLAHE karena keterbatasan kanal.

**Perbedaan FSDP:** model lima slice yang mendapat 27,05% memakai Conv2D residual biasa, tiga tahap downsampling, GroupNorm, dan Dice + balanced BCE dengan deep supervision. Gradien mengagregasi seluruh volume ROI/patch, bukan minibatch slice acak seperti Brain. Input memakai native CT z-score. Metode baru ini belum mengadopsi C8, focal dan edge-weighted loss Brain.

**Perbedaan evaluasi:** kode Brain menjumlahkan TP/FP/FN seluruh slice menjadi global Dice. FSDP memakai mean bilateral patient Dice pada volume native. Global Dice lebih dipengaruhi pasien/lesi dengan denominator besar; keduanya perlu dilaporkan bersama untuk membandingkan, tanpa mengganti metrik utama FSDP. Split Brain didasarkan pada baris DataFrame pasien; keunikan identitas tetap perlu dibuktikan dari manifest sebelum klaim disjointness.

**Rencana transfer yang layak:** (1) laporkan global Dice serta macro per pasien dari prediksi FSDP yang sama; (2) uji minibatch slice dan pengawasan slice tengah dengan negative slices tetap hadir; (3) pada backbone tetap, bandingkan loss lama dengan focal + Dice + edge weighting yang dinormalisasi dan mempertimbangkan spacing; (4) setelah itu uji backbone C8 dengan budget memori yang diukur. Weight positif 10, CLAHE, sharpening dan deformasi tidak dipindahkan sekaligus karena precision FSDP hanya 21,35% pada threshold 0,95: pembobotan positif berlebihan merupakan hipotesis yang perlu diperiksa. Tidak ada bukti bahwa satu komponen menyebabkan hasil Brain 90% atau menjamin FSDP 80%. Checkpoint SAM sekitar 50% tetap benchmark interaktif terpisah. Status: telaah dan rencana tercatat; transfer kode maupun training baru belum dilakukan pada langkah ini.

### Implementasi tahap pertama transfer Brain

Ditambahkan resep minibatch slice acak lintas pasien, input lima slice dan target slice tengah, dengan negatif tetap hadir. Dua arm baru memakai backbone residual yang sama: control loss lama dan focal–Dice–boundary dengan bobot batas dinormalisasi dalam satuan fisik. Bobot positif 10× tidak disalin. Global Dice ditambahkan bersama macro volume Dice dan metrik permukaan. Target 80% dicatat sebagai sasaran, bukan hasil. Maksimal 100 epoch per arm, patience 30, LR 1e-4. C8 belum diaktifkan pada tahap ini; pengaruh sampling/loss perlu diukur lebih dulu. Tiga tes sampler/loss CPU lulus; training DGX belum dijalankan dari sesi ini. Panduan: FSDP_BRAIN_RECIPE_2026-11-27.md.

### Hasil Brain recipe dan koreksi sampler

Run `fsdp_brain_recipe_20261127T022501Z_3624684`: control volume Dice **23,696%**, focal–boundary **22,540%** (−1,156 poin), best epoch 15, berhenti 45. Focal–boundary: train Dice **35,188%**, precision **20,589%**, recall **25,534%**, surface Dice @1 mm **57,256%**, median HD95 **5,067 mm**. Global Dice **22,770%** juga rendah, sehingga perbedaan agregasi dengan Brain bukan penjelasan cukup untuk kesenjangan hasil run ini.

Ditemukan masalah implementasi batch: padding antarukuran ROI memengaruhi Conv/GroupNorm saat train, meski dimask pada loss, sedangkan inference tidak memakai padding tambahan itu. Uji sintetis menunjukkan sensitivitas prediksi terhadap padding; dampaknya pada checkpoint/pasien nyata belum diukur. Diperbaiki dengan kelompok ukuran XY persis sama, semua slice/negatif tetap dicakup, urutan batch diacak, dan collator menolak campuran ukuran. Lima tes lokal lulus, termasuk prediksi batch versus individual. Performa koreksi belum tersedia; data tidak dapat dipakai untuk menjanjikan target 80% atau menetapkan padding sebagai satu-satunya penyebab. C8 belum diuji pada run ini.

### Backbone C8 dari struktur Brain

Ditambahkan `BrainUNet25D` dengan pembanding plain4 dan C8: empat tahap encoder/decoder, lima slice, skip connections dan deep supervision. Default dua regular fields menghasilkan kanal skalar 16/32/64/128/256; jauh lebih ringkas daripada Brain asli. Data/loss/sampler yang sama dipakai kedua arm. Snapshot preflight memulihkan running statistics BatchNorm; checkpoint C8 tidak menyertakan cache filter transient mode eval. Empat tes CPU memakai ESCNN nyata lulus. Training DGX belum dijalankan dari sesi ini; belum ada hasil Dice C8.

SAM tetap benchmark terpisah: volume Dice 49,983%, surface Dice @1 mm 88,110% dengan GT ROI dan lima klik/sisi. Bobot SAM tidak dimuat ke C8 2D. Setup dan pipeline baru tersedia di FSDP_C8_2026-11-27.md. Sasaran volume Dice 80% belum tercapai; tidak ada klaim bahwa kecocokan struktur Brain menjamin skor yang sama pada FSDP.

### Hasil C8: penurunan terhadap pembanding

Run `fsdp_c8_20261127T061904Z_27398` selesai. Plain4: volume Dice **25,688%**, surface Dice @1 mm **62,538%**, median HD95 **4,893 mm**, best epoch 50, berhenti pada 80. C8: volume Dice **12,601%**, surface Dice @1 mm **50,419%**, median HD95 **6,897 mm**, best epoch 20, berhenti pada 50. Selisih C8 terhadap plain4 adalah **−13,087 poin persentase**.

C8 train Dice **39,650%**, validasi **12,601%**; gap **27,049 poin**. Precision validasi **14,588%**, recall **11,556%** pada threshold train-only 0,95. Pada threshold 0,5, volume Dice **12,074%**; perubahan threshold itu tidak menyelesaikan kegagalan. Global Dice **13,346%** juga rendah. Ada sisi validasi dengan Dice nol. Tidak ada bukti manfaat C8 pada konfigurasi ini.

Model C8 yang diuji memakai base_fields 2; model Brain yang ditelaah memakai 32. Kanal skalar yang disamakan dengan control tidak berarti jumlah parameter bebas sama. Normalisasi menggunakan InnerBatchNorm dengan running statistics, batch kecil serta ukuran ROI beragam. Ketidaksesuaian statistik, kapasitas dan karakteristik target adalah hipotesis, belum diagnosis penyebab. Uji kode lokal membuktikan bentuk/gradien/checkpoint tertentu, bukan reproduksi skor pasien atau kualitas training.

Keputusan: pertahankan hasil ini sebagai eksperimen negatif; jangan menaikkan kapasitas atau mengulang training penuh hanya berdasarkan ringkasan. Diperlukan config, preflight (jumlah parameter), history, CSV per pasien/sisi, mask dan overlay dari kedua arm untuk memeriksa kegagalan dan menilai uji BatchNorm train-only atau perbaikan lain yang benar-benar didukung bukti. Benchmark SAM tetap terpisah: volume Dice 49,983%, surface Dice @1 mm 88,110%, dengan GT ROI dan lima klik oracle/sisi. Tidak ada optimizer update tambahan pada langkah pembacaan hasil ini.

### Jalur SAM untuk seluruh struktur selain FSDP

Atas persetujuan pengguna, disiapkan pipeline15kelas selain RFSDP1/LFSDP9. Berbeda dari percobaan multiclass lama, model memuat encoder, prompt encoder, mask decoder dan CNN refiner dari checkpoint SAM affine49,983%volumeDice/88,110%surfaceDice1mm. Model biner bersama bergantian memproses struktur berdasarkan ROI/titik petunjuk, lalu hasil disusun menjadi peta label native eksklusif. Sasaran utama studi: surface Dice1mm80%; volume Dice tetap dilaporkan terpisah.

Audit CPU mencari cache full17kelas lama dengan manifest dan hash yang sama, memeriksa mapping/jumlah target, dan berhenti sebelum GPU bila label penuh tidak tersedia. Pengecualian train_28 menghasilkan53train/10val. Class-specific physical grids dan normalisasi dipasang dari training saja, dengan roundtrip0,95sebagai gate. Evaluasi menggunakan5oraclepoints per struktur yang beranotasi, maksimum75perpasien; tidak mendeteksi kelas absen dan tidak diklaim otomatisCTutuh. Enam tes CPU lulus: data/audit, ekspor native eksklusif dari logits sintetis, dan penghentian launcher saat seleksi GPU gagal. Sintaks Python/shell lulus. Ada peringatan NumPy matmul dari helper fisik lama pada lingkungan lokal, dengan pemeriksaan keutuhan target/geometri dan finite tensor tetap lulus. Training DGX belum dijalankan; ketersediaan cache lengkap dan skor baru belum dapat dipastikan. Panduan: SAM_OTHER_PARTS_2026-11-27.md.

### 27 November 2026 — SAM non-FSDP validation exceeds the 80% target

**Milestone:** the DGX completion report records **85.7441% patient-macro Volume Dice**, **97.3086% Surface Dice at 1 mm**, and **93.4820% Surface Dice at 0.5 mm** for the annotated non-FSDP structures. Both the average Volume Dice and Surface Dice at 1 mm exceed 80% under the assisted evaluation protocol. Training completed all **20/20 epochs**; the best checkpoint was selected at **epoch 20**. These results update the earlier implementation-only status above.

**Team reading note:** “non-FSDP” means that **RFSDP (class 1) and LFSDP (class 9) are excluded**. The reported 85.74% Volume Dice and 97.31% Surface Dice at 1 mm describe the other annotated structures. They are not new FSDP segmentation scores. Evaluation uses a GT-derived ROI and five corrective oracle points per annotated structure.

The preceding launcher report identified run `sam_other_parts_20261127T084702Z_384068` on physical GPU 3, with a 24 GiB allocator cap. The full-label cache audit passed for 53 training and 10 validation patients after excluding train_28. There were 441 annotated training patient–structure views per epoch and 111 optimizer updates per epoch. The workflow inherited the complete learned SAM-Med3D Turbo/history/CNN32 hybrid weights; it did not initialize a new random 2D backbone. Model/data preparation and evaluation implementation: commit `fa2f883`.

Validation covers **91 annotated patient–structure pairs**, representing **14 evaluable classes** among the 15 requested non-FSDP classes. LSOEC has zero annotated validation examples and therefore receives no score. Every evaluated class has a mean Surface Dice at 1 mm above 80%; this does not imply that every individual patient–structure prediction exceeds 80%.

```text
VALIDATION REPORT — SAM-Med3D + CNN Refiner
Non-FSDP structures | All scores in percent

Training: 53 patients | Validation: 10 patients
Completed: 20 epochs | Best checkpoint: epoch 20

Structure Precision  Recall   Dice   SD@1mm   N
------------------------------------------------
RANC          87.80   81.44   84.30    98.52  10
RSAC          90.93   81.06   84.89    99.55   6
RSAFC         91.13   80.13   84.34    97.94   3
RBE           92.06   87.58   89.58    98.07  10
RSBC          84.00   89.98   86.29    96.01   7
RSBFC         90.18   69.31   74.29    83.71   3
RSOEC         90.75   83.65   86.37    99.39   2
LANC          84.56   89.60   86.72    99.06  10
LSAC          93.49   81.85   87.00    99.59   7
LSAFC         93.27   77.35   84.41    94.55   3
LBE           91.78   81.93   86.44    96.73  10
LSBC          89.22   84.54   86.37    97.44   9
LSBFC         87.12   91.22   88.87    98.95   7
LSOEC             —       —       —        —   0
FSC           83.87   83.55   83.18    96.86   4
------------------------------------------------
Patient-macro 88.82   84.10   85.74    97.31  10

Surface Dice @0.5mm: 93.48%
N = annotated validation patients per structure.
14 evaluated classes; 91 patient–structure pairs.

Protocol: GT-derived ROI + 5 oracle points/class.
FSDP excluded. Not automatic full-CT evaluation.
Test images not used.
```

**Calculation and verification.** For each annotated patient–structure pair, Precision = TP/(TP+FP), Recall = TP/(TP+FN), and Volume Dice = 2TP/(2TP+FP+FN). Each class row averages its available validation patient scores. The overall patient-macro row instead averages the annotated class scores within each patient, then averages the ten patient results. It is not the unweighted mean of the fourteen class rows or a pooled-voxel Dice. Precision, recall and all three overall Dice values were recalculated from the 91 supplied case-class records and match the report within floating-point precision.

Example from `val_01/RANC`: GT = 883 voxels, predicted = 943 voxels. The supplied precision and voxel counts give integer TP = 774, FP = 169, FN = 109. Precision = 774/943 = **82.08%**; Recall = 774/883 = **87.66%**; Dice = 1548/1826 = **84.78%**. The ten unrounded patient-average Volume Dice values sum to 8.574408147710288; dividing by ten and multiplying by 100 gives **85.7440814771%**. Surface Dice uses physical 3D surface distances with the stated tolerance, not this voxel-overlap formula. A 1 mm tolerance does not mean the CT was resampled to 1 mm voxels. A complete interclass confusion matrix cannot be reconstructed from these summaries; no TN counts or off-diagonal class-confusion counts were invented.

Before continuation on this new task, the source checkpoint evaluated with the new preprocessing and target set recorded Volume Dice **12.1958%**, Surface Dice at 1 mm **33.8692%**, and Surface Dice at 0.5 mm **21.5039%**. Its rise to 85.7441% Volume Dice is **+73.5483 percentage points within this task/protocol**. This is not a controlled comparison against the earlier FSDP-only 49.983% benchmark: target anatomy, normalization, grids and evaluation aggregation differ.

**Remaining weaknesses:** RSBFC is the weakest evaluated class, with mean Volume Dice **74.2910%** across only three annotated validation patients. `val_08/RSBFC` records Volume Dice **45.6187%**, Surface Dice at 1 mm **53.1980%**, precision **93.6275%**, and recall **30.1559%**. The lowest patient-average Volume Dice is `val_08` at **79.0548%**. Some other classes have only two or three validation examples, limiting confidence in their class averages.

**Scope and interpretation:** ROI centers, class presence and five corrective oracle points per annotated structure are obtained from GT. Scores are computed on the final mutually exclusive native-field label map after merging prompted masks. Absent-class localization is not evaluated. These are assisted validation results for structures other than FSDP, not a new FSDP score, an automatic whole-CT result, an independent held-out test result, or a clinical deployment claim. The validation set has already been examined repeatedly. Overlays and masks have not been independently inspected in this local reporting step.

Evidence: user-supplied DGX completion log, SHA256 `ddfb167ccf8b082ea42b086fd55b56a5aeac0e1f80b0f4d7c6d0e90188ab0668`. Reported best checkpoint SHA256: `c6d6eb911e595bf8393702f4716ee1648e3ca32c9cc94c23ba5d6e8dd2040241`. The supplied log and its nested metrics were parsed locally; the DGX checkpoint itself was not downloaded or rerun. Decision: retain this checkpoint and its exports as the current non-FSDP assisted validation benchmark; inspect overlays and the weak RSBFC case before expanding claims. No additional optimizer updates were performed while recording these results.

An English evaluation presentation was also prepared with nine slides in a journal-style layout. The architecture slide follows the cover and shows the 128³ SAM context, 8³ Turbo embedding, CNN detail32³ branch, 67-to-64-channel feature fusion, skip decoder and residual mask correction. The remaining slides cover cohort/protocol, calculations, overall and per-structure results, patient variation, the weak RSBFC case, and study limitations. The standalone reproducibility slide was removed at the user's request; this logbook retains the evidence and checkpoint record above.
