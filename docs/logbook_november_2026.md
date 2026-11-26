# Logbook FSDP 3D Segmentation — November 2026

Catatan harian eksperimen, diagnosis, dan keputusan untuk meningkatkan Dice segmentasi FSDP. Struktur tabel mengikuti logbook ColonoMind November 2026. Disusun pada 26 November dari laporan DGX yang diberikan pengguna, catatan proyek, dan pembacaan kode lokal. Waktu mulai run mengikuti timestamp UTC pada nama run, sedangkan tanggal log memakai Asia/Jakarta.

Ukuran keberhasilan utama adalah mean bilateral patient Dice pada grid asli. Evaluasi SAM memakai ROI yang ditentukan dari ground truth dan titik petunjuk yang dibentuk dari ground truth (oracle clicks). Satu atau lima klik diberikan per sisi, sehingga lima klik per sisi berarti sepuluh titik per pasien. Hasil tersebut merupakan evaluasi segmentasi interaktif dengan bantuan lokasi.

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
