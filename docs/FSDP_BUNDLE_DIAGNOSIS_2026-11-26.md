# Diagnosis artefak DGX FSDP, 26 November 2026

## Ruang lingkup dan keputusan produk

Pengguna mengonfirmasi target merupakan seluruh volume anatomi FSDP dengan inferensi otomatis dari CT. Analisis ini membaca dua run continuation 53 train/10 validation, seluruh history dan CSV native yang tersedia, metadata diagnosis 128 ROI beserta 64 header, serta delapan gambar overlay yang dipilih untuk melihat kasus sulit dan pembanding. Definisi volume yang dikonfirmasi pengguna menjadi acuan. Tampilan target yang tipis pada PNG belum menunjukkan kesalahan anotasi.

Arsip sumber berada di `fsdp_diagnostic_bundle.tar.gz`, dengan ekstraksi di `fsdp_diagnostic_bundle/`. Model checkpoint, CT sumber, dan mask prediksi/target volumetrik dikecualikan saat pengemasan. Analisis tidak menjalankan training ataupun menilai ulang citra test. Seluruh hasil SAM di sini memakai ROI ground truth dan lima oracle clicks per sisi, sehingga kinerja otomatis CT utuh masih memerlukan evaluasi tersendiri.

Sumber utama di bawah root ekstraksi:

- `fsdp-train-train28fix/runs/fsdp_private_foundation_20261125T155656Z_1188354/` — partial encoder.
- `fsdp-train-train28fix/runs/fsdp_private_foundation_20261126T032200Z_3059519/` — full image encoder.
- `fsdp-train/outputs/private/system_diagnosis/latest/summary.json` — profil cache, header dan laporan diagnosis sebelumnya.

Artefak hitungan lokal tersimpan pada `outputs/private/diagnostic_review_20261126/`: `analysis.json`, `learning_curves.csv`, `validation_pairs.csv`, dan `learning_and_input_review.png`. JSON mencatat hash arsip dan berkas sumber agar hasil dapat ditelusuri.

## 1. Kurva menunjukkan train membaik dan validasi kemudian melemah

| Mode dan checkpoint | Train Dice@5 | Validation Dice@5 | Threshold train-only |
|---|---|---|---|
| Inisialisasi bersama, epoch tambahan 0 | 50,13% | 45,63% | 0,35 |
| Partial, best epoch 5 | 53,25% | 48,09% | 0,45 |
| Partial, epoch terakhir 55 | 60,31% | 42,34% | 0,45 |
| Full, best epoch 25 | 57,98% | 46,45% | 0,50 |
| Full, epoch terakhir 60 | 63,65% | 41,89% | 0,50 |

Pada full encoder, best ke last menghasilkan kenaikan train 5,67 poin dan penurunan validasi 4,56 poin pada threshold yang sama. Partial juga menunjukkan kenaikan train sekitar 7,07 poin dan penurunan validasi sekitar 5,75 poin setelah best. Pola tersebut mendukung diagnosis overfitting pada resep ini. Fitting cohort train masih jauh dari hasil tiny dua pasien, sehingga generalisasi belum merupakan satu-satunya aspek yang perlu diperiksa.

Mode partial mencatat 770 update optimizer dan mode full 840 update, masing-masing tanpa AMP-skipped update. History tersusun kontinu dan jumlah update cocok dengan expected_updates pada setiap epoch. Kedua run memakai seed, checkpoint awal, physical cache, manifest, normalisasi, LR, accumulation, loss, augmentation, jadwal, dan aturan evaluasi yang sama. Hash kode model/data sama. Perubahan kode trainer/foundation sesuai penambahan mode encoder. Perbandingan checkpoint terbaik tetap melibatkan epoch dan threshold terpilih yang berbeda.

Early stopping full terjadi pada epoch 60 karena min_delta=0,001 dan patience=50. Epoch 10 mencapai 46,4188%, lalu epoch 25 mencapai 46,4498%, naik sekitar 0,031 poin persentase. Kenaikan itu cukup untuk menyimpan best checkpoint baru tetapi berada di bawah min_delta 0,1 poin untuk mereset patience. Perilaku ini sesuai fungsi early_stop_update pada kode.

## 2. Gradien mencapai seluruh encoder pada probe awal

`encoder_trainability.json` mencatat seluruh 92.924.928 parameter image encoder aktif. Mask decoder 7.575.636 parameter aktif dan prompt encoder 8.668 parameter beku. CNN/refiner berjumlah 407.817 parameter aktif.

Dua probe pada epoch awal continuation full menunjukkan gradien finite dan bernorma positif pada patch embedding, 12 blok transformer, serta neck. Norma gabungan SAM/CNN sebelum clipping sekitar 4,98 dan 3,21. Dengan clip_norm=1, faktor skala gradien pada kedua update itu sekitar 0,201 dan 0,311. Norma CNN/refiner lebih besar daripada kelompok SAM pada kedua probe.

Temuan ini menunjukkan jalur backward mencapai encoder pada update awal. Dua probe belum mengukur statistik clipping seluruh epoch, perubahan bobot relatif tiap lapisan sepanjang training, atau kegunaan representasi fitur. Menaikkan skala loss saja juga harus mempertimbangkan clipping yang sudah aktif.

Laporan loss lama di dalam diagnosis memakai checkpoint dan contoh berbeda. Keenam contoh yang tersedia mempunyai cosine gradien detail/context positif pada encoder dan mask decoder. Contoh tersebut belum mendukung klaim bahwa kedua loss selalu saling berlawanan. Bukti yang tersedia belum menetapkan loss atau vanishing gradient sebagai penyebab utama plateau.

## 3. Perbandingan validasi sensitif terhadap beberapa pasien

Lima pasien meningkat dan lima menurun pada checkpoint terpilih full versus partial. Perubahan rata-rata adalah −1,64 poin persentase. `val_06` turun 46,63% menjadi 32,44%, selisih −14,18 poin. Kontribusi pasien tersebut terhadap perubahan rata-rata sepuluh pasien sekitar −1,42 poin. Semua pasien tetap masuk laporan.

| Pasien | Partial Dice@5 | Full Dice@5 | Perubahan, poin persentase |
|---|---|---|---|
| val_01 | 56,60% | 50,95% | −5,64 |
| val_02 | 36,88% | 42,31% | +5,42 |
| val_03 | 45,17% | 47,11% | +1,95 |
| val_04 | 43,94% | 44,98% | +1,04 |
| val_05 | 54,83% | 52,38% | −2,45 |
| val_06 | 46,63% | 32,44% | −14,18 |
| val_07 | 47,76% | 44,50% | −3,26 |
| val_08 | 58,65% | 55,24% | −3,41 |
| val_09 | 54,40% | 55,26% | +0,87 |
| val_10 | 36,06% | 39,32% | +3,26 |

Interpretasi peningkatan atau penurunan memakai satu seed dan sepuluh pasien yang telah berulang kali diperiksa selama pengembangan. Besarnya selisih agregat belum menunjukkan keunggulan yang stabil di populasi baru. Perbandingan fixed-0,5 juga menurun, dari 47,43% ke 46,45%, sehingga perbedaan threshold belum menjelaskan seluruh selisih.

## 4. Kesalahan overlap mencakup target terlewat dan area berlebih

Pada checkpoint full, `val_06` memiliki 447 voxel target, 564 voxel prediksi, TP 164, FP 400, FN 283. Jumlah prediksi 1,26 kali target, tetapi overlap masih rendah. Pada `val_09`, volume prediksi 693 versus target 675 mendekati sama, sementara Dice hanya 55,26%. Perubahan volume atau threshold saja belum cukup untuk menjelaskan kesalahan spasial.

Kasus train_53 tetap sulit. Pada full encoder, target bilateral 564 voxel, prediksi 2.336 voxel, TP 441, FP 1.895 dan FN 123, Dice 30,41%. Recall 78,19% disertai precision 18,88% menunjukkan burden area berlebih. Diagnosis input dan konteks pada kasus tersebut layak ditinjau bersama kasus train lain yang sulit, dengan perbandingan terhadap kasus baik.

Overlay yang diperiksa secara langsung:

- Full `val_06_RFSDP` dan `val_06_LFSDP`, serta partial `val_06_RFSDP`.
- Full `val_02_RFSDP`, `val_10_LFSDP`, `val_08_RFSDP`, dan `val_01_RFSDP`.
- Probe ResUNet zero-click `train_53_RFSDP` pada threshold 0,95.

Pada irisan terpilih terlihat bagian target tipis yang terlewat, ketidakcocokan bentuk, serta posisi prediksi yang bersebelahan atau bergeser dari target pada beberapa kasus. Gambar tersebut belum menyediakan ukuran jarak dalam mm maupun pemeriksaan seluruh volume. Pemeriksaan anatomi oleh pemberi anotasi diperlukan untuk menilai konsistensi batas. Perubahan label belum memiliki dasar dari review PNG ini saja.

## 5. Clipping terukur, dampak kausal belum terisolasi

Setelah mengecualikan train_28 dari ringkasan, terdapat 106 ROI train dan 20 ROI validation.

| Pengukuran | Train, median antar-ROI | Validation, median antar-ROI |
|---|---|---|
| Voxel detail valid pada clip atas | 19,82% | 21,16% |
| Voxel SAM context valid pada clip atas | 15,21% | 15,63% |
| Foreground pada detail grid | 0,1246% | 0,1183% |
| Roundtrip label minimum | 99,06% | 99,17% |

Clipping atas detail train berkisar 12,21–28,73%. Jadi variasi sebagian konteks memang diratakan oleh clip_upper=136,84. Besarnya serupa di train dan validation. Korelasi Pearson eksploratif antara rerata clipping detail per pasien dan Dice full encoder justru positif pada sampel ini, sekitar 0,25 di train dan 0,31 di validation. Angka tersebut belum mendukung hubungan sederhana bahwa pasien dengan clipping lebih besar selalu memiliki Dice lebih rendah. Korelasi pada sepuluh validation pasien mempunyai ketidakpastian besar.

Normalisasi masih layak diuji lewat matched control karena informasi intensitas di atas ambang hilang. Hasil analisis belum menetapkan bahwa perubahan normalisasi akan mencapai target 80%.

## 6. Kandidat normalisasi lama memerlukan revisi

Kandidat −1000 hingga 2000 dalam laporan belum dilatih. Statistiknya memakai 54 pasien, termasuk train_28 yang berisi label map pada jalur citra. Kandidat berikutnya perlu menghitung statistik dari 53 train yang valid.

Batas bawah −1000 juga perlu dikaji. Raw target train_01_RFSDP memiliki median −1009 dan LFSDP −1042,5. Kebijakan baru dengan batas bawah −1000 akan meratakan sebagian besar nilai target native pada kasus tersebut. Pemilihan rentang perlu memeriksa distribusi raw foreground serta konteks train dan satuan intensitas sumber. Header slope=1/intercept=0 belum sendiri membuktikan kalibrasi HU fisik yang benar.

Provenance kandidat lama mencatat 128 native targets dan 256 model targets/support, tetapi penghitung kesetaraan grid bernilai 0. Ini merupakan kekurangan bukti pada laporan kandidat itu. Kesetaraan semua grid perlu diperiksa ulang saat kandidat baru dibuat. Kode builder yang ada sekarang memiliki penghitung grid, tetapi laporan lama belum dapat dianggap membuktikan pemeriksaan tersebut telah terjadi pada artefaknya.

## 7. Percobaan berikutnya untuk kebutuhan otomatis

1. **Lengkapi diagnosis spasial dan input.** Gunakan mask prediksi native, target cache, serta source CT di DGX untuk mengukur lokasi/extent error per sisi dan mengecek distribusi raw train. Pertahankan target volume sesuai definisi pengguna. Laporkan komponen target yang terlewat, FP/FN, dan jarak permukaan dalam mm sebagai pendamping Dice volume.
2. **Bangun kandidat normalisasi 53 pasien.** Tentukan satu rentang dari audit train, dengan pengecekan lower-tail target dan konteks. Verifikasi geometri, target dan support terhadap kontrol. Awali pembandingan dari checkpoint yang sama dengan budget yang sama, dan beri waktu adaptasi terhadap perubahan input. Partial encoder menjadi kontrol yang relevan dari dua run ini.
3. **Implementasikan lokalisasi dari CT utuh.** Train localizer pada CT resolusi lebih rendah menggunakan target volume native terverifikasi. Ukur recall/retensi target di crop yang diprediksi serta kegagalan per pasien. Ukuran crop, fallback dan threshold ditentukan dari train.
4. **Latih segmenter otomatis pada crop yang menyerupai keluaran localizer.** Sertakan pergeseran crop dan hard negatives yang dipilih dari train. SAM dapat menerima prompt hasil prediksi sistem, atau decoder zero-click dapat digunakan. Protokol train dan inferensi harus cocok. Prediksi localizer out-of-fold untuk train memerlukan model yang belum dilatih pada pasien fold evaluasinya.
5. **Nilai pipeline gabungan pada CT utuh.** Semua pasien masuk perhitungan, dengan target yang terlewat localizer ikut menjadi FN. Evaluasi ROI ground truth dipakai untuk mengukur kemampuan segmenter secara terpisah selama diagnosis.

Pemilihan loss/regularisasi berikutnya mengikuti pola error train yang terukur. Bukti baru mendukung prioritas pada generalisasi, representasi target tipis, konteks input, dan kecocokan protokol otomatis. Pembukaan encoder lebih jauh sudah teruji dalam resep ini dan belum memberi kenaikan validasi. Kualitas anatomi target dan hasil otomatis tetap belum terukur dari paket ini.

## Pemeriksaan konsistensi analisis

Dice dihitung ulang dari TP, jumlah prediksi dan seluruh target untuk dua run, dua split, serta satu/lima klik per sisi. Jumlah pasien/ROI dan larangan train_28 pada skor train diperiksa. Jumlah voxel, TP, dan prediksi kedua sisi cocok dengan total pasien. Rata-rata pasien cocok dengan summary. History kontinu, accounting update cocok, serta nilai best pada history cocok dengan summary. Inisialisasi dan resep pasangan diverifikasi dari config/preflight. Semua pemeriksaan tersebut lulus. CT, mask volumetrik, checkpoint dan anatomi belum diperiksa ulang secara langsung.
