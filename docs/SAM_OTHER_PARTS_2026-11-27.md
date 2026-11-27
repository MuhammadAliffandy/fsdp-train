# Segmentasi struktur selain FSDP dengan SAM-Med3D hybrid terlatih

## Cakupan

Pipeline mempelajari 15 kelas anatomi selain RFSDP (ID 1) dan LFSDP (ID 9). Nama kelas dibaca dari mapping cache multikelas asli, tidak ditebak dari nomor kelas. Model mempertahankan image encoder, prompt encoder, mask decoder dan CNN detail/refiner dari checkpoint SAM affine terbaik. Semua struktur memakai satu model biner bersama dengan ROI dan titik petunjuk spesifik struktur, lalu logits digabung menjadi peta label 3D eksklusif. Tidak ada classifier18kelas acak yang menggantikan decoder SAM.

Benchmark sumber: volume Dice FSDP 49,983%, surface Dice @1 mm 88,110%, dengan ROI GT dan lima klik oracle per sisi. Angka ini tidak menetapkan performa pada 15 struktur lain. Bobot sumber pernah dilatih dengan train_28; continuation baru mengecualikannya. Target studi baru adalah macro surface Dice @1 mm >=80%, disertai volume Dice dan hasil per kelas. Target volume Dice >=80% dicatat terpisah.

## Pemeriksaan data sebelum GPU

Launcher memeriksa kandidat `runs/*/preprocessing/prepared.json` pada proyek yang sama. Kandidat harus memiliki manifest yang sama dengan fondasi SAM, mapping kelas0..17 dengan RFSDP1/LFSDP9, jumlah pasien54/10 asli, serta hash cache tiap pasien yang sesuai. train_28 dilewati; hasil aktif harus53train/10val. Target native per kelas harus sama jumlah voxelnya dengan coverage asli. Bila path cache lama berpindah, pencarian dibatasi pada subfolder preprocessing kandidat dan hanya menerima hash identik.

Jika anotasi penuh tidak ditemukan, audit menghasilkan ready=false dan tidak memilih GPU atau memulai training. Cache biner FSDP tidak dapat dipakai untuk menciptakan label struktur lain. Jalur ini belum memulihkan original ColorTable/labelsTr; mapping dan mask terverifikasi berasal dari artefak multiclass lama. Jika cache tersebut hilang, diperlukan pemulihan anotasi penuh dari sumber sah.

## Preprocessing dan training

Grid detail per kelas berbentuk96x160x208. Spacing ditentukan dari rentang anotasi TRAIN kelas tersebut ditambah margin12mm, dengan minimum0,25mm. Konteks SAM128kubik berpusat pada lokasi fisik yang sama. Statistik intensitas memakai konteks CT native dari53pasien training, clipping persentil0,1/99,9 dan z-score. Perubahan input terhadap benchmark sumber ini merupakan adaptasi task, bukan jaminan reproduksi angka FSDP lama.

Seluruh target harus tertahan. Pada training, label roundtrip harus mencapai0,95 sebelum optimizer dijalankan. Bila gagal, staging berhenti dan kelas/kasus dilaporkan; grid tidak dipilih dari validasi. Target native untuk scoring tetap dipertahankan. Class-view cache memerlukan ruang disk tambahan; script memeriksa batas awal dan berhenti sebelum ruang bebas turun di bawah5GiB. File CT/cache asli tidak dimodifikasi.

Shared model dilatih bergantian pada semua pasangan pasien/kelas yang memiliki anotasi. Resep awal20epochmaksimum, patience10, accumulation4, optimizerAdamW dengan LR SAM2e-6 dan CNN2e-5, encoder parsial blok10/11 dan mask decoder seperti foundation. Prompt encoder dan encoder lain tetap mengikuti jalur pretrained. Loss DiceCE dan auxiliary context mengikuti hybrid sukses; augmentasi flip serta gain/shift ringan. Tidak ada run classifier/C8 yang disisipkan. Default screening dapat diperpanjang melalui PARTS_EPOCHS setelah hasil dinilai.

## Protokol evaluasi dan interpretasi

Lima titik oracle per struktur GT-present, maksimum75titik untuk15struktur pasien. ROI dan keberadaan struktur diperoleh dari anotasi; ini segmentasi interaktif berbantuan lokasi, bukan deteksi/pencarian otomatis semua anatomi pada CT utuh.

Setiap mask logit dipulihkan secara fisik ke native scoring field cache. Background memakai logit0 (probabilitas0,5); pada konflik antarkelas, logit tertinggi menjadi label akhir. Hasil volume Dice dan surface Dice dihitung pada peta eksklusif yang sudah digabung. Ground truth kelas1/9 tidak termasuk target yang dilaporkan. Struktur yang tidak memiliki anotasi tidak dibuatkan ROI/prediksi; deteksi false positive kelas absen belum diukur. Tidak ada skor1untuk empty/empty. Macro dihitung pada kelas GT-present dalam pasien, kemudian dirata-ratakan antar10pasien. Perbedaan jumlah kelas yang beranotasi harus dibaca dari laporan.

Per-class table dan patient-class table menyertakan volume Dice, surface Dice0,5/1mm, precision/recall, jumlah voxel dan HD95. Summary mencantumkan kelas mana yang mencapai surface Dice1mm80%. Checkpoint dipilih dari macro surface Dice1mm, dengan checkpoint sumber sebagai kandidat awal. Ini evaluasi eksploratif pada validasi yang sudah ditinjau berulang; test tidak dibaca. File NIfTI memakai ukuran/spacing/origin/direction native field cache, bukan grid volume CT utuh yang baru ditemukan secara otomatis.

## Menjalankan di DGX

```bash
cd ~/raid/Clara/fsdp-train
git pull origin codex/exclude-mislabeled-train28
conda activate fsdp-med3d
ALLOW_GPU_SHARING=1 GPU_ID=auto bash scripts/launch_sam_other_parts_nohup.sh
tail -f logs/sam_other_parts/latest.log
```

Default sumber: `runs/sam_augmentation_20261126T084626Z_4096150/affine`. Override melalui FSDP_PARTS_SOURCE_DIR bila checkpoint yang sama dipindah. FSDP_MULTICLASS_PREPARED dapat menunjuk prepared.json tertentu; jika tidak diset, audit memeriksa kandidat terbatas pada direktori runs proyek. Laporan audit disimpan pada logs/sam_other_parts/*.audit.json. Saat ready=false, tidak ada latest training log baru; baca laporan audit tersebut. GPU hanya dipilih setelah audit label lulus, dengan48GiBkosong dan batas allocator24GiB. Pemilihan awal bukan reservasi GPU.

PARTS_EPOCHS default20 dan PARTS_PATIENCE default10. Setelah training, `summary.json` memuat initial/best metrics, kelas yang mencapai target, serta scope anotasi yang absen. `best_exports` berisi peta15kelas per pasien, target dan CSV diagnostik. `failure.json` menampung kegagalan staging/training. Run baru tidak mendukung resume; sumber/model terdahulu tetap terpisah. Training penuh, cache DGX dan peningkatan Dice belum dijalankan/diverifikasi dari sesi lokal.

## Verifikasi lokal

Tes CPU memeriksa pemilihan15kelas, penolakan jumlah target yang berubah, keselarasan fisik detail/konteks, keutuhan target native dan penolakan kelas tanpa anotasi. Tes audit membuktikan tidak ditemukannya cache multikelas menghasilkan ready=false. Tes launcher memastikan kegagalan pemilihan GPU menghentikan proses. Tes komposisi memakai logits sintetis untuk memeriksa konflik kelas, pengecualian kelas absen, serta ukuran/spacing/origin ekspor native. Enam tes tersebut lulus, disertai pemeriksaan sintaks Python/shell. Helper fisik lama mengeluarkan peringatan NumPy matmul pada lingkungan lokal; pemeriksaan target/geometri dan finite tensor lulus. Tes belum merupakan bukti kualitas segmentasi atau training GPU menyeluruh.
