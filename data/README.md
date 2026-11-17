# Data layout

Keep image data and generated masks outside version control. The repository ignores files under these data directories.

```text
data/
├── raw/
│   ├── nasalseg/
│   └── private/              # never copy into public archives or commits
├── interim/
├── pseudo_labels/
│   └── totalsegmentator/
├── processed/
│   └── sam_med3d/
└── manifests/                # keep subject identifiers local/private
```

Keep raw scans unchanged. Store preprocessing outputs separately and record the source file, geometry, label mapping, and preprocessing configuration for each derived volume. Do not put patient identifiers or private paths in tracked manifests.

The first dataset audit should verify that public CT volumes include the frontal sinus region and that the private masks use the intended target definition. No resampling, intensity normalization, cropping, or pseudo-label generation is assumed until those checks are complete.
