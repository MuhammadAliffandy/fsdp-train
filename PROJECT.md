# FSDP Segmentation Experiment

## Status

The first private validation completed with 0 failures on all 10 reserved validation patients. Latest, training-loss-best, and training-Dice-best checkpoints all reported saved epoch 50 and identical mean Dice: 0.11110096921867103 at 1 click, 0.10670412495253445 at 5 clicks. Mean target retention was 1.0, with 0 empty crops. This does not support target truncation as the cause on this validation set; the root cause of low segmentation overlap is still unresolved. The private training/validation numbers use different protocols and are not direct evidence of overfitting.

The user authorized a comparison of original turbo, the public initialization checkpoint, and private-fine-tuned weights under the same 10-patient validation protocol. `compare_pretraining_nohup.sh` now launches this through the shared evaluator. It resolves the actual public initialization from private checkpoint metadata, records file checksums, and adds precision/recall, prediction size ratios, TP/FP/FN, paired Dice changes, and scientific comparison plots. Only model weights vary; adaptive oracle click positions after the first click may differ as predictions differ. The 20 test patients remain reserved. This three-stage comparison has not yet run on DGX.

Private fine-tuning completed through epoch 49/49 with 54/54 training batches (user screenshot, 2026-11-18). Final reported training loss: 7.16802140518471; training Dice: 0.2425045617338684. These training metrics are low and are not validation/test results. The user authorized the next diagnostic stage: curves, checkpoint comparison on the frozen 10-patient validation set, and overlays. Twenty test patients remain reserved.

`evaluate_private_validation.py` and `launch_validation_nohup.sh` prepare this stage. Evaluation uses the training-matched GT-centered crop/normalization, oracle error clicks at 1/5 interactions, crop/full-target Dice, crop retention, and a comparison of logit thresholds 0 and 0.5. It compares only the three stored private checkpoints (latest, training-loss-best, training-Dice-best). Full-target scoring includes target voxels lost to cropping and ignores padded voxels outside original CT bounds. It records protocol/version metadata and exports scientific PNGs/CSV/NIfTI under ignored local outputs. This is an assisted segmentation diagnostic, not automatic full-volume performance. Validation has not yet been run on DGX.

Public fine-tuning completed through epoch 49/49 with 130/130 batches in the final epoch (user-reported screenshot): training loss 2.0503935735959273, training Dice 0.785527542921213. These are training crop metrics against TotalSegmentator pseudo-labels, with empty-pair Dice defined as 1.0. All 130 cases were used for training; no held-out public evaluation was performed.

The next authorized stage is supervised private FSDP fine-tuning. The dataset resides on DGX outside this repo at `~/raid/normal case_86`. The user confirmed `labelsTr` contains the final FSDP annotations. Other label versions exist in `labels_exp1`, `labels_exp2`, and `labels_exp3`. The first audit found 86 images and 86 labels but could not match their different basenames. The naming convention is now confirmed as `<patient_id>.nii.gz` for images and `<patient_id>-label.nii.gz` for masks. The audit supports this suffix and checks membership of the existing JSON `test_set`, which contains 26 string entries.

The user explicitly requested exactly 20 patients for the new private test set (not 20%). This supersedes using all 26 entries of the old split; keep the old split file intact for provenance. The updated audit matched all 26 old test entries without missing or duplicate IDs. The supplied five ColorTables agree on `1=RFSDP` and `9=LFSDP`. Preparation now checks this mapping in every patient's table, extracts the bilateral binary FSDP union, and freezes a deterministic seed-2026 split: 20 test drawn from the old 26, 10 validation, remaining training (56 if all 86 pass). This assumes one numeric ID per patient and one CT per ID; other grouping requires an explicit map. All audit rows and source geometry/labels must pass before preparation succeeds. Frozen splits and source identifiers stay in ignored local data paths.

The private launcher initializes a new run from the completed 50-epoch public checkpoint, with fresh optimizer/scheduler, learning rate 2e-5, and only the private training folder. The common launcher checks and passes the dataset path via a runtime environment variable. Validation/test are reserved in separate directories; upstream does not compute validation metrics or select checkpoints by validation performance. Preparation and launcher were checked with synthetic NIfTI patients and a mocked trainer process before the completed private DGX run reported above.

The complete private audit output confirmed 84 `ok` pairs and 2 `geometry_mismatch` pairs. Both mismatches have different dimensions, spacing, and origins; shared anatomy/scan correspondence has not been established. Preparation initially stopped at its strict audit gate. The first private run can now explicitly exclude these two cases with `--exclude-geometry-mismatch`, record them in the frozen manifest, and use **54 train / 10 validation / 20 test** from the 84 eligible patients. Original files remain intact; test selection uses only eligible IDs from the original 26-patient pool. Other audit failures still block preparation. Geometry alignment of excluded cases is a separate follow-up, not assumed to be solved by copying metadata.

### Earlier public training debugging

The patched run completed epoch 0 and reached batch 83/130 of epoch 1, then failed in training Dice: two empty masks returned a Python NaN float and the batch mean called `.item()`. Setup now gives empty pairs a tensor Dice of 1.0. The launcher can resume the latest completed-epoch checkpoint with `RESUME=latest`, preserving prior logs. This metric convention counts empty-pair agreement and must be considered when interpreting crop-level training Dice.

The next DGX attempt passed the loss calculation and reached batch 88/130 of epoch 0, then stopped in the upstream prompt sampler with `ValueError: high <= 0` (no error voxels to sample). Setup now patches a target-based fallback for this condition. CPU checks reproduced the upstream crash and verified empty targets, identical foreground, full masks, false negatives, false positives, mixed batches, and repeated setup. The corrected run still needs to execute on DGX.

The first public training launch loaded the turbo checkpoint and found 130 batches, but stopped at epoch 0 before finishing the first batch: MONAI's Dice loss received an integer target. Setup now casts the target to float32 specifically at the loss call; the nohup launcher uses unbuffered Python output for log monitoring. The corrected launch still needs to run on DGX.

The public pseudo-label QC passed for all 130 NasalSeg cases before the training attempts described above.

## Project goal

Design and run a reproducible public frontal-sinus experiment using pseudo-labels, then use the available private NIfTI image/mask dataset for supervised fine-tuning and evaluation. The long-term clinical target is the frontal sinus drainage pathway (FSDP); the exact relationship between that target and the private mask label still needs confirmation. A frontal sinus mask must not be assumed to equal an FSDP mask.

The eventual product may be a website or mobile application, depending on the workflow and user needs. Product design and implementation are outside this experiment's current scope.

## Current proposed sequence

1. Inspect public CT and private NIfTI geometry, image coverage, and label values.
2. Generate and review TotalSegmentator frontal-sinus pseudo-labels on suitable public CT scans. User-reported QC: 130 cases, 130 masks, all statuses `ok`, no geometry mismatches, empty masks, or failures. This is automated QC and does not verify anatomical correctness.
3. Convert accepted pseudo-labeled cases to SAM-Med3D's `imagesTr`/`labelsTr` layout and run a public pseudo-label adaptation baseline. The standard upstream trainer logs training metrics and does not create a held-out validation set.
4. Confirm the private label definition, then define a patient-level fine-tuning and held-out evaluation split.
5. Fine-tune and evaluate on private data; preserve code, configs, logs, and checkpoints.

Each step that changes the experiment direction will be reviewed together before proceeding.

## Public dataset under consideration

**NasalSeg Dataset for Nasal Cavity and Paranasal Sinuses Segmentation from CT Images**

- Record: [Zenodo 13893419](https://zenodo.org/records/13893419)
- DOI: `10.5281/zenodo.13893419`
- Published version shown on the record: v2, October 5, 2024
- Record description: 130 CT scans with pixel-wise labels for five structures: left and right nasal cavity, nasopharynx, and left and right maxillary sinus.
- Listed archive: `NasalSeg.zip`, 224 MB.

### Dataset-to-goal fit: unresolved blocker

The Zenodo record's description does not list the frontal sinus or FSDP among its annotated structures. Therefore, it does not currently establish that the public labels can train or directly evaluate an FSDP segmentation model. Before training, we need to inspect the archive and its documentation to verify the label map and determine whether there are relevant target labels. If there are none, we must choose a defensible alternative, such as a clearly scoped adjacent-structure benchmark, before calling the result an FSDP experiment.

The dataset's license field was not clear in the record metadata reviewed so far; verify the license and permitted use before distributing data or derived artifacts.

## Model selection

The intended architecture is confirmed as **SAM-Med3D**.

References: [official repository](https://github.com/uni-medical/SAM-Med3D), [paper](https://arxiv.org/abs/2310.15161).

No claim is made here that this model is already suitable for FSDP or that its released checkpoint can directly produce semantic anatomical labels without an agreed prompting/training setup.

## NasalSeg archive audit (user-reported, 2026-11-17)

- Extracted archive contains 130 NRRD images and 130 NRRD labels. The first sample pairs (`P001_img.nrrd` and `P001_seg.nrrd`) have matching size and spacing.
- Example image size is `153 × 205 × 52` voxels with spacing approximately `0.586 × 0.586 × 1.5 mm`.
- Label values are 0 through 5. They correspond to background, right/left maxillary sinus, right/left nasal cavity, and nasopharynx. No label is frontal sinus or FSDP.
- The downloaded image volume appears substantially smaller than the median dimensions reported in the dataset paper. We have not confirmed whether its field of view contains the frontal sinus.
- User cannot perform visual review. Proceed with TotalSegmentator pseudo-label generation as a public pilot, while automatically reporting empty masks, inference failures, and geometry mismatches. This does not replace expert review or establish anatomical correctness.

## Open-data research: current findings

Searches for public CT datasets labeled for the frontal sinus drainage pathway or frontal recess did not identify an openly downloadable, annotated dataset for the pathway itself. A useful additional finding is that an open-source model can produce a frontal sinus mask, and a sinus CT dataset may contain relevant surrounding anatomy; neither finding establishes a public ground-truth FSDP training set.

| Resource | What it contains | Access and fit for FSDP |
|---|---|---|
| [NasalSeg, Zenodo 13893419](https://zenodo.org/records/13893419) | 130 CT scans; five labels: bilateral nasal cavities, nasopharynx, and bilateral maxillary sinuses. | Open access, but no frontal sinus or FSDP label. Not a direct supervised FSDP dataset. |
| [CT-SCOPE, Zenodo 15085103](https://zenodo.org/records/15085103) | Paranasal sinus CT from 40 subjects; 13 subjects have manual slice masks of surrounding osseous structures, with additional pseudo-labels. | The record currently marks the files as restricted. It is not presently an openly downloadable dataset, and the labels are osseous structures rather than FSDP. |
| [Frontal sinus CBCT segmentation study](https://www.nature.com/articles/s41598-024-62211-y) | Reports radiologist-labeled frontal sinus masks on 50 CBCT scans for a frontal sinus segmentation task. | The study is not evidence that the scans and masks are publicly downloadable. The target is the frontal sinus, not its drainage pathway. Data availability needs direct confirmation from the authors/repository before considering it. |
| [TotalSegmentator `craniofacial_structures`](https://github.com/wasserth/TotalSegmentator) | Its released CT model outputs a `sinus_frontal` class, along with other craniofacial structures. | A useful frontal-sinus inference baseline or candidate pseudo-label source. The repository says this task is trained on a small separate dataset and may be less robust. Public model/code availability does not mean the training images and ground-truth masks for this class are available as a dataset. Its task-specific model license also needs review before use. |
| [Deep learning multi-class paranasal sinus study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10975213/) | Labels frontal, ethmoid, sphenoid, and maxillary sinus regions in a clinical CT study. | The paper describes a collected clinical dataset; this search did not establish public access to the scans and masks. Labels do not isolate FSDP. |
| [SinusSegment](https://github.com/rheadkaul/SinusSegment) | Open segmentation code for five paranasal sinuses and nasal cavity. | Repository instructions expect users to supply CT volumes and masks. Code availability is not dataset availability; target is sinus/cavity segmentation, not FSDP. |

Sources reviewed also describe NasalSeg's exact label map and CT-SCOPE's file restrictions: [NasalSeg paper](https://www.nature.com/articles/s41597-024-04176-1), [CT-SCOPE paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12398895/).

### Research conclusion

Based on the sources checked so far, there is no confirmed openly downloadable dataset with ground-truth FSDP annotations ready for supervised training and direct FSDP evaluation. There is at least one available model that predicts frontal sinus masks (TotalSegmentator), and there are research datasets/studies with frontal sinus labels, but the identified annotated cohorts are not openly downloadable. A frontal sinus label is not a valid substitute for a drainage-pathway label. This conclusion is limited to the datasets and publications located in this research pass; it is not proof that no such dataset exists anywhere.

The working public-data direction is to use suitable public CT images with TotalSegmentator frontal-sinus pseudo-labels for a SAM-Med3D pipeline experiment. This provides no independent ground-truth evaluation. Agreement with the pseudo-labels must not be reported as clinical or expert-label accuracy.

## Proposed public-to-private experiment path

Given the current data situation, the practical candidate is a pseudo-label public experiment, followed by fine-tuning and evaluation on the private ground-truth labels:

1. **Public image source:** start with the open NasalSeg CT images, after inspecting scan coverage and confirming that the frontal sinus is included at adequate resolution. Do not assume its published masks label the frontal sinus; they do not.
2. **Public pseudo-label generation:** run TotalSegmentator's `craniofacial_structures` task and extract `sinus_frontal`. Visually inspect a sample of masks for coverage, laterality/label convention, and gross failures. These are model-generated labels, not expert ground truth.
3. **Public model experiment:** train/fine-tune SAM-Med3D on the public CT images and accepted pseudo-labels. Report this as learning from pseudo-labels. A score against the same pseudo-label source measures agreement with TotalSegmentator, not anatomical accuracy.
4. **Private supervised stage:** fine-tune using the private `.nii.gz` images and their existing frontal sinus masks, once label semantics, geometry, and train/validation/test split are checked. Keep a patient-level held-out test set untouched until final evaluation.
5. **Final evaluation:** report results against private expert labels on the held-out test set. Keep the public pseudo-label agreement results separate from this ground-truth evaluation.

The working design is approved for dataset preparation, but model training configuration is still open. The main go/no-go checks are whether NasalSeg images include the target region and whether TotalSegmentator's sinus output is anatomically acceptable on those scans. If either fails, seek access to the CBCT cohort or move to annotation/access planning rather than treating pseudo-labels as truth.

The TotalSegmentator project also links a public 1,939-CT segmentation dataset, but that dataset description covers 117 structures and does not establish that the `sinus_frontal` task's separate training labels are included. Do not assume it is a direct frontal-sinus training dataset.

## Experiment design decisions still open

- What exactly counts as FSDP: anatomical boundaries, laterality, and whether the output is a binary mask, multiple classes, or a pathway/landmark representation.
- Whether the public dataset contains usable labels for the selected task.
- SAM-Med3D checkpoint and license to use for the selected experiment.
- Training from a released checkpoint versus training from scratch; which parameters to fine-tune.
- Data split unit and strategy, including patient-level separation and any external validation.
- Preprocessing: voxel spacing, orientation, intensity windowing, cropping, and patch/volume strategy.
- Metrics and reporting, for example per-class Dice and surface-distance measures, selected to match the clinical geometry and task.
- Available compute (GPU model and memory), training time, and storage limits.
- How expert annotations or adjudication will be obtained if the public labels do not cover the target.

## Proposed record of each run

For every approved run, record the dataset version and checksum, subject IDs per split, label mapping, preprocessing parameters, repository commit, checkpoint source, configuration, random seed, hardware, metrics, and known failure cases. Keep private data and its identifiers out of public logs and artifacts.

## Immediate preparation gate

Run the NIfTI audit on the public images and private image/mask pairs on DGX. Review image coverage, voxel spacing/orientation, and the private mask value map before resampling, cropping, pseudo-label generation, or any patient split. Training starts only after these findings are recorded and the split protocol is agreed.
