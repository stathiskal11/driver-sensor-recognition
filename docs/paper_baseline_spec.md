# Paper Baseline Spec

Paper: "Incorporating Gaze Behavior Using Joint Embedding With Scene Context for Driver Takeover Detection" (ICASSP 2022)

## Goal

Reproduce the paper baseline on HDBD as faithfully as possible before adapting the pipeline toward the thesis topic on driver recognition from non-visual vehicle signals.

## Task Definition

- Primary task: binary takeover detection.
- Prediction target: whether the driver takes over at the final timestamp of a selected driving segment.
- Default look-back window: 3 seconds.
- Sampling rate after synchronization: 10 Hz.
- Default sequence length: 30 timesteps.

The paper also reports forecasting performance 1, 3, and 5 seconds ahead, but the first implementation target should be the 0-second detection setup.

## Dataset Protocol From The Paper

- Original study participants: 32.
- Removed because of data errors: 4.
- Effective participants used in experiments: 28.
- Sessions per participant: 4.
- Each session: about 8 minutes, 10 intersections.
- Raw behavior setting: L2 automated urban driving with different transparency levels, maneuvers, and weather conditions.
- Takeover intent is reported by pressing the Spacebar.
- The paper states that 17.0% of intersections involve takeover and 1.0% of data points are labeled as takeover.

## Available HDBD Assets In The Local Archive

The local `hdbd.tar.gz` bundle includes:

- `Synced_csv_files-participant_level.tar.gz`
- `Synced_csv_files.tar.gz`
- `seg_img_90_160_new_dash.tar.gz`
- `SemanticSegmentationOrg.tar.gz`
- `Heat_maps_90_160_sigma_16.tar.gz`
- `Heat_maps_90_160_sigma_32.tar.gz`
- `Heat_maps_90_160_sigma_64.tar.gz`
- `Heat_maps_90_160_laplace.tar.gz`
- `Video.tar.gz`

For the baseline reproduction, the simplest path is to use the precomputed 90x160 segmentation images and the precomputed gaze heatmaps instead of regenerating them from scratch.

## Modalities Used By The Paper

### Scene And Gaze Joint Embedding

- Semantic segmentation images from the driving scene.
- Gaze heatmaps aligned to the same image coordinate system.
- The paper treats them as a 2-channel temporal input.
- Input shape reported in the paper: `(30, 90, 160, 2)`.

### CAN-Bus And Physiology

- The paper flattens a 3-second sequence into `180` values, described as `6 signals x 3 seconds x 10 Hz`.
- The extracted text available locally does not enumerate the exact six selected signals, so this must be confirmed during implementation from the dataset and paper details.

### Complementary HMI Metadata

- Navigation information.
- Transparency level.
- Weather condition.

These are concatenated as one-hot features with the learned embeddings.

## Preprocessing Notes From The Paper

- All modalities are synchronized and down-sampled to 10 Hz.
- Heart rate and GSR are z-normalized per participant using the full four-session history.
- Road scene images and gaze points are interpolated with nearest-neighbor interpolation.
- CAN-Bus and physiological signals are linearly interpolated.
- Gaze heatmaps are created from five consecutive gaze points.
- The default paper baseline uses Gaussian heatmaps.
- The paper compares multiple heatmap settings and reports the best ablation result with Laplace heatmaps, but the reproduced baseline should begin with the default Gaussian setting before ablations.

## Model Architecture To Reproduce First

### 3D-CNN For Joint Scene-Gaze Embedding

- Input: `(30, 90, 160, 2)`
- 3D-CNN: 32 channels, kernel `(3, 3, 3)`, stride `(1, 1, 1)`
- 3D max-pooling: kernel `(2, 2, 2)`, stride `(2, 2, 2)`
- 3D-CNN: 64 channels
- 3D max-pooling
- 3D-CNN: 128 channels
- 3D max-pooling
- 3D-CNN: 256 channels
- 3D max-pooling
- Flatten: `10240`
- Linear: `256`
- Dropout: `0.5`
- Linear: `256`

### Dense Module For CAN-Bus And Physiology

- Flattened input length: `180`
- Fully connected layers: `128 -> 64 -> 32`
- ReLU activations

### Final DNN Classifier

- Fully connected layers: `512 -> 512 -> 256 -> 1`
- ReLU on hidden layers
- Sigmoid on output layer

## Training Protocol To Match First

- Epochs: 20
- Optimizer: Adam
- Initial learning rate: `0.001`
- Batch size: `32`

## Evaluation Protocol To Match First

- Main metric: ROC AUC
- Split strategy: participant-independent
- Per split: `20` participants for training, `4` for validation, `4` for testing
- Number of shuffled partition groups: `5`

## Reported Baseline Result

- Proposed model AUC: `0.8615`

This number is the first target reference for the reproduction effort. We do not need to match it exactly in the first run, but our implementation should be close enough that any gap can be investigated systematically.

## Important Reproduction Risks

- Label definition is not the same as simply checking `main_keydown` in the raw CSV rows. The raw archive contains sparse key events, while the paper reports 1.0% positive data points. This suggests an additional label-construction step around takeover timing.
- The exact six CAN-Bus plus physiology signals used in the dense branch still need to be pinned down.
- The paper uses participant-independent splits. Random row-level or frame-level splits would cause leakage and invalid results.

## Current Working Label Assumption

For the current baseline index builder, we use this practical assumption:

- positive label if a `main_keydown` appears within the next `10` timesteps from the end of the window
- at 10 Hz, this corresponds to a `1.0s` prediction horizon

Why this is the current default:

- using only `final_timestep == main_keydown` yields about `0.47%` positives in the local archive
- using a `1.0s` future-keydown horizon yields about `1.08%` positives, which is much closer to the paper's reported `1.0%`

This is still an implementation assumption, not a fully confirmed fact from the paper text, so it should be treated as the current baseline hypothesis to test and refine.

## Immediate Implementation Milestones

1. Inspect the local HDBD bundle and verify how the CSV rows link to segmentation images and heatmaps.
2. Build a clean sample index for 3-second windows.
3. Define the paper-faithful label generation rule for the final timestep of each window.
4. Implement the dataset loader for the multimodal inputs.
5. Implement the 3D-CNN baseline and training loop.
6. Reproduce the paper detection setup before any thesis-specific change.

## Thesis Transition After Reproduction

Once the baseline is stable, the thesis-specific branch can replace or augment the takeover label with a driver-recognition objective such as:

- owner vs non-owner classification
- driver identity classification
- driving-style based user profiling

At that point, we can reuse the same project skeleton, data pipeline, and experiment tracking, while changing the target, the useful modalities, and the ablation plan.
