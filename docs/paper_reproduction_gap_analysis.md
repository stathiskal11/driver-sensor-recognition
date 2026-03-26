# Paper Reproduction Gap Analysis

Paper: "Incorporating Gaze Behavior Using Joint Embedding With Scene Context for Driver Takeover Detection" (ICASSP 2022)

## Purpose

This document tracks how closely the current repository matches the paper baseline, before any thesis-specific modification is introduced.

## Quick Status

- Dataset inspection: implemented
- Window index generation: implemented
- Multimodal dataset loader: implemented
- 3D-CNN baseline model: implemented
- Participant-independent split: implemented
- Training loop: implemented
- ROC AUC evaluation: implemented
- Five-partition evaluation protocol: implemented
- Paper-faithful label definition: partially implemented
- Exact six CAN-Bus plus physiology signals: implemented
- Paper-faithful preprocessing and normalization: partially implemented
- Full baseline experiment matching the paper setup: not implemented yet

## Comparison Against The Paper

### 1. Dataset protocol

Paper:
- 32 participants collected
- 4 removed because of data errors
- 28 effective participants
- 4 sessions per participant
- 20/4/4 participant-independent split
- 5 shuffled train/validation/test partition groups

Current repo:
- We verified 28 participant IDs in the local HDBD archive
- We verified 112 participant-level CSV files, which matches 28 participants x 4 sessions
- We implemented participant-independent splits with 20 train, 4 validation, 4 test participants
- We implemented support for running multiple shuffled split groups from the training script
- We verified the `5` split-group protocol in report-only mode

Status:
- Matched at protocol level
- Full multi-split training remains computationally heavy on CPU, so practical large runs are still pending

### 2. Target task

Paper:
- Binary classification
- Predict whether the driver takes over at the final timestamp of a selected driving segment
- Detection task uses look-back = 3 seconds
- Forecast experiments also evaluate 1 second, 3 seconds and 5 seconds ahead

Current repo:
- We implemented binary classification
- We use 3-second windows at 10 Hz, so 30 timesteps
- We currently use a working label rule: positive if a `main_keydown` occurs within the next 10 timesteps
- This gives 1.0802% positives, close to the paper's reported 1.0%
- We implemented a label-candidate analysis utility and compared keydown-based versus throttle-onset based rules on the local HDBD archive
- The released archive strongly favors the 1-second future-keydown rule over throttle-onset rules as the best current paper proxy

Status:
- Structurally close
- Exact detection label at the final timestamp is still not fully confirmed from the paper text alone
- Forecast setups at 3 seconds and 5 seconds are not implemented as experiment presets yet

### 3. Scene plus gaze modality

Paper:
- Uses semantic segmentation images and gaze heatmaps
- Treats them as two grayscale layers
- Input shape: `(30, 90, 160, 2)`

Current repo:
- We load segmentation images from `seg_img_90_160_new_dash.tar.gz`
- We load precomputed gaze heatmaps from `Heat_maps_90_160_sigma_64.tar.gz`
- We stack them into `scene_gaze` with shape `(2, 30, 90, 160)` in PyTorch channel-first format

Status:
- Matched for the baseline structure
- Close to paper, with PyTorch axis order instead of the paper's tabular notation

### 4. CAN-Bus and physiology modality

Paper:
- Dense branch input is 180 values
- Defined as `6 signals x 3 seconds x 10 Hz`

Current repo:
- We implemented the dense branch with flattened input length 180
- We now use the exact 6 signals named by the paper:
  - `ECGtoHR`
  - `GSR`
  - `Throttle`
  - `RPM`
  - `Steering`
  - `Speed`
- We normalize `ECGtoHR` and `GSR` per participant using cached statistics over all sessions
- We normalize the four CAN-Bus signals to the `0..1` range using cached min-max statistics

Status:
- Matched

### 5. HMI complementary modalities

Paper:
- Uses HMI information level
- Uses navigational information
- Uses weather information
- Concatenates one-hot embeddings with learned features

Current repo:
- We implement one-hot encoding for:
  - `navigation`
  - `transparency`
  - `weather`
- This produces a 9D HMI vector

Status:
- Matched

### 6. 3D-CNN architecture

Paper Table 1:
- Input 2 channels
- Conv3D 32
- MaxPool3D
- Conv3D 64
- MaxPool3D
- Conv3D 128
- MaxPool3D
- Conv3D 256
- MaxPool3D
- Flatten 10240
- Linear 256
- Dropout 0.5
- Linear 256

Current repo:
- Implemented in `src/models/joint_embedding.py`
- Matches the paper layer pattern and sizes
- Uses ReLU activations as described in the paper

Status:
- Matched

### 7. Dense signal branch

Paper:
- Fully connected layers `128 -> 64 -> 32`
- ReLU activations

Current repo:
- Implemented exactly with `128 -> 64 -> 32`

Status:
- Matched

### 8. Final DNN classifier

Paper:
- Fully connected layers `512 -> 512 -> 256 -> 1`
- ReLU on hidden layers
- Sigmoid on final output

Current repo:
- Implemented as `Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear`
- Final sigmoid is applied through `BCEWithLogitsLoss` or `predict_proba`

Status:
- Matched

### 9. Training protocol

Paper:
- 20 epochs
- Adam
- learning rate `0.001`
- batch size `32`

Current repo:
- Adam implemented
- learning rate default `0.001`
- epochs and batch size are configurable
- current sanity runs use smaller values due CPU constraints

Status:
- Matched at configuration level
- Not yet matched in a full paper-style training run

### 10. Evaluation protocol

Paper:
- Main metric: ROC AUC
- Uses 5 shuffled participant-independent partitions
- Reports ROC/AUC aggregated across the 5 models

Current repo:
- ROC AUC implemented
- Validation and optional test evaluation implemented
- Split reporting implemented
- Balanced small debug subsets implemented for sanity runs

Status:
- Partially matched
- Missing the exact 5-partition aggregated evaluation procedure

### 11. Preprocessing details

Paper:
- modalities synchronized and down-sampled to 10 Hz
- HR and GSR z-normalized per participant using all 4 sessions
- road scene images and gaze points interpolated with nearest-neighbor interpolation
- CAN-Bus and physiology linearly interpolated
- gaze heatmaps created from 5 consecutive gaze points
- default baseline uses Gaussian heatmaps

Current repo:
- We rely on the already synchronized participant-level CSV files
- We rely on precomputed segmentation images and precomputed heatmaps from the archive
- We now perform participant-level HR/GSR normalization in the dataset loader
- We do not yet regenerate heatmaps from raw gaze points
- We do not yet explicitly reproduce interpolation choices
- We currently use sigma-64 Gaussian heatmaps by default, which is consistent with the paper's default comparison setting
- We normalize the four selected CAN-Bus signals in the loader

Status:
- Partially matched
- Several preprocessing steps are assumed to already be reflected in the released archive, but not independently reimplemented

## What Is Already Strong

- The repository already has a real end-to-end baseline path
- The main multimodal architecture from the paper is already in place
- The core input structure of the paper is already represented
- The split logic is participant-independent, which avoids leakage
- We can already run train/validation/test passes and compute ROC AUC

## What Is Still Risky

- The label rule is not fully verified against the exact detection setup of the paper
- The exact six signals used in the dense branch are still inferred
- The evaluation is not yet run over 5 shuffled partition groups
- The preprocessing steps from the paper are not all reproduced from scratch
- We have not yet produced a full baseline result that can be honestly compared to AUC = 0.8615

## Practical Estimate Of Completion

If we divide the paper reproduction into two levels:

- Engineering completion: "can the baseline pipeline run end-to-end?"
- Scientific completion: "can we fairly claim a paper-faithful reproduction?"

Then the current status is:

- Engineering completion: high
- Scientific completion: medium

A practical estimate is:

- about 75% complete as an end-to-end engineering baseline
- about 65-70% complete as a paper-faithful reproduction

The reason for the remaining gap is not architecture anymore. The main open items are exact labels, full multi-split experiments, and a tighter reproduction of preprocessing details that are only partially described in the released materials.

## What Should Happen Next

1. Lock down the exact label definition for the 0-second detection task.
2. Add experiment logging and checkpoint saving.
3. Run the first full baseline experiments with the paper training setup across multiple shuffled splits.
4. Compare the resulting ROC AUC against the paper's reported 0.8615.
5. Only after that, branch into thesis-specific modifications.
