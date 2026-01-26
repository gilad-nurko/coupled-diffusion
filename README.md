# Coupled Diffusion

This repository implements **Coupled Diffusion**, a domain-agnostic framework that integrates two interacting diffusion models—one operating on the input signal and another on the classifier's output logits—to enable mutual guidance during enhancement. This formulation allows the evolving class logits to guide signal reconstruction towards discriminative regions while the enhancing signal refines the class estimation.

## Joint Diffusion Strategies

The following strategies are implemented for both image and audio experiments to model the joint distribution of the input and logits:

* **Parallel**: Uses an *interleaved scheduler* performing a single joint diffusion loop where timesteps alternate between signal and logit updates. Each update is guided by the most recent prediction of the other process.
* **Alternating**: Uses a *block scheduler* that decouples the processes temporally. It executes a full signal diffusion trajectory followed by a full logit trajectory, repeating this for multiple iterations using clean estimates from the previous step.
* **Nested**: Uses a *hierarchical scheduler* that performs a single logit diffusion loop while embedding a complete signal diffusion process within each logit timestep to ensure high-quality signal estimation.
* **Enhanced**: A baseline strategy that performs regular diffusion for signal enhancement and subsequently applies a classifier to the result.

---

## Image Experiments (ImageNet-32)

Experiments are conducted on a 100-class subset of ImageNet-32.

### 1. Pretraining

Before running the coupled training strategies, you must first train the classifier and then the logit diffusion model to generate the required `pretrained_model_y_ckpt`.

**Step A: Train the Classifier**

```bash
python images/classifiers/classifier_imagenet32_100subset.py \
    --imagenet32_root /path/to/imagenet32 \
    --save_dir /path/to/save_classifier

```

**Step B: Train the Logits Model**

```bash
python images/pretrain_logits_model_imagenet32_100subset.py \
    --imagenet32_root /path/to/imagenet32 \
    --classifier_ckpt /path/to/classifier_model \
    --results_dir /path/to/save_pretrained_logits_model

```

*This step generates the `pretrained_model_y_ckpt` used in the main training scripts.*

### 2. Coupled Training

We provide training scripts for all four strategies:

* `images/train_imagenet32_100subset_parallel.py`
* `images/train_imagenet32_100subset_alternating.py`
* `images/train_imagenet32_100subset_nested.py`
* `images/train_imagenet32_100subset_enhanced.py`

**Training Example (Parallel):**

```bash
python images/train_imagenet32_100subset_parallel.py \
    --imagenet32_root /path/to/imagenet32 \
    --classifier_path /path/to/classifier_model \
    --pretrained_model_y_ckpt /path/to/logits_model \
    --results_dir ./results

```

---

## Audio Experiments (EARS)

Audio experiments focus on speech enhancement and de-reverberation using the EARS corpus.

### 1. Dataset Generation

Prepare the **EARS WHAM** (noise) or **EARS Reverb** (reverberation) datasets using these scripts:

```bash
# Example for EARS WHAM
python audio/create_ears_WHAM/generate_ears_wham.py \
    --data_dir /path/to/save/generated_dataset

# Example for EARS Reverb
python audio/create_ears_reverbed/generate_ears_reverb.py \
    --data_dir /path/to/save/generated_dataset

```

### 2. Logits Model Pretraining

Before running the coupled training for audio, you must pretrain the logits diffusion model to generate the required `logits_pretrain_ckpt`.

```bash
# Example for WHAM
python audio/coupled_enhancement/pretrain_eps_pred_logits_model_wham.py \
    --base_dir /path/to/generated_EARS_WHAM_dataset \
    --logits_pretrain_ckpt /path/to/save_pretrained_logits_model \
    --transcripts_path /path/to/audio_transcriptions \

```

### 3. Coupled Training

Once the logits checkpoint is ready, you can run the main training script:

* **EARS WHAM**: `audio/coupled_enhancement/train_wham.py`
* **EARS Reverb**: `audio/coupled_enhancement/train_reverb.py`

**Training Example (EARS WHAM):**

```bash
python audio/coupled_enhancement/train_wham.py \
    --base_dir /path/to/generated_EARS_WHAM_dataset \
    --pretrained_ckpt /path/to/audio_model \
    --logits_pretrain_ckpt /path/to/pretrained_logits_model \
    --log_dir ./logs/coupled_train \
    --transcripts_path /path/to/audio_transcriptions \

```

### 4. Inference (Enhancement)

Enhance noisy audio files with your trained model:

```bash
python audio/coupled_enhancement/enhancement.py \
    --test_dir /path/to/noisy_wavs \
    --enhanced_dir ./output \
    --ckpt /path/to/checkpoint.ckpt \
    --transcripts_path /path/to/audio_transcriptions \
    --N 50

```

### 5. Baseline Strategy

The implementation for the **Enhanced** (sequential) audio baseline is available in the `audio/regular_enhancement` directory.