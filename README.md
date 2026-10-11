# Adversarial Attack Challenge for Secure Face Recognition 2025: SaeidUCC entry

Code of team **SaeidUCC** (Saeid Rezaei, University College Cork) for the detection and
resilience tracks of the **Adversarial Attack Challenge for Secure Face Recognition 2025**,
held at IJCB 2025. It implements the method submitted to the challenge and reported in the
paper below, which describes the challenge, its data and the results of all teams:

> J. Tremoço, I. Medvedev, N. Freitas, A. Costa, D. Nunes, N. Bunzel, L. Graner, N. Göller,
> L. Pellegrini, N. Di Domenico, G. Borghi, M. Verghese, S. Bhilare, A. Hati, M. Lourenço and
> N. Gonçalves, "Adversarial Attack Challenge for Secure Face Recognition 2025," in *2025 IEEE
> International Joint Conference on Biometrics (IJCB)*, 2025.
> [doi:10.1109/IJCB65343.2025.11411446](https://doi.org/10.1109/IJCB65343.2025.11411446)
> ([publication page](https://visteam.isr.uc.pt/publications/adversarial-attack-challenge-for-secure-face-recognition-2025/))

The results of this entry and of the other teams are in Tables 4 to 6 of the paper.

## The two tasks

- **Detection.** Decide whether a face image is clean or adversarial. Scored with APCER,
  BPCER, F1 and ROC AUC.
- **Resilience.** Verify pairs of face images when one image has been attacked, either to
  break a genuine match (evasion) or to create a false one (impersonation). Scored with the
  equal error rate (EER), the attack success rate (ASR) and the robustness score
  (1 − EER)(1 − ASR).

The training data were built from CelebA and LFW, with adversarial images from ten attack
methods crafted on a surrogate face recognition model. For the challenge evaluation the models
are served by a small REST API on port 7007 and queried by two client scripts, both included
here.

## Method

**Detection track.** The detector is an image classifier: a ResNet-18 initialised with
ImageNet weights, whose classification layer is replaced by a single logit. It is trained on
the clean and attacked training images, resized to 224 × 224, with binary cross-entropy, Adam
(learning rate 1e-4) and cosine annealing for 100 epochs. The sigmoid of the logit is the
adversarial score, and an image is flagged when the score reaches the decision threshold
(0.5 by default).

**Resilience track.** The recognition model is a Siamese network. Its two branches share one
encoder, a ResNet-18 (ImageNet initialisation) followed by a linear layer that produces a
512-d embedding, which is L2-normalised; inputs are resized to 500 × 500. The encoder is
trained on the genuine and impostor pairs provided by the organisers with a cosine contrastive
loss that pulls genuine pairs towards a similarity of 1 and pushes impostor pairs below 0.5.
Robustness comes from PGD adversarial training (Madry et al., 2018): for every batch, an
L∞-bounded perturbation is computed for one image of each pair so that genuine pairs move apart
and impostor pairs move together, mimicking the evasion and impersonation attacks, and the
model is trained on the clean and the perturbed pairs. Two images are compared with the cosine
similarity of their embeddings.

## Additions since the challenge

With their default settings the training scripts build the models of the challenge entry
(architecture, input size, loss and, for resilience, PGD adversarial training). The options
below were added afterwards and were not part of the evaluated submission.

Detection
- Any [timm](https://github.com/huggingface/pytorch-image-models) backbone (ConvNeXt,
  EfficientNet, ViT, ...).
- Extra input streams stacked with the RGB channels: SRM high-pass noise residuals and the
  log-magnitude Fourier spectrum, where adversarial perturbations are easier to see.
- Attack-type labels: one class per attack list, with 1 − p(clean) as the adversarial score.
- Synthetic attacks on clean training images (grid occlusion, most-significant-bit flips,
  uniform noise, patch noise) as extra adversarial samples.
- Focal loss, class weighting, label smoothing and class-balanced sampling.
- A decision threshold chosen on the validation split (EER point, Youden index, best F1 or a
  target BPCER) and stored with the model.

Resilience
- IResNet face backbones (18 to 200 layers) whose parameter names match the InsightFace
  releases, next to all timm backbones.
- BN and MLP embedding heads; double-margin and logistic pair losses; online hard pair mining.
- Single-step "fast" adversarial training (FGSM with random start) as a cheaper alternative
  to PGD, and validation EER under a white-box PGD attack (`adv_eer`). Both perturb the
  network input, after resizing and any input purification, so `adv_eer` measures the
  network and not what the purification adds.
- Corruption augmentation and input purification (Gaussian blur, median filter, JPEG,
  bit-depth reduction). The purification is stored in the checkpoint and applied in the
  same way by the API.

Both tracks
- A validation split with the challenge metrics, best-checkpoint selection and early stopping.
- Mixed precision, EMA of the weights, warm-up, step and plateau schedules, gradient
  clipping, a frozen-backbone warm-up and a separate backbone learning rate.
- Multi-GPU training (`DataParallel`), resumable runs, and a checkpoint, predictions and a
  report line for every epoch.
- Flip test-time augmentation and model ensembles in the API.
- Data checks: training stops when the attack lists do not match the image files or an image
  cannot be decoded, instead of silently treating unmatched images as clean.

Checkpoints written by the original challenge scripts (`detection_model_final.pt`,
`resilience_best.pt`) still load in the API.

## Repository layout

```
aac/                          shared code
  attacks.py                  PGD and FGSM on image pairs (adversarial training)
  checkpoint.py               checkpoint and JSON input/output
  client.py                   HTTP helpers for the evaluation clients
  corruptions.py              synthetic attacks, corruptions and input purification
  data.py                     reading the challenge data, datasets and splits
  inference.py                helpers shared by the inference classes
  iresnet.py                  IResNet face backbones (InsightFace layout)
  losses.py                   detection and pair losses
  metrics.py                  APCER, BPCER, F1, AUC, EER, ASR, TAR@FAR
  models.py                   detection and embedding networks
  protocols.py                challenge protocol files
  training.py                 devices, schedules, EMA, reports, resumable state
  transforms.py               image loading and preprocessing
train_detection_track.py      training, detection track
train_resilience_track.py     training, resilience track
classifier.py                 detection inference (Classifier)
template_extractor.py         resilience inference (TemplateExtractor)
main.py                       REST API (port 7007)
run_detection_track.py        evaluation client, detection track
run_resilience_track.py       evaluation client, resilience track
evaluate_predictions.py       local scoring with the challenge metrics
tests/                        unit and end-to-end tests on synthetic data
weights/                      checkpoints served by the API
Dockerfile                    CPU image of the API
```

## Installation

Python 3.9 or newer. Install the PyTorch build that matches your CUDA driver first, then the
remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate

# CUDA 12.x driver (e.g. 12.2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.x driver (e.g. 11.4)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 "numpy<2" \
    --extra-index-url https://download.pytorch.org/whl/cu113
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install umap-learn matplotlib    # only needed for umap_plot = True
```

The same code runs on PyTorch 1.12 and 2.x; the test suite passes with PyTorch 1.12.1
(Python 3.9) and PyTorch 2.x (Python 3.13).

## Data

**Detection** (`data_root` in `train_detection_track.py`, default `./AdvCelebA`):

```
AdvCelebA/
  images/                         clean and attacked images
  final_attack_attackid_0.txt     one list per attack method; the first column
  final_attack_attackid_1.txt     holds the name of an attacked image
  ...
```

Every image named in a list is adversarial (and belongs to that attack); every other image
is clean. List fields may be separated by spaces, tabs, commas or semicolons, and names are
matched by path, file name or file name without extension.

If the lists also give the clean source image of each attacked image, set `source_column` to
that column. An image and its attacked copies then stay on the same side of the
train/validation split; without it, a model that memorises image content gets a misleading
validation score.

**Resilience** (`data_root` in `train_resilience_track.py`, default `./resilience_pairs`):

```
resilience_pairs/
  pair_0_label_1/    im_0.png  im_1.png     label 1: same identity (genuine, evasion setting)
  pair_1_label_0/    im_0.png  im_1.png     label 0: different identities (impostor, impersonation setting)
  ...
```

## Training

All options are in the `CONFIG` dictionary at the top of each training script, with the
accepted values written next to every option. Edit it and run:

```bash
python train_detection_track.py
python train_resilience_track.py
```

Each run creates `runs/<track>/<timestamp>/` with:

| file | content |
|---|---|
| `config.json`, `training_report.txt` | all parameters, then one line per epoch (learning rate, losses, validation metrics, threshold) |
| `history.csv`, `train.log` | the same per-epoch values as a table, and the full console log |
| `epoch_XXX.pt` | checkpoint of every epoch (`save_every_epoch`) |
| `epoch_XXX_val.json` | validation scores or similarities of every epoch |
| `epoch_XXX_detection_track.json`, `epoch_XXX_resilience_track.json` | API-format outputs for `submission_dir`, if set |
| `best.pt` | checkpoint with the best `selection_metric` (the latest epoch while the metric is undefined or without validation) |
| `last_state.pt` | model, optimiser, scaler, EMA and scheduler state, for `resume` |
| `umap_val_embeddings.png` | resilience only, with `umap_plot = True` |

`gpu_ids` selects the GPUs (`[0, 1, 2, 3]` trains on four with `DataParallel`, `[]` uses the
CPU). An interrupted run continues from where it stopped with
`"resume": "runs/<track>/<run>/last_state.pt"`; options changed since the start of the run are
listed in the log.

Mixed precision (`amp`) and the validation split (`val_fraction`) are on by default. To train
as for the challenge submission, set `amp = False` and `val_fraction = 0.0`.

The results in the paper suggest a few settings to try first:

- Detection: `label_mode = "attack_type"` (the best detector in the challenge was trained per
  attack type), `streams = ["rgb", "srm"]`, `threshold_mode = "eer"`, and
  `synthetic_attack_prob = 0.2`, since the unseen grid-occlusion attack of the test set was
  the hardest one to detect.
- Resilience: a face-pretrained backbone, e.g. `backbone = "iresnet50"`, `image_size = 112`,
  `normalization = "face"`, `head = "identity"` and `backbone_weights` pointing to IResNet-50
  weights, together with `input_defense = "gaussian_blur"` (the strongest resilience entry used
  a face-pretrained model and Gaussian blur). The challenge attacks were crafted on
  InsightFace's ArcFace model, so do not start from that model: its adversarial examples would
  be white-box attacks on yours. Check the licence of any pretrained face model; InsightFace's
  released models are for non-commercial research use.

## Serving the models

Copy the checkpoints to `weights/detection.pt` and `weights/resilience.pt` (or point the
environment variables below to them) and start the API:

```bash
cp runs/detection/<run>/best.pt weights/detection.pt
cp runs/resilience/<run>/best.pt weights/resilience.pt
python main.py
```

| endpoint (GET) | parameters | response |
|---|---|---|
| `/detection/detect` | `image_path` | `score` (adversarial probability), `decision` (true = adversarial), `comment` |
| `/resilience/extract_template` | `image_path` | `data` (template as a list), `comment` |
| `/resilience/compare_templates` | `template_1`, `template_2` (JSON lists) | `similarity`, `comment` |
| `/resilience/compare_images` | `image_path_1`, `image_path_2` | `similarity`, `comment` |
| `/health` | | which models are loaded |

Errors are returned in the response (`"ERROR"` values with the message in `comment`).

| variable | default | meaning |
|---|---|---|
| `AAC_DETECTION_WEIGHTS` | `weights/detection.pt` | checkpoint(s), comma-separated for an ensemble; `none` disables the track |
| `AAC_RESILIENCE_WEIGHTS` | `weights/resilience.pt` | same for the resilience track |
| `AAC_DETECTION_THRESHOLD` | from the checkpoint, else 0.5 | decision threshold |
| `AAC_DEVICE` | `cuda` if available, else `cpu` | inference device |
| `AAC_TTA` | from the checkpoint | `1` averages each image with its mirror image |
| `AAC_INPUT_DEFENSE`, `AAC_DEFENSE_PARAM` | from the checkpoint | override the input purification |
| `AAC_IMAGE_ROOT` | unset | if set, only images inside this folder can be read |
| `AAC_HOST`, `AAC_PORT` | `0.0.0.0`, `7007` | bind address |

With Docker (CPU image). Copy the checkpoints into `weights/` first, as above; the evaluation
clients run inside the container so that the image paths they send are paths the API can read:

```bash
docker build -t aac-saeiducc .
docker run -d --name aac -p 7007:7007 -v /path/to/test_images:/data aac-saeiducc
docker exec -e images_dir=/data -e save_dir=/data/out aac python run_detection_track.py --wait 60
docker exec -e images_dir=/data -e save_dir=/data/out aac python run_resilience_track.py
```

## Evaluation clients

Two clients query the API as in the challenge evaluation. Each reads a protocol file from
`images_dir` and writes its results to `save_dir`:

```bash
images_dir=/data/test save_dir=/data/out python run_detection_track.py
images_dir=/data/test save_dir=/data/out python run_resilience_track.py
```

`detection_track.txt` lists one image per line and gives `detection_track.json`;
`resilience_track.txt` lists two images per line and gives `resilience_track.json`. Both
clients also accept `--images-dir`, `--save-dir`, `--url` and `--wait` (seconds to wait for
the API to start).

## Local scoring

With ground-truth labels, `evaluate_predictions.py` computes the challenge metrics from the
client outputs:

```bash
python evaluate_predictions.py detection --predictions /data/out/detection_track.json --labels labels.txt
python evaluate_predictions.py resilience --predictions /data/out/resilience_track.json --labels pairs.txt
```

`labels.txt` holds `<image> <label> [attack]` per line (1 or `adversarial`, 0 or `clean`).
`pairs.txt` holds `<image 1> <image 2> <label> [attacked] [scenario]` per line (1 or `genuine`,
0 or `impostor`; `attacked` is 1 when one image of the pair was attacked). Images are given as
paths relative to `images_dir`. The paper names the metrics but not every detail of how they
were computed; here the EER is taken over all labelled pairs and the ASR at the EER threshold
of the bona fide pairs (or at `--threshold`). These are choices of this script, not the
organisers' scoring code, so numbers on the official test set may differ.

## Tests

```bash
pip install pytest httpx
pytest
```

The tests use small synthetic datasets and run on the CPU in under a minute. They cover the
metrics, losses and attacks, the data readers, short training runs of both tracks (including
resuming), the API, local scoring and loading of the original challenge checkpoints. The code
is formatted with black and checked with ruff (settings in `pyproject.toml`).

## Citation

If you use this code, please cite the challenge paper:

```bibtex
@inproceedings{tremoco2025adversarial,
  title     = {Adversarial Attack Challenge for Secure Face Recognition 2025},
  author    = {Tremo{\c{c}}o, Jo{\~a}o and Medvedev, Iurii and Freitas, Nuno and Costa, Andreia and
               Nunes, Diogo and Bunzel, Niklas and Graner, Lukas and G{\"o}ller, Nicholas and
               Pellegrini, Lorenzo and Di Domenico, Nicol{\`o} and Borghi, Guido and
               Verghese, Monson and Bhilare, Shruti and Hati, Avik and Louren{\c{c}}o, Miguel and
               Gon{\c{c}}alves, Nuno},
  booktitle = {2025 IEEE International Joint Conference on Biometrics (IJCB)},
  year      = {2025},
  doi       = {10.1109/IJCB65343.2025.11411446}
}
```

## Acknowledgements

Thanks to the organisers of the challenge, Youverse and the Institute of Systems and Robotics
of the University of Coimbra, for the data, the attack package and the evaluation.
