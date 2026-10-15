# ML Training Dashboard

A self-hosted dashboard for training and evaluating scikit-learn, XGBoost, LightGBM, PyTorch,
Stable-Baselines3 and Hugging Face models on any dataset or RL environment, from any device on your home network.

## Run it

```bash
./run.sh                 # builds the frontend if needed, serves on 0.0.0.0:8000 (keep the terminal open)
PORT=9000 ./run.sh       # another port
./run.sh --background    # detached; logs in data/server.log
./run.sh --stop          # stop the background dashboard
```

Open `http://<this-machine-ip>:8000` from any device. On first start a password is generated and
printed in the console (also stored in `data/config.json`); change it under **Settings**.

**LAN-only.** The server listens on all interfaces so devices at home can reach it; it is not exposed to the
internet unless your router forwards a port. On top of that, LAN-only mode (on by default, Settings → Network)
rejects any request that doesn't come from a private address. Add VPN ranges (e.g. Tailscale `100.64.0.0/10`)
there if you want access away from home.

Requirements: the `mltrain` conda env (Python 3.12, torch, sklearn, SB3, gymnasium, transformers…)
and Node ≥ 20 for building the frontend. Recreate the env with:

```bash
conda create -n mltrain python=3.12 && conda activate mltrain
pip install torch torchvision "fastapi[standard]" scikit-learn pandas "stable-baselines3[extra]" "gymnasium[box2d]" \
    transformers datasets evaluate sacrebleu rouge-score accelerate sentencepiece python-multipart itsdangerous psutil pyyaml \
    sb3-contrib xgboost lightgbm "imageio[ffmpeg]" opencv-python mcp optuna "psycopg[binary]" pymysql
```

### Keep it running (optional, systemd user service)

```ini
# ~/.config/systemd/user/mltrain.service
[Unit]
Description=ML Training Dashboard
[Service]
ExecStart=/mnt/Linux_Data_2/Projects/DL_dashboard/run.sh
Restart=on-failure
KillMode=process   # restarting the service leaves training runs alive
[Install]
WantedBy=default.target
```
`systemctl --user enable --now mltrain && loginctl enable-linger $USER`

Training runs are separate processes, so restarting the server does not kill them; the server
re-attaches to them on startup.

## Concepts

| Term | Meaning |
|---|---|
| **Problem** | A dataset (tabular / image / text) or an RL environment |
| **Model** | A plugin that knows how to train on compatible problems (matched by task + modality) |
| **Session** | One configuration (problem + model + params) trained on N seeds |
| **Run** | One seed of a session, executed as its own worker process |
| **Lineage** | "Train again" creates a new session with `parent_id` set; the Lineage tab shows the tree, parameter diffs and the headline metric over time |
| **Saved model** | `data/runs/<run>/model/` (+ `meta.json`), kept when *save_model* is on (default). Delete per seed or per session from the **Models** tab or Settings → Storage |
| **Best vs. final** | Option *keep* (default `both`): `model/` holds the best checkpoint by validation score, `model_final/` the last-epoch weights (PyTorch, SB3, HF). The final weights are scored on the same splits and shown as *Best vs. final* on the Overview — a gap means late overfitting. Fine-tune, evaluate, register and explain either one |
| **Fine-tune** | Train starting from a saved model (same or different problem); continues the lineage |
| **Evaluate** | Run saved models (or zero-shot pretrained checkpoints) on a dataset/env without training |
| **Group** | Several sessions launched together: a **sweep** (grid search), a **benchmark** (evaluate many models on one problem) or a **compare** (train several model types on one problem). Opens in Compare with a ranked table and chart |
| **Snapshots** | What the model does during training, with a step slider: episode videos (RL), generated text (seq2seq / text classifiers), prediction galleries + confusion matrices (images), decision boundaries (2-feature tabular) and fitted curves (1-feature regression) |

The job queue runs at most `max_gpu_jobs` GPU runs and `max_cpu_jobs` CPU runs at once (Settings).
**Stop** asks the run to finish gracefully: the current epoch ends, the best checkpoint is evaluated on the
test set and results are kept. **Resume** continues a stopped/failed PyTorch or SB3 run from its checkpoint
(metrics continue where they left off); **Restart** re-runs it from scratch.

RL sessions also offer a **live view** (frames from the training env streamed to the browser), periodic
**videos**, and an optional **GUI window** on the host desktop during evaluation episodes.

## Data: static datasets vs. live sources

* **Static datasets** (Data → Static datasets): downloaded / uploaded data that rarely changes gets a cached
  **quality report** — missing values, duplicates, constant and ID-like columns, outliers, class imbalance, possible
  target leakage, train/test rows that overlap, and whether the train/test split is representative. Shifts are only
  flagged when they exceed both the PSI threshold *and* the sampling-noise level for that split size.
* **Live sources** (Data → Live sources → Connect): a SQLite file, PostgreSQL/MySQL (SQLAlchemy URL) or a folder of
  CSV/Parquet files that keeps receiving rows, with a timestamp column (epoch s/ms or datetime; UTC or local time).
  Opened read-only. Profiled on a schedule; the health page shows freshness (stale alerts), rows per hour (drops /
  spikes), schema changes, missing values / new categories / out-of-range values on a short quality window, drift of
  every column (recent window vs. trailing reference), target mix per day and label coverage. Unhealthy sources send
  notifications.
* **Training on live data**: every live source with a target appears as the dataset `live.<id>` with
  time-window parameters — last N hours or a date range, a frozen snapshot (`until`), a time-based split (train on
  older rows, test on the newest), **recency weighting** (exponential half-life, linear, or a step ×k for the newest
  rows — measured from the newest training row; supported by sklearn, XGBoost, LightGBM and PyTorch) and optional
  downsampling of old rows. The run records the actual window and effective number of training rows.
* **Schedules** (New training → Runs & options → Repeat on a schedule): every N hours, daily at HH:MM or weekly;
  each run trains on a fresh frozen snapshot, can register itself in the model registry, and notifies you. Deployment
  automation can also retrain from a live source.
* **Demo**: Data → *Add demo live database* creates a SQLite table of loan applications with 30 days of history (the
  economy worsens over the last 10 days) and keeps streaming rows; outcomes arrive ~1 minute later. Inject incidents
  (missing values, a new category) or stop the stream to see health react.

## Cloud data (AWS, Google Cloud, Azure)

**Object storage.** Live sources and the "CSV / Parquet file" datasets accept cloud URLs as well as local paths:
`s3://bucket/prefix/`, `gs://bucket/*.parquet`, `az://container/folder` (also `abfs://`). A folder or glob picks up new files as they arrive.

**Warehouses.** Use the SQL source with the built-in URL templates (Data → Connect a source → SQL database / cloud warehouse):

| Warehouse | URL form |
|---|---|
| Amazon Redshift | `postgresql+psycopg://user:${REDSHIFT_PASSWORD}@cluster…redshift.amazonaws.com:5439/db` |
| Amazon Athena | `awsathena+rest://@athena.REGION.amazonaws.com:443/db?s3_staging_dir=s3://bucket/results/` |
| Google BigQuery | `bigquery://project/dataset` |
| Azure SQL / Synapse | `mssql+pymssql://user:${AZURE_SQL_PASSWORD}@server.database.windows.net:1433/db` |

**Credentials: the dashboard never stores secrets.**
- Each cloud's standard login is used:
  - AWS: `aws configure` / SSO profiles, environment variables, IAM roles
  - Google Cloud: `gcloud auth application-default login` or a service-account key file
  - Azure: `az login`, managed identity, or a connection string / account key read from an environment variable
- Settings → **Cloud data** stores only non-secret values (profile, region, key-file path, storage account, env-var names, custom endpoint). It shows whether each library and credential is found, and has a **Test** button for any URL. A live source can override these per source, e.g. a different AWS profile.
- Secrets can go in `data/cloud.env` (`KEY=value` lines, git-ignored, never displayed) and be referenced in SQL URLs as `${KEY}`. They're expanded only when connecting, and error messages are redacted.

The drivers come from the training environment: `s3fs`, `gcsfs`, `adlfs`, `azure-identity`, `sqlalchemy-bigquery`, `pyathena`, `pymssql`.

## Notifications, explainability, smart search, queue & power

* **Notifications** (Settings → Notifications): Discord, Slack, **ntfy** (free phone push), email (SMTP) or any webhook.
  Per channel choose events: session finished / failed, sweep · benchmark · search finished, deployment critical alert /
  warning, automated retraining (challenger ready / promoted / rejected). One message per finished group, rate-limited
  alerts, "Send test" button, delivery history.
* **Explainability** (session → **Explain**; deployment → Predictions → click a row):
  * Tabular: SHAP values on the *raw* input columns through the model's serving pipeline (works for sklearn, XGBoost,
    LightGBM and PyTorch alike) — global importance, beeswarm, and a waterfall for any single prediction, including
    live production predictions.
  * Images (PyTorch): Grad-CAM heatmaps on test images or images you upload.
* **Smart hyperparameter search** (New training → Hyperparameter search → *Smart search*): Optuna TPE with median
  pruning. Each trial is a normal session; the objective is a *validation* metric (test stays untouched); bad trials are
  stopped early from their live learning curves; results show optimization history, parameter importance and the best
  settings ("train with these settings"). Studies persist in `data/optuna.db`.
* **Queue priorities & scheduling**: every session has a priority (low · normal · high · urgent) and an optional start
  time. The Sessions page shows the queue in execution order with re-prioritize / reschedule / start-now.
* **Stay awake, screen off** (Settings → Power, on by default): while anything is running, queued or scheduled, the
  dashboard holds a logind *sleep* inhibitor (`systemd-inhibit --what=sleep`) so the PC doesn't suspend — the screen
  still dims, locks and turns off normally. "Turn the display off now" blanks the monitors on demand
  (`kscreen-doctor` on KDE, Mutter on GNOME, `xset` on X11). The inhibitor disappears if the dashboard stops.

## MLOps: registry, serving, monitoring, continuous training

```
 training session ──Register──▶ Model registry (immutable versions: model + fitted preprocessing + training profile + model card)
                                     │ Deploy
                                     ▼
 apps / simulator ──predict──▶ Deployment (served here)  ─┐        external service (exported package / your app)
                  ◀─feedback──                             ├─▶ prediction log ◀──push logs + labels (SDK)
                                                          ─┘          │
                        monitoring every N min: drift (PSI, KS, Wasserstein, JS), data quality,
                        prediction & score distribution, live performance (bootstrap CI), latency, errors
                                                                      │ health + alerts
                                                                      ▼
                        automation: retrain challenger on fresh data ─▶ evaluate champion on the same out-of-time split
                                    ─▶ quality gate ─▶ register ─▶ shadow / canary / approve / auto-promote ─▶ rollback
```

* **Registry** (`/registry`): *Register* on any session copies the model to `data/registry/<name>/v<N>/`, so deleting
  sessions never breaks deployments. Stages: registered → staging → production → archived.
* **Serving contract**: tabular models store their fitted preprocessing (imputation, one-hot, scaling) so endpoints
  accept **raw records** (`{"income": 52000, "home_ownership": "RENT"}`) exactly as training saw them — no
  train/serve skew. Images are sent as base64, text as strings, RL observations as lists.
* **Deployments** (`/deployments`): *served here* (`POST /api/serve/<id>/predict`, `X-API-Key` header) or *external*
  (run the exported package or instrument your own service; it pushes logs with the SDK — the dashboard never needs
  to reach into your infrastructure). Ground truth arrives later via `/feedback` (joined on `request_id`).
* **Monitoring**: each check compares a rolling window of the *champion's* traffic with its training profile.
  Tabular: per-feature drift + data quality (missing, out of range, unseen categories). Images: brightness, contrast,
  edge density… Text: length, punctuation, case, out-of-vocabulary rate. RL: observation dimensions + action mix.
  Live performance uses AUC for binary classifiers (accuracy / RMSE / BLEU / episode return otherwise) against a
  baseline computed the same way on held-out data; drops only alert when larger than 2× the bootstrap standard error.
* **Automation** (per deployment): triggers = drift, critical performance drop, schedule; retrains on labeled
  production data (out-of-time split: train on older traffic, test on the newest), the original dataset (e.g. an
  updated CSV) or any dataset. Gate on AUC/headline metric, then auto-promote, shadow first, or wait for approval.
  Scheduled re-testing of the champion on fresh data is also available.
* **Champion / challenger**: shadow (scores every request silently; agreement + live metrics compared) or canary
  (a fraction of real traffic). Promote = hot swap; every promotion is in the audit log and can be rolled back.
* **Demo**: Deployments → *Set up demo deployment* trains + deploys an XGBoost credit-risk model; the traffic
  simulator sends synthetic applications with switchable scenarios — economic downturn (gradual covariate + label
  shift), new customer segment (sudden), broken upstream feature (data-quality incident), concept drift.
  CLI: `python -m mltrain.mlops.simulator --deployment loan-default --key <key> --scenario downturn`.
* **SDK**: `backend/mltrain/sdk/mltrain_client.py` (single file, stdlib only) — `predict`, `log`, `feedback`, batched.

## Built-ins

* **sklearn**: logistic/ridge regression, random forest, gradient boosting (per-stage curves), SVM, kNN, MLP, *any estimator by import path*
* **XGBoost / LightGBM**: live per-round train/val curves, early stopping, GPU option (XGBoost), continue boosting from a saved model
* **PyTorch**: MLP, CNN, torchvision ResNet/MobileNet/EfficientNet/ConvNeXt (optional ImageNet weights). Shared loop: AdamW/Adam/SGD/RMSprop, cosine/step/plateau schedulers, early stopping on any metric, best-checkpoint restore, AMP, grad clipping
* **Stable-Baselines3 / sb3-contrib**: PPO, Recurrent PPO (LSTM), TRPO, A2C, DQN, QR-DQN, SAC, TD3, with periodic evaluation, videos, live view and early stopping on a reward threshold or when evaluation stops improving
* **Hugging Face**: encoder text classification (DistilBERT, BERT, RoBERTa…), seq2seq (T5, BART, Marian) with **BLEU, chrF, ROUGE-1/2/L** and sample predictions
* **Datasets**: one default per problem type ships with the repo:

  | Problem type | Default |
  |---|---|
  | Tabular classification | Iris |
  | Tabular regression | Diabetes |
  | Image classification / generation | MNIST |
  | Text classification | SST-2 |
  | Translation | Opus Books EN→FR |
  | Audio generation | synthetic tones |
  | Video generation | Moving MNIST |
  | Text generation | Tiny Shakespeare |

  Also shipped: loaders for your own data (uploaded CSVs, a CSV on disk, image / audio / text folders, any Hugging Face text or seq2seq dataset) and the MLOps demo/feedback datasets.

  More datasets live in `user_plugins/datasets/optional/`, one file each and **git-ignored**, so they stay on your machine but aren't committed: Wine, Breast cancer, Digits, California housing, synthetic generators, Fashion-MNIST, CIFAR-10, AG News, IMDB, spoken digits, noisy sine. Add your own the same way, e.g. by copying one of those files.
* **Removing datasets**: **Remove** on a dataset (Library → Datasets, or Data → Static datasets) first shows how many sessions used it and how much downloaded data it would free.
  - Uploaded CSV → deleted.
  - Optional or user plugin file → moved to `user_plugins/.trash/`.
  - Built-in or linked dataset → hidden.

  Its downloads in `data/cache` can be deleted at the same time. Everything except uploads can be brought back under **Removed datasets → Restore**. Past sessions keep their results either way.
* **Environments**: one default per RL problem type ships with the repo: **CartPole** (discrete actions), **Pendulum** (continuous actions) and **Atari** (pixels; any ALE game). The "any registered gymnasium id" loader is also included. Acrobot, MountainCar, LunarLander (+continuous), BipedalWalker, CarRacing and the example envs live in `user_plugins/envs/optional/` (git-ignored, still available locally). Environments can be removed and restored like datasets (Library → Environments).

### Basilisk spacecraft scenarios

`user_plugins/envs/basilisk.py` adds the five scenarios of the Basilisk sim-server
(`/mnt/Linux_Data_2/Projects/Basilisk_env`, run `docker compose up -d` there) as environments
**Basilisk · Single satellite ops / Constellation scheduling / Storage & downlink / RF-cued imaging /
Constellation competition**:

- Constellation scenarios: the per-satellite observations are concatenated into one vector, and the actions
  become MultiDiscrete (one choice per satellite). Train them with PPO, A2C, TRPO or Recurrent PPO; DQN works
  for single-satellite ops only.
- Constellation competition (multi-agent): the policy controls one provider (`control`, or `all`) and is
  rewarded with that provider's reward. The other provider's satellites act randomly or idle (`opponents`).
- Scenario knobs (satellites, planes, storage buffer, catch window…), `anomalies` (fault injection) and any
  other `overrides` are form fields.
- Live view and episode videos draw a mission-control panel from the server's telemetry (battery,
  reaction-wheel momentum, storage, eclipse, faults, last action, reward), one frame per decision.
- The client is imported from `Basilisk_env/client` (or `pip install -e` it); set `BASILISK_URL` /
  `BASILISK_CLIENT` to override.

Metrics: loss, accuracy, macro precision / recall / F1, confusion matrices, MSE/RMSE/MAE/R², episode
reward/length, evaluation reward, BLEU/chrF/ROUGE. Every metric is aggregated across seeds (mean ± std).

## Generative models (images, audio, video, text)

Pick any image dataset (MNIST, CIFAR-10, an image folder…) or a generative dataset, then a generative model:

| Method | Built-in network | Your own network plugs in when it… | Losses / knobs |
|---|---|---|---|
| **Diffusion** (DDPM training; DDPM or DDIM sampling) | time-conditioned UNet, 1-D/2-D/3-D | has `forward(x, t)` | linear/cosine schedule, ε / x₀ / v prediction, any loss, class conditioning |
| **VAE** | conv encoder + upsample/conv decoder | has `encode(x) -> (mu, logvar)` and `decode(z)` | reconstruction loss, β, KL warm-up |
| **GAN** | DCGAN generator + spectral-norm discriminator | file has a `*Generator` and a `*Discriminator`/`*Critic` class | non-saturating, hinge, WGAN-GP, R1, n_critic |
| **Language model** | GPT (char-level or GPT-2 BPE), or fine-tune `hf:<model>` | takes `vocab_size` and has `forward(idx) -> logits` | cross-entropy or custom, temperature/top-k sampling |
| **Custom** | — | implements `losses(x, y)` and `sample(n)` (see `examples/plugins/models/custom_generative.py`) | anything |

**Datasets.**
- Images: reuse the vision datasets, or point "Image folder (any layout)" at a folder.
- Audio: synthetic tones, the Free Spoken Digit Dataset, or a folder of wavs.
- Video: Moving MNIST.
- Text: Tiny Shakespeare, local text files, or any Hugging Face text dataset.

**The training loop handles the rest.**
- Gradient accumulation, AMP, EMA weights (with warm-up), warm-up + cosine learning-rate schedules.
- Best + final checkpoints, resume, and multiple seeds.
- Snapshots use fixed noise, so the same samples improve over training: image grids, a denoising video, VAE reconstructions, playable audio with spectrograms, mp4 grids for video, and text samples.
- End-of-training metrics: **FID / KID** (Inception-v3, downloaded once) for images, perplexity for text.

**Generate tab.** Sample new images, audio, video or text from a trained session with any seed (and DDIM steps / prompt / temperature).

**Custom loss functions.** Every public `fn(pred, target)` in `user_plugins/losses/*.py` appears in the loss fields (see `examples/plugins/losses/example_losses.py`).

**Linked project folders** (Library → Custom models → Link a folder). Use models from an existing project without copying or changing it.
- Files are inspected without running them; the chosen ones are imported in place.
- The folder is put on the import path, so the project's own imports keep working, and no `__pycache__` is written into it.
- Building-block classes are skipped.

**Architecture reference** (Library → Custom models, or **Architecture** on any PyTorch model). Shows:
- form fields with types, defaults and help
- a layer tree with output shapes and parameter counts, built on the meta device so even huge models are instant
- the class source

**Parameter auto-fill** for your `nn.Module` constructors reads:
- defaults and type hints (`Literal[...]` → dropdown)
- docstring `Args:` sections and trailing `# comments`
- config dataclasses (expanded into one field each)

Arguments the dataset determines (`in_channels`/`in_ch`, `out_ch`, `data_shape`, `num_classes`, `vocab_size`, `block_size`…) are filled automatically.

### Interchangeable components

Models built from several networks expose *slots*. Each slot is a dropdown in the model form; only the fields for the
chosen option are shown:

| Model | Slots (interface) |
|---|---|
| GAN | generator (z → x), discriminator (x → score) |
| VAE | encoder (x → features), decoder (z → x) |
| Diffusion | denoiser (x, t → x) |
| **Latent diffusion** | autoencoder (encode/decode, always frozen), denoiser (on the latent) |
| **Classifier · interchangeable encoder** | encoder (x → features) + a head — transfer learning / linear probes |

Every slot can be filled from:

1. **Built-in**: the method's configurable network.
2. **Your networks**: any class whose interface fits, uploaded or from a linked folder. This includes classes named `*Encoder`, `*Decoder`, `*Generator` or `*Discriminator`. A `<slot>_weights` field initializes the class from a `.pt`/`.pth` state dict, e.g. an old checkpoint.
3. **Trained in this dashboard**: a part of any saved generative model, taken from its best seed. Examples: a VAE's decoder as a GAN generator, a VAE's encoder as a classifier backbone, a VAE as the latent-diffusion autoencoder, a denoiser to keep training.
4. **Pretrained (downloaded once)**:
   - TAESD and the Stable Diffusion VAE as autoencoders
   - Google's DDPM CIFAR-10 UNet as a denoiser
   - ResNet-18/50 and EfficientNet-B0 (ImageNet) as encoders or discriminator backbones

Any non-built-in part can be frozen (`<slot>_freeze`). Frozen parts:
- stay in eval mode, so batch-norm and dropout don't change
- get no optimizer state
- are shared with the EMA copy instead of duplicated
- are not re-saved if they were downloaded

Sizes adapt automatically. A linear head maps any encoder's features to what the slot needs (2×latent, 1 score, or N classes). A reused decoder or generator sets `latent_dim`/`z_dim`. Latent diffusion measures the latent shape and estimates the latent scale (or uses the Stable Diffusion constant). Shape mismatches give a readable error.

The session page has a **Components** card listing each part's source (linked) and which other models reuse this model's parts.

**Per-component training settings.**
- Every trainable slot has `<slot>_lr` and `<slot>_weight_decay` fields; GAN slots also have `<slot>_beta1`. An empty field uses the main training value.
- Each slot is its own optimizer parameter group, and when the learning rates differ, each one is charted (`lr_generator`, `lr_discriminator`, …).
- Typical uses:
  - two-timescale GAN training, e.g. D at 4e-4 and G at 1e-4 with beta1 = 0
  - fine-tuning a pretrained encoder at 1e-5 while its new head trains at 1e-3
  - a slower decoder in a VAE
- Your own generative classes can declare `OPTIMIZER_GROUPS = ("g", "d")` to get a learning-rate field per group.

**GAN losses per network.**
- `adversarial` presets: non-saturating, hinge, WGAN-GP, least-squares.
- `custom` sets `discriminator_loss` and `generator_loss` separately: `bce`, `hinge`, `wasserstein`, `least_squares`, or any of your loss functions, called as `fn(scores, targets)` with real = 1 and fake = 0.
- The gradient penalty applies to a Wasserstein discriminator; R1 works with every loss.

## Adding your own models, datasets and environments

Upload `.py` files on **Library → Add / upload**, or drop them into `user_plugins/{models,datasets,envs}/`
and press **Reload plugins**. Templates live in `examples/plugins/`:

| File | Pattern |
|---|---|
| `models/residual_mlp.py` | Plain `nn.Module`: constructor args become form fields, trained by the built-in loop |
| `models/custom_training_loop.py` | `ModelPlugin` with its own `train()` (use for XGBoost, GANs, JAX, …) |
| `models/ppo_custom_net.py` | Custom SB3 PPO (residual feature extractor) — subclass `SB3Model`, keep videos/eval/fine-tune |
| `envs/grid_world.py` | Plain `gymnasium.Env`: works with PPO/A2C/DQN |
| `envs/cartpole_variants.py` | `EnvPlugin` wrapping a Gym env with noise / physics knobs (great for robustness evaluations) |
| `datasets/noisy_sine.py` | `DatasetPlugin` returning a `DataBundle` |

Inside `train(problem, ctx)`: `ctx.params`, `ctx.seed`, `ctx.device`, `ctx.log(step, {"val/acc": …})`,
`ctx.log_artifact(name, kind, data)`, `ctx.log_media(step, name, kind, data|file)` (snapshots),
`ctx.live_frame(rgb)`, `ctx.set_progress(frac, text)`, `ctx.should_stop()`, `ctx.save_model` /
`ctx.model_path(name)` (where to save), `ctx.init_path` (saved model to start from). Implement `evaluate()` and
set `supports_eval` / `supports_init` / `supports_resume` to enable those modes.

⚠️ Plugins are arbitrary Python executed with your user's permissions. Keep the dashboard on your LAN
(don't port-forward it) and only upload code you trust.

## Portfolio exports (GitHub)

Select sessions on **Sessions** (or **Export** on a session) → *Export to GitHub…* builds `data/exports/<name>/`, a standalone git
repository that needs no dashboard:

- `README.md` — results table (mean ± std over seeds), comparison plot, per-experiment metrics, best-vs-final table, learning
  curves, confusion matrices, RL episode videos, hyperparameters, library versions
- `experiments/*.json` (configs) · `results/` (original metrics per seed) · `mltrain/` (training code only — no server/DB)
- `train.py` re-runs everything with the original seeds and prints *this run vs. original*; `predict.py` runs a saved model
- data that can't be downloaded (live sources, uploads, feedback) is frozen into `data/snapshots/` with the exact train/val/test
  split and recency weights; optional weights in `models/` (files over GitHub's 100 MB limit are skipped)
- secrets (dashboard password/tokens, database passwords) and your home path are scrubbed

The **Exports** page previews the README, downloads a zip, and publishes: push to an empty GitHub repo you created (SSH or HTTPS URL),
or create one with the GitHub CLI if `gh` is installed and logged in. Nothing is pushed without that explicit, confirmed action.

## JSON API, LLM tools and MCP

Everything the UI shows comes from a JSON API, with interactive documentation at `/docs`. To authenticate
from scripts, send `Authorization: Bearer <token>` (the token is shown in Settings).

| Endpoint | What |
|---|---|
| `GET /api/llm/overview` | Compact digest of recent sessions: configs, headline metric, final metric means |
| `GET /api/sessions/{id}/export` | Everything about one session: config, per-seed and aggregated curves, final stats, artifacts, lineage (`schema: mltrain.session/v1`) |
| `GET /api/sessions` / `GET /api/sessions/{id}` | List / detail |
| `POST /api/sessions` | Start training (same body the UI sends; set `parent_id` to link a retrain) |
| `GET /api/sessions/{id}/lineage` | Family tree and parameter changes |
| `GET /api/plugins` | All models / datasets / envs and their parameter schemas |
| `GET /api/models/saved` | Saved models usable for fine-tuning / evaluation |
| `POST /api/sweeps` | Grid search: `{"base": <session body>, "grid": {"model.lr": [1e-3, 3e-3]}}` |
| `POST /api/groups` | Several sessions at once (benchmarks, multi-model comparisons) |
| `GET /api/groups/{id}` | Ranked group results |
| `GET /api/llm/tools` | Tool definitions (name / description / input_schema) mapping onto these endpoints |
| `GET /api/deployments` · `/api/deployments/{id}/analysis` · `/api/alerts` | MLOps health, drift, live performance, alerts |
| `POST /api/serve/{id}/predict` · `/feedback` · `/log` | Serving, ground truth, external log push (`X-API-Key`) |

**MCP server** (Claude Desktop / Claude Code / any MCP client), talks to the dashboard over HTTP:

```bash
claude mcp add mltrain -e MLTRAIN_URL=http://<host>:8000 -e MLTRAIN_TOKEN=<token> -- \
  env PYTHONPATH=/mnt/Linux_Data_2/Projects/DL_dashboard/backend ~/anaconda3/envs/mltrain/bin/python -m mltrain.mcp_server
```

## Layout

```
backend/mltrain/
  api/app.py          app setup, LAN guard, auth, settings, plugins, serves frontend/dist
  api/sessions.py     sessions (train / fine-tune / evaluate), metrics, snapshots, WebSocket
  api/experiments.py  saved models, sweeps, groups, storage
  api/llm.py          LLM digest + tool definitions;  mcp_server.py  MCP server
  api/mlops.py        registry, deployments, serving, monitoring, automation, demo endpoints
  mlops/              registry, predictors, profiles, monitoring, serving, automation, export, demo + simulator
  preprocessing.py    fitted, serializable tabular preprocessing (the serving contract)
  data/               connectors, profiling, static reports, live sources, schedules, demo stream
  api/data.py         data sources, quality reports, schedules endpoints
  sdk/                mltrain_client.py — client SDK for apps and external deployments
  scheduler.py        GPU/CPU-limited job queue, launches worker processes
  worker.py           runs one seed: loads plugins, calls model.train()
  analysis.py         seed aggregation, lineage, param diffs
  plugins/            Param / plugin base classes, registry (+ nn.Module / gym.Env auto-wrapping)
  builtin/            built-in datasets, envs, sklearn / torch / sb3 / hf models
  training/           RunContext (logging, snapshots, live frames), options, snapshot builders, metrics
frontend/src/         React + Vite + Plotly UI
user_plugins/         your uploaded plugins
data/                 SQLite DB, run logs/checkpoints, uploads, dataset cache (gitignored)
```

Development: run the backend (`./run.sh`), then `cd frontend && npm run dev` for hot reload on :5173.
