# Session notes — RMC stereo prior (2026-07-11)

Working notes from a Claude Code session, kept here so work can be picked up
later without re-deriving context. Covers environment fixes, the fork-specific
bug fixes, the first training run, and where quality currently stands.

## Goal

Train an `msprior` prior model on top of the stereo RAVE model in
`rave-rmc/runs/v3_full_stereo/runs.ts`, using audio in `rave-rmc/audio/`.

## Environment fixes (not code changes, just local env state)

- The `msprior` conda env had a stale `torchaudio==2.11.0` mismatched against
  `torch==2.5.0+cu124`, whose compiled extension needed `libcudart.so.13`
  (not installed — env only has CUDA 12.4 runtime libs). Fixed by installing
  `torchaudio==2.5.0+cu124` to match torch.
- `~/.local/bin` is ahead of the conda env's `bin` in `$PATH`, so bare
  `msprior` / `pip` can silently resolve to the wrong Python install. When in
  doubt, call the conda env's binaries by full path, e.g.
  `/home/ucloud/miniconda3/envs/msprior/bin/msprior`.
- The `msprior` conda env originally had `acids-msprior==1.1.3` installed from
  PyPI (predates stereo support). Reinstalled as editable from this repo:
  `MSPRIOR_VERSION=1.1.3+local python -m pip install -e . --no-deps` (run
  with the conda env's own `python -m pip`, not the ambient `pip`).

## Bug fixes made in this fork

1. **`msprior_scripts/preprocess.py`** — `RAVEEncoder` never adapted mono
   input audio to the channel count a stereo-trained RAVE model expects.
   Added `self.n_channels = self.model.encode_params[0].item()` in
   `__post_init__` and `audio_batch = audio_batch.expand(-1, self.n_channels, -1)`
   in `__call__`, so mono chunks get tiled to match `n_channels` before
   `model.encode()`. Without this, encoding a stereo RAVE model errors with a
   channel-mismatch (`expected 32, got 16` for a 2-channel/16-band PQMF
   model) because raw mono was fed straight into `encode()`.
2. **`msprior/scripted.py`** — EMA weight loading had a stray trailing comma:
   `ckpt = ckpt["callbacks"]["EMA"],` wrapped the state dict in a 1-tuple,
   breaking `load_state_dict`. Fixed to `ckpt = ckpt["callbacks"]["EMA"]`.
3. **`msprior_scripts/train.py`** — added a `--early_stopping` bool flag
   (default `True`) so `EarlyStopping` on `val_cross_entropy` can be
   disabled explicitly, instead of always being wired in.
4. Added a thin **`train.py`** wrapper at repo root (not part of upstream)
   so `python train.py --config ...` works from the repo root without
   installing the package first.
5. **`msprior/scripted.py` + `msprior_scripts/export.py`** — added a
   `--ckpt` flag to `msprior export` (default `"best"`, also accepts
   `"last"` or an explicit path). Replaces the old implicit selection
   (`rglob("*.ckpt")` sorted to prefer any filename containing `"last"`),
   which silently exported end-of-training/overfit weights by default and
   broke outright once Lightning version-bumped a colliding filename to
   `last-v1.ckpt` (selection is now by filename *prefix*, resolved to the
   most recently written match).

## Preprocessing

```
/home/ucloud/miniconda3/envs/msprior/bin/msprior preprocess \
  --audio /home/ucloud/projects/rave-rmc/audio \
  --out_path /home/ucloud/projects/msprior/data/rmc_preprocessed \
  --rave /home/ucloud/projects/rave-rmc/runs/v3_full_stereo/runs.ts
```

Used defaults: `--num_secs 16` (rounds up to 2^20 samples = 23.78s chunks =
512 latent frames at this model's 2048 temporal ratio), `--resolution 64`
(token vocab size — must match `NUM_TOKENS` at train time).

**Gotcha**: `preprocess.py` rounds chunk length up to the next power-of-two
*sample* count. `--num_secs` values from ~6s to ~11.9s all collapse to the
same 2^19 = 524288 samples = exactly 256 latent frames, which exactly
matches `SEQ_LEN=256` used below and breaks `decoder_only_rave`'s random
crop (`randint(0, shape[0]-seq_len-1)` needs `shape[0] > seq_len`). The
current 16s setting (512 frames) is effectively the smallest safe chunk
size for `SEQ_LEN=256` — there's no reachable intermediate size.

27 source files in `rave-rmc/audio/`, ~3 hours total → roughly 400-500
chunks after preprocessing at 23.78s each.

## Training

Run name: `prior_rmc_full_stereo_v1`, config saved at
`runs/prior_rmc_full_stereo_v1/config.gin`.

```
cd /home/ucloud/projects/msprior
python train.py \
  --config recurrent \
  --db_path /home/ucloud/projects/msprior/data/rmc_preprocessed \
  --name prior_rmc_full_stereo_v1 \
  --pretrained_embedding /home/ucloud/projects/rave-rmc/runs/v3_full_stereo/runs.ts \
  --override "NUM_TOKENS=64" \
  --override "MODEL_DIM=128" \
  --override "NUM_LAYERS=4" \
  --override "DROPOUT_RATE=0.15" \
  --override "SEQ_LEN=256" \
  --override "utils.build_warmed_exponential_lr_scheduler.peak_iteration=1500" \
  --override "torch.optim.AdamW.weight_decay=0.005" \
  --ema 0.999 \
  --batch_size 16 \
  --val_size 60 \
  --val_every 50 \
  --early_stopping=false
```

Ran the full 1000 epochs / 15,000 steps (early stopping deliberately off).

### TensorBoard diagnosis (2026-07-11)

Pulled scalars from
`runs/prior_rmc_full_stereo_v1/version_0/events.out.tfevents.*`:

- Train `cross_entropy`: 4.20 → 3.22, steadily decreasing (model fits
  training data fine).
- `val_cross_entropy`: bottomed out at **3.87 around step 1,889** (~epoch
  125), then rose back to **4.14 by step 14,999** — i.e. it got *worse*
  than early training for the remaining ~90% of the run. For reference,
  uniform-random over `NUM_TOKENS=64` gives cross-entropy = ln(64) ≈ 4.16,
  so end-of-training validation performance is barely above chance.
- `val_acc_top_1` peaked at only ~3.1%, `val_acc_top_10` at ~29%.
- Conclusion: **clear overfitting** past step ~1,900, and even at its best
  point the model only modestly beat the random baseline — likely because
  ~400-500 chunks (with `val_size=60` held out) is thin data for a
  4-layer/128-dim model.

### Checkpoint export (fixed)

Export now takes an explicit `--ckpt best|last|<path>` flag (default
`best`) — see bug fix #5 above. No more manual move-aside-and-restore
needed; e.g.:

```
msprior export --run runs/<name> --temporal_ratio 2048 --continuous \
  --ema_weights --ckpt best
```

## Follow-up runs: early stopping + capacity comparison (2026-07-11)

Compared against [devstermarts/Notebooks](https://github.com/devstermarts/Notebooks)
(MSPrior training templates) and its `devstermarts/msprior` fork. Their
recipe always keeps `EarlyStopping(patience=20)` on `val_cross_entropy`
active (they only raise the epoch *ceiling*, via a `--epochs` flag, not
remove the stopping criterion) and trains with the `recurrent.gin`
*default* capacity (`MODEL_DIM=512`, `NUM_LAYERS=8`, `DROPOUT_RATE=0.01`)
rather than a scaled-down model. Ran two more trainings on the same
`rmc_preprocessed` dataset to test both changes in isolation:

- **v2** (`prior_rmc_full_stereo_v2_smallmodel_earlystop`): same
  hyperparameters as v1, early stopping back on. Best val_cross_entropy
  3.870 @ step 1,979 — matches v1's best (3.867) almost exactly, but
  training correctly stopped at step 2,879 instead of running to 14,984.
  **This validates the early-stopping fix**: same quality ceiling, no
  overfit collapse, no risk of exporting from the wrong end of training.
- **v3** (`prior_rmc_full_stereo_v3_defaultcapacity_earlystop`): dropped
  the `MODEL_DIM`/`NUM_LAYERS`/`DROPOUT_RATE` overrides to use
  `recurrent.gin` defaults, early stopping on. Best val_cross_entropy
  **3.968 @ step 854** — worse than v1/v2, and reached (then overfit) much
  faster: train `cross_entropy` collapsed to 1.6 while val loss never beat
  3.97. **The bigger/less-regularized model did not help here** — with
  only ~400 training chunks, more capacity mainly means faster
  memorization, not better generalization.
- All three runs plateau around val_cross_entropy 3.87–3.97, only
  modestly below the random baseline (ln(64) ≈ 4.16). This ceiling looks
  like a **data-quantity limit**, not a checkpoint-timing or architecture
  problem.

Best export to date:
`runs/prior_rmc_full_stereo_v2_smallmodel_earlystop/prior_rmc_full_stereo_v2_bestckpt.ts`
(from v2's `best.ckpt`, via `--ckpt best`).

## Status / next steps

- [ ] Listen to `prior_rmc_full_stereo_v2_bestckpt.ts` and judge quality —
      this is the current best candidate (early stopping fixed, matches
      the best quality seen across all three runs).
- [ ] If still weak: the real lever is more/longer source audio, not
      further hyperparameter search on this same ~400-chunk dataset —
      neither early stopping nor model capacity moved the quality ceiling
      in these experiments.
- [ ] `--early_stopping` now defaults to `true`; no need to pass it
      explicitly unless deliberately disabling it for a diagnostic run.

## Repro/environment cheatsheet

- Conda env: `msprior` (python 3.9). Activate with
  `source /home/ucloud/miniconda3/etc/profile.d/conda.sh && conda activate msprior`.
- Always double check `which python` / use full binary paths — `~/.local/bin`
  shadows the conda env in `$PATH`.
- RAVE model: `/home/ucloud/projects/rave-rmc/runs/v3_full_stereo/runs.ts`
  (stereo, `encode_params = [2, 1, 16, 2048]` → n_channels=2, latent_size=16,
  temporal_ratio=2048).
- Preprocessed dataset: `/home/ucloud/projects/msprior/data/rmc_preprocessed`
  (gitignored — regenerate from `rave-rmc/audio/` if missing).
