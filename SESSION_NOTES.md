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

### Checkpoint export gotcha

`ScriptedPrior` (`msprior/scripted.py`) auto-selects a checkpoint via
`pathlib.Path(run).rglob("*.ckpt")`, sorted so filenames containing `"last"`
are preferred over `"best"` — **regardless of the `--ema_weights` flag**.
So a plain `msprior export --run ... --ema_weights` pulls EMA weights
anchored near the *end* of training (already overfit), not the
best-validation checkpoint. To export from `best.ckpt` instead, temporarily
move `last.ckpt` out of the checkpoints directory, run export, then restore
it (both checkpoints are gitignored, so this is safe/local-only).

Two exports currently exist locally (both gitignored, not in this repo):
- `runs/prior_rmc_full_stereo_v1/prior_rmc_full_stereo_v1_bestckpt.ts` —
  from `best.ckpt` (val_cross_entropy 3.87 @ step ~1,889). **This is the one
  to listen to/evaluate first.**
- The original last-ckpt export was overwritten when the best-ckpt one was
  produced (export always writes to the same filename); it can be
  regenerated from `last.ckpt` if needed for comparison since the
  checkpoint itself is untouched.

## Status / next steps

- [ ] Listen to `prior_rmc_full_stereo_v1_bestckpt.ts` and judge quality.
- [ ] If still weak: the likely lever is more/longer source audio rather
      than further checkpoint archaeology on this run — dataset size looks
      like the binding constraint, not just where training stopped.
- [ ] Consider re-enabling `--early_stopping` (now `true` by default again
      after the flag addition) for the next run so training doesn't run
      1000 epochs past the best point.
- [ ] If retrying with the same data, current hyperparameters
      (`MODEL_DIM=128`, `NUM_LAYERS=4`, `DROPOUT_RATE=0.15`) are already
      fairly conservative for a small dataset; a smaller model or stronger
      weight decay could be tried if overfitting persists even with early
      stopping.

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
