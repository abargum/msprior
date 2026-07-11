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
6. **`msprior/utils.py` + `msprior_scripts/preprocess.py`** — added a
   `--overlap` flag (0–1, default 0) to `msprior preprocess`. Previously
   `load_audio_chunk` read strictly sequential, non-overlapping chunks from
   ffmpeg's raw PCM stream; a nonzero `--overlap` now uses a smaller hop via
   a sliding buffer (`hop_signal = n_signal * (1 - overlap)`), extracting
   more distinct (if correlated) training crops from the same source audio.
   `--overlap 0.5` roughly doubled the dataset (309 → 610 chunks on
   `rave-rmc/audio/`) — see the v4/v5 experiment below.

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

Best export to date (superseded by v4/v5 below):
`runs/prior_rmc_full_stereo_v2_smallmodel_earlystop/prior_rmc_full_stereo_v2_bestckpt.ts`
(from v2's `best.ckpt`, via `--ckpt best`).

### RAVE reconstruction sanity check (2026-07-11)

Before investing further in the prior, checked whether `v3_full_stereo`
actually represents this audio well (round-tripped a 20s clip of
`01 Azure _etmstr1.wav` through `model.encode`/`model.decode` and compared
log-mel spectrograms). Reconstruction closely matches the original — same
rhythmic structure, harmonic bands, and noise floor, with only the usual
RAVE loss of the very highest-frequency detail. **Conclusion: RAVE is not
the bottleneck** — confirms the issue is squarely in the prior/data, not
the autoencoder.

## Follow-up runs: overlapping chunks + fewer quantizers (2026-07-11)

Two more levers, motivated by the data-quantity conclusion above:

- **More training signal from the same audio**: added `--overlap` to
  `preprocess` (bug fix #6). Re-preprocessed into
  `data/rmc_preprocessed_overlap50` with `--overlap 0.5`: 309 → 610 chunks.
- **Simplify the task to match the data budget**: RAVE's 16 latent
  channels are fidelity/PCA-ordered, so most information is in the first
  few. Passing `--override "NUM_QUANTIZERS=8"` at train time crops the
  prior to only model/predict the first 8 — confirmed empirically that
  this override actually takes effect (gin resolves `%NUM_QUANTIZERS`
  lazily, so `train.py`'s dataset-driven auto-detection line is idempotent
  with an explicit override, not a silent clobber as initially suspected).

Results on `rmc_preprocessed_overlap50` (early stopping on throughout):

| Run | Best val_cross_entropy | Best val_acc_top_1 | Best val_acc_top_10 |
|---|---|---|---|
| v1 (orig data, no ES) | 3.867 | 3.13% | 29.31% |
| v2 (orig data, ES) | 3.870 | 3.10% | 29.34% |
| v3 (orig data, default capacity) | 3.968 | 2.90% | 27.07% |
| **v4** (`prior_rmc_full_stereo_v4_overlap_smallmodel`) | **3.805** | **3.36%** | **31.75%** |
| **v5** (`prior_rmc_full_stereo_v5_overlap_fewerquant`, `NUM_QUANTIZERS=8`) | **3.710** | **3.79%** | **36.27%** |

- **v4** (overlap data alone, otherwise identical to v2) already beats
  every prior run, and training ran longer before stopping (step 3,569 vs
  2,879) — direct evidence there was more genuine signal to learn from.
- **v5** (overlap + 8 quantizers) is the best result so far by a clear
  margin, stopping at step 4,861.
- **Caveat on v5**: `NUM_QUANTIZERS=8` means the model only ever predicts
  the first 8 of RAVE's 16 latent channels. The better numbers reflect a
  genuinely easier, better-fit statistical task — not necessarily
  "identical audio, less noise." The remaining 8 (finer detail/texture)
  are simply never generated by this model. Whether that fidelity
  tradeoff is worth the improved coherence can only be judged by
  listening, ideally comparing v5 against v2/v4.

Exports:
- `runs/prior_rmc_full_stereo_v4_overlap_smallmodel/prior_rmc_full_stereo_v4_overlap_smallmodel.ts`
- `runs/prior_rmc_full_stereo_v5_overlap_fewerquant/prior_rmc_full_stereo_v5_overlap_fewerquant.ts`
  (both via `--ckpt best`)

**Important correction after listening**: v5's 8-quantizer output is *not*
a drop-in option for realtime nn~ use — the RAVE decoder needs all 16
latent channels, so an 8-quantizer prior isn't directly compatible without
padding/zero-filling the missing 8 (which audibly hurts quality). v4
sounded more coherent in practice. **Conclusion: stick to `NUM_QUANTIZERS=16`
for anything meant to drive the realtime nn~ decoder** — the
quantizer-reduction lever from the previous section is only useful if you
don't need full RAVE-decoder compatibility.

## Follow-up runs: more overlap + re-testing capacity (2026-07-11, cont'd)

Two more experiments, keeping `NUM_QUANTIZERS=16` (full compatibility)
throughout:

- **v6** (`prior_rmc_full_stereo_v6_overlap75_smallmodel`): pushed overlap
  further, `--overlap 0.75` → `data/rmc_preprocessed_overlap75`, 309 → 1,206
  chunks (~4x). Same small-model hyperparameters as v4/v2, `val_size=120`
  (scaled up with the larger dataset).
- **v7** (`prior_rmc_full_stereo_v7_overlap50_defaultcapacity`): re-tested
  the bigger default capacity (`MODEL_DIM=512`, `NUM_LAYERS=8`,
  `DROPOUT_RATE=0.01`) that lost to the small model in v3 — this time on
  the larger `overlap50` (610-chunk) dataset, to check whether more data
  changes that conclusion.

| Run | Best val_cross_entropy | Best val_acc_top_1 | Best val_acc_top_10 | Stopped @ |
|---|---|---|---|---|
| v2 (orig data, small model) | 3.870 | 3.10% | 29.34% | 2,879 |
| v4 (overlap50, small model) | 3.805 | 3.36% | 31.75% | 3,569 |
| **v6 (overlap75, small model)** | **3.753** | **3.56%** | **34.02%** | **7,687** |
| v7 (overlap50, default capacity) | 3.923 | 2.97% | 27.95% | 1,665 |

- **v6 is the new best 16-quantizer candidate**, beating v4 across every
  metric. It also trained more than 2x longer before early stopping
  (7,687 vs 3,569 steps) — more overlap kept paying off, not just
  saturating.
- **v7 confirms v3's conclusion holds even with 2x the data**: bigger/
  less-regularized capacity still overfits faster (train loss down to
  2.79) and lands worse on validation than every small-model run. Model
  capacity is not the lever here, at least not up to ~600-1200 chunks.
- Exported: `runs/prior_rmc_full_stereo_v6_overlap75_smallmodel/prior_rmc_full_stereo_v6_overlap75_smallmodel.ts`
  (via `--ckpt best`) — **current best real-world candidate.**

## Status / next steps

- [ ] Listen to v6 — current best candidate, full 16-quantizer output,
      compatible with the realtime nn~ decoder.
- [ ] Overlap has now been pushed 0% → 50% → 75% with consistent gains at
      each step and no sign of saturating yet (v6 still trained longer
      before stopping than v4). Worth trying even higher overlap
      (e.g. 0.85-0.9) as a cheap next step before reaching for more source
      audio.
- [ ] Biggest remaining lever is still **more source audio** — overlap
      squeezes more crops out of a fixed ~3 hours, but is fundamentally
      reusing the same underlying information.
- [ ] Not yet tried: pretraining the prior on a larger, stylistically
      compatible corpus (processed through this same RAVE model), then
      fine-tuning (`--ckpt <path>`) on the RMC-specific data — standard
      fix for "good encoder, too little data for the downstream model,"
      likely higher-impact than further hyperparameter search alone.
- [ ] Could also try a shorter `SEQ_LEN` (more distinct crops per chunk
      per epoch) — not yet tested in combination with any overlap dataset.
- [ ] Model capacity conclusion now confirmed twice (v3 and v7): don't
      reach for `MODEL_DIM=512`/`NUM_LAYERS=8` defaults on this data scale
      — they consistently overfit faster and land worse than the smaller
      `MODEL_DIM=128`/`NUM_LAYERS=4` config.
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
