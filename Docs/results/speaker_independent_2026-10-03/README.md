# 2026-10-03 speaker-independent comparison

- `comparison_summary.json`: full aggregate, speaker metrics, matched-vowel and paired bootstrap summaries.
- `comparison/{fresh,warm_start,phoneme_multi}/{test,validation}.json`: every prediction, raw/normalized scores and artifact SHA-256.
- `comparison/initial_test.json`: historical checkpoint baseline.
- `comparison/manifest.json`, `bundle_manifest.json`: exact inputs, source hashes, settings and provenance.
- `training.log`: collected epoch history; best checkpoints for warm-start and multitask are epoch 1, followed by 5 non-improving epochs.
- `source_snapshot.tar.gz`: exact Python sources and fixed CSVs used in the completed run, before integrating Main PR #53. Extract into a separate directory when inspecting that historical execution. Recorded Python source hashes were verified against the archive.
- `comparison/*/checkpoint/*.json`: checkpoint metadata, not model tensors.

All 7 evaluation JSONs are retained. Large tensor files and incomplete transfer chunks stay in the ignored local `experiments/` directory. Fresh and warm-start tensors were recovered and verified locally; the measured phoneme-multitask tensor was not recovered and cannot be deployed. The registered `wav2vec2_phoneme_multi` strategy describes the architecture; it does not imply the measured 55.00% checkpoint is installed.

The dataset and all 7,802 audio hashes were checked again after merging Main. Main's legacy Colab/combined-split tools remain unchanged; the new dedicated runner uses fixed speaker-independent manifests and no automatic HF publication. Main also adds microphone saturation augmentation, so future runs use the merged implementation and are distinct experiments from this archived result.
