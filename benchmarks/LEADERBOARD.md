# Voicerecognizer Strategy Leaderboard

汎用精度および実環境ロバストネス向上のための各実験ストラテジーの横並びスコアです。

| Strategy | Category | Status | General Val Acc | Speakerphone Acc | Female Acc | CPU Latency | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `wav2vec2_baseline` | Baseline | active | 92.8% | 90.4% | 67.3% | 70.1 ms | 116.6 MB |
| `wav2vec2_ipa_kd` | Knowledge Distillation | active | 90.2% | 89.1% | 71.2% | 44.6 ms | 116.6 MB |
| `wav2vec2_whisper_kd` | Knowledge Distillation | active | 90.1% | 87.8% | 63.5% | 40.8 ms | 116.6 MB |
| `wav2vec2_phoneme_multi` | Representation & Loss | active | 90.3% | 88.5% | 76.9% | 41.3 ms | 116.6 MB |
| `wav2vec2_denoise_kd` | Knowledge Distillation | planned | - | - | - | - | - |
| `wav2vec2_arcface` | Representation & Loss | planned | - | - | - | - | - |
| `wav2vec2_onset_focused` | Representation & Loss | planned | - | - | - | - | - |
| `wav2vec2_rir_simulation` | Acoustic Augmentation | planned | - | - | - | - | - |
| `wav2vec2_pseudo_label` | Acoustic Augmentation | planned | - | - | - | - | - |
| `whisper_encoder_onnx` | Backbone & Inference | planned | - | - | - | - | - |
| `wav2vec2_tta_ensemble` | Backbone & Inference | planned | - | - | - | - | - |
