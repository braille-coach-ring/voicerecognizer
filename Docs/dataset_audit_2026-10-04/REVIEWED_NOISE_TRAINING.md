# レビュー結果と新規CSVを使う学習

対象Main: `8a2296d3`（PR #58）。Val話者r2/r4、Test話者r3/r5はユーザー指定。

## データの確定

- r話者の保存済み判定998件: keep957、relabel28、other13。
- 判定JSONの末尾に余分な `}` があったため、元ファイルを `completed_review_raw_*.json` にバックアップし、完全な先頭JSONを保ったまま末尾を修復した。
- 旧不一致候補のうち116件には保存された判定がなく、今回の集合から除外した。ユーザーの確認を待っており、自動で確認済み扱いにはしていない。
- 既存rinryの異ラベル同一音声13組・26ファイルも除外。take等の以前の評価集合は今回のTrainに含めない。
- 新規CSVの69件はラベルをそのまま採用: other64、a/ya/ha/pa/u各1。otherへの一括変更はしない。
- 新規other64録音はseed42で並べ替え、Val10、Test10、Train44。非otherの5件はTrain。
- 環境音のIDは録音単位で、人間の話者IDではない。同じ録音セッション内のファイル保留評価であり、未知の環境への汎化評価とは主張しない。
- 人間の修正は学習CSVのlabel列に反映する。元のWAVパスとファイル本体は変更しない。
- 既存Trainの生声話者を継続使用。TTS、以前のVal/Test話者、人物不明の古いPC録音は保留。

| 分割 | 件数 | other | ラベル数 |
| --- | ---: | ---: | ---: |
| Train | 7,262 | 56 | 105 |
| Val | 406 | 11 | 105 |
| Test | 396 | 10 | 105 |

全体8,064件。人物/録音ID、パス、SHA-256の重複・集合間交差がないことを検査した。

## 学習条件

既存ローカル `weights/wav2vec2_best` からのWav2Vec2継続学習1条件。最大15epoch、batch8、学習率3e-5、Transformer先頭4層固定、patience5、seed42。固定Valの同音統合後Macro-F1でモデルを選ぶ。Test由来の混同サンプラー、共有モデルへの自動アップロード、実験中のONNX生成を停止する。

保存されている初期モデルの過去学習データは未確認。今回の比較は固定集合での継続学習の効果を測り、初期モデルの来歴まで保証するものではない。

生成スクリプト: `script/prepare_reviewed_training.py`。固定分割: `data_splits/reviewed_noise_20261004/{train,val,test}.csv`。準備記録: 同ディレクトリの `preparation_manifest.json` と `review_decisions_snapshot.json`。

関連テスト8件成功。ラベル修正、other変更、新規CSVの非otherラベル保持、未保存判定の除外、元WAVの保持を検証した。

## 実行状態

ローカル前処理が完了。TrainとValの7,668件を前処理し、Testの音声が含まれないことを確認した。実行先: `experiments/reviewed-noise-20261004-local/`。

Colab用のコマンドは以下。外部転送は初回の自動承認レビューで拒否されたが、その後ユーザーが音声8,064件・CSV・コード・既存モデルのGoogle Colab T4への転送と学習を明示的に許可した。許可後に以下の実行を開始した。

```powershell
uv run --no-sync python -u script/run_speaker_independent_colab.py --initial-model weights/wav2vec2_best --split-dir data_splits/reviewed_noise_20261004 --runs warm_start --epochs 15 --session vr-reviewed-noise-20261004 --download-models
```

`script/run_speaker_independent_colab.py` に対象split指定と実行条件選択を追加した。既定の以前の3条件比較は維持する。転送許可後はColab T4で実行し、評価JSONとモデル本体を専用実験フォルダへ回収する。


## 完了結果

Google Colab T4で15エポックを完走し、ValのMacro-F1が最高だった15エポック目のチェックポイントを選択した。最良モデル本体、設定、全予測、評価JSON、実行ログをローカルへ回収した。Colabセッションは停止済み。

### 同じTest 396件での比較

| 指標 | 既存モデル | 継続学習後 | 差 |
| --- | ---: | ---: | ---: |
| 同音統合後Accuracy | 57.07%（226/396） | 71.97%（285/396） | +14.90ポイント |
| 同音統合後Macro-F1 | 0.5332 | 0.7063 | +0.1731 |
| 厳密Accuracy | 55.81%（221/396） | 70.71%（280/396） | +14.90ポイント |
| 厳密Macro-F1 | 0.5223 | 0.6876 | +0.1652 |
| r3同音統合後Accuracy（192件） | 66.67% | 81.25% | +14.58ポイント |
| r5同音統合後Accuracy（194件） | 50.00% | 62.89% | +12.89ポイント |
| 保留環境音other判定（10件） | 1/10 | 7/10 | +6件 |

Val 406件の同音統合後Accuracyは76.85%（312/406）、Macro-F1は0.7609。厳密Accuracyは75.37%（306/406）。同音統合はdi→ji、du→zu、wo→oで、保存予測そのものは変更していない。

環境音評価は同じ収録セッションの10録音に限られる。Test話者はr3/r5だが、初期モデルの過去学習データは未確認で、今回のTestはラベル確認に使った固定集合である。新しい盲検評価での性能保証とは区別する。

### 保存先

- 最良モデル: `experiments/vr-reviewed-noise-20261004/results/experiments/comparison/warm_start/checkpoint/`
- モデルSHA-256: `72931e7bed7ea701eec80bd98467d65d06fd8369819dd5da517179c08546d81e`
- 既存Test予測: `experiments/vr-reviewed-noise-20261004/results/experiments/comparison/initial_test.json`
- 学習後Test予測: `experiments/vr-reviewed-noise-20261004/results/experiments/comparison/warm_start/test.json`
- 学習後Val予測: `experiments/vr-reviewed-noise-20261004/results/experiments/comparison/warm_start/validation.json`
- 実行記録: `experiments/vr-reviewed-noise-20261004/results/experiments/comparison/manifest.json`
- 全実行ログ: `experiments/vr-reviewed-noise-20261004/results/experiments/comparison.log`
- 集約結果: `Docs/dataset_audit_2026-10-04/training_result_summary.json`

回収したモデル本体378,408,212 bytesのSHA-256と、Val/Test/manifestのモデル識別値の一致を確認した。CSVハッシュと各評価レコードの一致、保存予測から全評価指標の再計算一致、転送時と実行時のデータ分割記録の一致を確認した。初期モデル本体も転送前のハッシュと一致しており変更されていない。

今回の成果物はPyTorchのsafetensorsチェックポイント。共有の既定モデルへの置換、Hugging Faceへの公開、ONNXへの変換は行っていない。


## 女性・旧ベンチマークの追加確認

`REGRESSION_BENCHMARK.md`に追加評価を保存した。女性生声takeは41.15%→60.96%、旧固定Test全体は50.00%→65.16%に改善した。一方、既存男性reon・yu-otaは合計96%→87%に低下。従来General Valとスピーカーフォン評価には今回Trainとの音声重複があるため独立評価として扱わない。
