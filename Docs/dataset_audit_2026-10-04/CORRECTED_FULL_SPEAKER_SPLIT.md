# 修正版：全データを話者分割で利用（2026-10-04）

ユーザー指定: 既存の話者単位の分割を維持し、新規r話者6名をTrain、2名をVal、2名をTestに追加。保存判定がなかった116件も現在の収集ラベルで採用する。元WAVは変更しない。

## 所属と件数

| 集合 | 既存固定集合 | r追加 | 新規CSV録音 | 合成音声追加 | 合計 | ラベル数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 6,124 | 1,552 | 49 | 312 | 8,037 | 105 |
| Val | 1,058 | 416 | 10 | 0 | 1,484 | 105 |
| Test | 620 | 416 | 10 | 0 | 1,046 | 105 |

- Train既存: haruyamike 990、rikuto 50、rikutomike 1,022、rinry 3,629、ryu 49、collected 384。
- Train新規r: r1 208、r6 208、r7 208、r8 307、r9 309、r10 312。r合計1,552件。
- Val: yumike 1,009、mikeryu 49、r2 208、r4 208、環境音10。
- Test: 女性生声take 520、reon 49、yu-ota 51、r3 208、r5 208、環境音10。
- 新規CSV69件はother64とa/ya/ha/pa/u各1。other64はTrain44、Val10、Test10。非other5件はTrain。CSVラベルを保持する。
- TTSのNanami312件は同じ合成話者として全件Trainへ割り当てた。女性合成音声52件を独立Testとしては使わない。女性生声の評価はtake520件。
- 判定が保存されているr音声はkeep/relabel/otherの判定を適用。残る116件はユーザーの指示で収集ラベルを保持する。人間の判定が保存済みと偽って扱わない。

## 全件の意味と検査

データパス10,601件のうち、既存の話者固定CSVで既に整理済みの同一音声別名34パス（rinry13・take21）を除く10,567音声を全件割り当てた。34パスは今回新たに捨てた録音ではなく、対応する同一音声が既存CSVの採用側にあることをSHA-256で検証した。詳細は`full_dataset_coverage.json`。音声が存在しない古いmetadata参照1件は元からデータ件数に含められない。

既存Train6,124・Val1,058・Test620の全7,802行について、filepath/label/speakerと所属を保持した。元CSVは`data_splits/speaker_independent/`に保持し、拡張CSVは`data_splits/speaker_independent_full_20261004/`に保存。

生成時に全件のパス・SHA-256・話者IDが集合間で交差しないことを検査した。環境音は人物IDではなく録音IDで分離し、同一収録セッション内の録音保留評価として扱う。既存collectedの人物対応は未確認で、旧CSVのcollected IDを保持した。初期モデルの過去学習履歴は未確認。

Train8,037件すべてを学習に使い、Val1,484件はモデル選択、Test1,046件はモデル選択後の評価に使う。Val/Testを勾配更新に混ぜない。前処理対象はTrain＋Valの9,521件のみ。

## 移行漏れの修正

話者分割を導入した後も`colab_train_pipeline.py`と`benchmark_strategies.py`が旧ランダムCSVを参照していたため、移行が未完了だった。今回の調査前に私も旧ランキングを有効な独立ベンチマークとして扱った。

- `combined_train.csv`と`combined_val.csv`を新しい話者分割CSVと同じ所属へ更新。旧Testの100件（reon/yu-ota）を含む620件と交差しないことを検証。
- 旧Colab学習入口と旧ランダム分割CLIを停止。専用話者分割runnerへ案内する。
- 専用runner・比較CLI・ランキングの既定分割を同じ`DEFAULT_SPEAKER_SPLIT_DIR`へ統一。
- ランキングはVal全体、Test全体、Testの女性生声takeを使用。旧Speakerphone/Female合成音声を独立Testとして測定しない。
- ランキングキャッシュには分割CSVハッシュを付け、別分割の過去数値を新しいランキングへ流用しない。
- 元のcombined CSVと旧ランキングは`legacy_random_split_before_migration/`にバックアップした。

## 再学習条件

前回の誤った分割で学習したモデルからは再開しない。変更されていない初期`weights/wav2vec2_best`（SHA-256 `55c6e116a8eeca355975071f8a3295b66eeb9116a5bef9d7200d01ee6dc7666d`）から再学習。最大15epoch、batch8、LR3e-5、先頭Transformer4層固定、patience5、seed42。Valの同音統合後Macro-F1で選択し、Test由来のサンプラーと自動HF公開を停止。

既存モデルとの比較は、同じTest1,046件、旧Test620件、女性take520件、新規r3/r5、環境音の内訳を保存する。


## 現在の実行状態

関連テスト9件成功。ローカル前処理9,521件（Train＋Val）が完了し、前処理済みsource_filepathの集合がTrain＋Valと完全一致、Testを含まないことを確認した。保存先は`experiments/full-speaker-20261004-local/processed/`。

拡大後10,567件のColab転送と再学習について、ユーザーが明示的に許可した。専用セッション`vr-full-speaker-20261004`で再学習を開始。初期モデルは変更されていない`weights/wav2vec2_best`を使用し、前回の誤った分割のモデルは使用しない。


## 再学習完了

8エポックで早期停止し、3エポック目を採用。モデル本体と全評価結果を回収し、Colabを停止した。Test全体は51.43%→58.80%、女性takeは41.15%→51.15%。既存男性は96%→74%に低下。詳細は`CORRECTED_TRAINING_RESULT.md`。


## 人物情報の訂正（ユーザー確認）

Rinryは女性1名＋男性複数名の混合データ。rinryの旧speaker列は収集群IDで、人物IDではない。パス・音声ハッシュ・収集群IDの集合間重複はないが、Rinry/collectedの個々の人物対応や既存別名対応が未確定のため、実人物の漏洩が検証済みとは主張できない。正式な継続方針は`Docs/SPEECH_TRAINING_EVALUATION_POLICY.md`。
