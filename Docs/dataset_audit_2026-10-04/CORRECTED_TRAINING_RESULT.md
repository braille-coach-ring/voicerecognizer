# 修正した話者分割での再学習結果（2026-10-04）

Train8,037件、Val1,484件、Test1,046件。全10,567音声を話者分割へ割り当て、Train全件を使用した。既存の話者固定CSV7,802行の所属・ラベル・話者を保持し、新規r全2,384件（未保存116件は収集ラベル）、新規CSV録音69件、TTS312件を追加した。分割の詳細は`CORRECTED_FULL_SPEAKER_SPLIT.md`。

最大15エポック、patience5で実行。3エポック目のVal Macro-F1が最良（学習中0.7760）で、その後5エポック改善せず8エポックで早期停止した。Testのスコアでチェックポイントを選び直していない。回収したモデルを再評価したValの同音統合後正解率77.70%、Macro-F1 0.7773。

| 固定評価集合 | 件数 | 学習前正解率 | 再学習後正解率 | 差 |
| --- | ---: | ---: | ---: | ---: |
| Test全体 | 1046 | 51.43%（538件） | 58.80%（615件） | +7.36ポイント |
| 旧Test（take/reon/yu-ota） | 620 | 50.00%（310件） | 54.84%（340件） | +4.84ポイント |
| 女性生声take | 520 | 41.15%（214件） | 51.15%（266件） | +10.00ポイント |
| 既存男性reon/yu-ota | 100 | 96.00%（96件） | 74.00%（74件） | -22.00ポイント |
| 新規男性r3/r5 | 416 | 54.57%（227件） | 65.38%（272件） | +10.82ポイント |
| 環境音other判定 | 10 | 10.00%（1件） | 30.00%（3件） | +20.00ポイント |

同音統合はdi→ji、du→zu、wo→o。Test全体のMacro-F1は0.4687→0.5671、厳密正解率は50.38%→57.74%。女性生声takeの成績は1名での回帰評価。

既存男性の内訳はreon49/49→39/49（100%→79.59%）、yu-ota47/51→35/51（92.16%→68.63%）。Test全体・女性take・新規rは改善したが、既存男性と環境音otherの性能はまだ十分でない。全体改善のみを理由に既定モデルへ自動置換していない。

## 検証と保存

モデル本体378,408,212 bytesを全回収。モデルSHA-256がTest/Valの評価JSONと一致すること、保存予測から全評価指標の再計算一致、分割CSVと評価行の完全一致、初期モデルが変更されていないことを確認した。Colabセッションは停止済み。

- 最良モデル: `experiments/vr-full-speaker-20261004/results/experiments/comparison/warm_start/checkpoint/`
- SHA-256: `8edcc3d34f1e0ecc9e357338eee07ca0653ff3679f68ead5c4bd7b0811fc8c32`
- 全予測と各評価集合JSON: `experiments/vr-full-speaker-20261004/results/experiments/comparison/`
- 全実行ログ: `experiments/vr-full-speaker-20261004/results/experiments/comparison.log`
- 入力ハッシュ記録: `experiments/vr-full-speaker-20261004/bundle_manifest.json`
- 集約結果: `Docs/dataset_audit_2026-10-04/corrected_training_result_summary.json`

初期モデルの過去学習履歴は未確認で、今回の固定Testは過去に参照済みの回帰評価。環境音のTest10件は同じ収録セッション内の録音保留であり、未知環境全体の評価ではない。既存collectedの人物対応は未確認で、旧CSVのcollected IDを保持した。PyTorchチェックポイントを保存し、ONNX変換・HF公開・既定モデル置換は行っていない。
