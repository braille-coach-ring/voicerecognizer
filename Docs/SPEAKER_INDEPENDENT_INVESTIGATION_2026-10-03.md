# 話者独立・女性音声汎化の調査と実験計画

調査日: 2026-10-03（日本時間）
対象: C:/Users/yamadarikuto/Mycode/voicerecognizer-review
対象HEAD: 41cbc61be6d06dd31c91556fbecea0e0694f11bb
対象ブランチ: train/speaker-independent-female

## 結論と目標の理解

目標は、点字学習装置における短い日本語音節の105クラス認識について、訓練に登場しない実話者、とりわけ女性話者への認識性能を改善し、男性話者や実機音響への性能も維持することと理解した。TTS生成・Colab同期はそのための手段である。

TTS追加は試す価値がある。ただし、最初に固定スプリットを学習が実際に守るようにし、テスト由来の重点サンプリングを止め、モデル・前処理・データの同一性を記録した実声のみの基準実験を作る方が優先度は高い。現状でTTSとモデル変更を同時に加えると、改善・悪化の原因を区別できない。

本調査ではコード、音声、CSVを変更せず、共有Pythonで読み取り専用の集計を行った。GPU学習、TTS生成、外部アップロードは行っていない。メイン作業ツリーには書き込んでいない。

## ブランチとこれまでの作業

- HEADは41cbc61で、ローカルに保存されたorigin/train/speaker-independent-femaleと一致する。ネットワークfetchは行っていないので、最新リモート状態の保証ではない。
- a28474c: 話者独立CSV、女性録音の修正・整理、Colabで既存学習済み重みを再開しない設定を追加。
- 41cbc61: データインデックスとsplit CSVのパスをPOSIX形式へ修正。
- 3380579および先行コミット: 女性音声・検証音声・スピーカーフォン音声の追加。
- 39b9bdeほか: 音響シミュレーション、同音Accuracy、混同ペアの重点サンプリング、Colab実行連携。
- 未追跡ファイル: analyze_confusion.py、list_voices.py、test_homophones.py。既存作業として保持した。
- 別worktreeのfeat/recognition-strategiesはd44eaa4。共通祖先は26d9d15、対象HEAD側15コミット・ストラテジー側27コミットの分岐がある。主作業ツリーのファイルを読む代わりに対象worktreeからgit showで確認した。

ストラテジーブランチにはscript/generate_ai_female_dataset.py、script/evaluate_female_accuracy.py、script/benchmark_strategies.pyが既にある。Nanamiの標準・高め・低めを各104音節生成する処理は再利用候補。ただし、現行の生成コードは音声が返らない場合にゼロ波形を返し、個別生成失敗も処理を継続するため、そのまま高品質データの生成器として採用しない。失敗時は明示的に未完了とし、波形検査・成功件数・生成設定の記録を追加する。

別ブランチのLEADERBOARD.mdにはphoneme_multiのFemale Acc 76.9%、baseline 67.3%等が記載されているが、今回の620件の固定テストと同一の評価条件・モデル出自は確認できていない。参考値であり、現時点で優劣は確定できない。

## データ調査の実測

全7,802行のCSVについてファイル存在、パス重複、SHA-256によるファイル内容の重複、WAVのサンプルレート・長さを調べた。

| Split | 件数 | 話者の内訳 | 正解ラベル数 | 欠けているラベル |
| --- | ---: | --- | ---: | --- |
| Train | 6,124 | rinry 3,629 / rikutomike 1,022 / haruyamike 990 / collected 384 / rikuto 50 / ryu 49 | 105 | なし |
| Val | 1,058 | yumike 1,009 / mikeryu 49 | 103 | ha, other |
| Test | 620 | take 520 / yu-ota 51 / reon 49 | 104 | de |

- 欠損ファイル0、CSV内の同一パス重複0、調査したファイルのバイト内容が一致する重複0。
- 全ファイル16kHz。中央値はrinry/rikutomike/haruyamikeが0.55秒、yumike/takeが0.5秒、5ラベル話者・collectedが1秒。
- Trainのotherはrinry 4件＋collected 61件、Testのotherはtake 9件。Valにはotherがない。
- rinryのTrain内比率は約59.3%。女性訓練の量は少なくないが、話者多様性は不足する可能性がある。
- yu-ota/reonなどは5ラベルのみ。takeは104ラベルを持つため、話者別Accuracyを同じ難易度の指標として比較しない。
- CSV上で話者IDが分離していても、同一人物の別録音IDが独立である保証にはならない。rikuto/rikutomike、ryu/mikeryu、collectedの実際の人間話者との対応は未確認。
- SHA-256検査は完全一致のみ。再エンコードや切り出しによる近似重複、ラベルの音響的妥当性、人間の話者同一性は未検証。

## 既存評価結果

保存済みJSONの集計値。新規のモデル推論はしていない。

| 指標 | benchmark_si_test.json | speaker_independent_test_after_colab.json |
| --- | ---: | ---: |
| 全体Accuracy | 52.10% | 50.16% |
| 全体macro-F1 | 0.4128 | 0.4335 |
| take Accuracy | 44.81%（233/520） | 47.88%（249/520） |
| yu-ota Accuracy | 88.24%（45/51） | 50.98%（26/51） |
| reon Accuracy | 91.84%（45/49） | 73.47%（36/49） |

female側のAccuracyは上昇し、男性2名は低下している。全体macro-F1は上昇しているため、単一の指標だけで成功・失敗を決めない。JSONにはモデルハッシュ・データハッシュ・実行設定の十分な情報がなく、厳密な同一条件の比較とは判断できない。

再学習後JSONの誤分類全309件が現在のTestのパスに対応していることを確認し、正解例も含めて3組の同音正規化を再計算した。

| 集計 | 正規化Accuracy | 正規化macro-F1 |
| --- | ---: | ---: |
| 全体 | 51.29%（318/620） | 0.4452 |
| take | 48.85%（254/520） | 0.4542 |
| yu-ota | 54.90%（28/51） | 0.1898 |
| reon | 73.47%（36/49） | 0.3175 |

正規化はdi→ji、du→zu、wo→oを正解・予測の双方へ適用した。全体macro-F1は105ラベルから統合した固定102ラベル、話者別macro-F1は既存Evaluatorに合わせて正解・予測に現れるラベルの和集合で算出した。話者別F1の平均対象が異なるため、値の単純比較には限界がある。

3組の同音正規化で救済されるのは7件、Accuracyの差は約1.13ポイント。主な誤りを解消するには音響・表現・データ側の改善が必要であり、正規化だけでは解決しない。上位の混同にはo→po、u→nu、ke→ge、nu→mu、ne→me、ru→gu等がある。

## 優先して解決する問題

### 1. CSVの固定Train/Valが学習段階で失われる

script/colab_train_pipeline.py:42でTrainとValを合流し、前処理済みデータを学習に渡す。
src/voicerecognizer/models/wav2vec2/train.py:1126でそのデータを再分割する。
speaker-aware-splitは既定Falseで、Colab側も指定していない。したがって指定された検証話者の保留を維持する仕組みがない。

既存safe_stratified_splitにCSV結合の順序で7,182行を渡した再現例ではTrain 5,745 / Val 1,437となり、yumikeの814件とmikeryuの39件がTrainへ入り、全8話者IDが両側へ現れた。実学習のファイル列挙順序とは異なり得るため、この件数は過去実行の実測ではなく、現在の既定動作の再現例である。

単にspeaker-aware-splitを付けるだけでも指定したyumike/mikeryuの固定Valは保証されない。前処理後のsource_filepathで元CSVに対応づけ、明示的なTrain/Valインデックスを使うのが修正方針。

### 2. テスト情報が訓練の重点サンプリングへ流れる

train.py:1678の既定参照先evaluation_results/evaluation_result.jsonは、rinry 519件とtake 520件の評価。train.py:1217では混同ペアサンプラーが既定で有効。
テスト話者takeの誤りが訓練時のラベル重点化へ使われる経路が存在する。

まず--no-confusion-pair-samplerで基準実験を行う。再導入する場合は訓練内部の独立した開発評価だけを出所として明示する。過去の実行でこのJSONが読まれたことは実行ログで未確認だが、現在のコードとファイルでは参照条件が成立する。

### 3. 女性話者向けの開発評価がない

指定されたValは男性話者のみ。takeを繰り返し見てTTS量や学習設定を調整すると、takeは実質的に開発評価になる。
既存Valは維持しつつ、新しい女性の開発話者を追加し、最終テストとは分けることを推奨する。可能なら別の女性の最終保留データも収集する。
ha、otherを含む評価範囲の不足も補う。今のテストはdeを評価できない。

### 4. TTSの話者メタデータが前処理で失われる

dataset_builder.py:264はCSVのspeakerを引き継がずパスから推定する。
dataset/tts_augmented/<profile>/<label>/...という構成では推定speakerがtts_augmentedへ統合される。明示されたspeakerを優先し、voice_idとprofile_idを別に記録する必要がある。

Nanamiの3設定は3人の独立話者ではない。同じ音声IDで設定を変えたTestは「未使用設定TTS」と呼ぶ。未知実話者や未知音声IDへの汎化は実声Testと別に評価する。

### 5. Colab連携と成果物の出自

現行run_colab_training.py:108はGitHubのブランチをcloneするだけで、ローカルworktreeの未コミット変更や未追跡TTS・音声を同期しない。Google Drive同期も未実装。
まず固定commitとsplit/audioのmanifest・ハッシュを送り、Colabで同じデータを使用したことを検証する。

Driveは既存認証や運用がある場合に選ぶ。今回の数百件規模のTTSは、既存Colab連携のファイル転送で足りるならDrive専用連携の追加を後回しにできる。利用できるCLI転送機能・認証状態は未調査。

WSL経由のdownloadにC:/...というWindowsパスを渡す処理もあり、WSL側の/mnt/c/...への変換と実際の回収動作の確認が必要。
共有のColabセッション名voicerecognizer-gpuは別作業と競合し得るため、実験ごとに固有名を使う。
pipelineの末尾ではHFへ無条件にアップロードする。研究実行と共有モデルの更新を分け、既定ではローカル回収にとどめる。

### 6. モデル選択の意味と前処理一致

このリポジトリのwav2vec2_lastは「最後のepoch」ではなく、今回のrunで最良macro-F1だったモデルである。したがってColabのlast→bestコピーを、最終epochで最良モデルを上書きするバグとは判断しない。

ただしglobal bestは過去モデルとの比較を含み、今回のモデル評価と区別が必要。実験固有の出力先にrun bestを保存し、そのディレクトリを明示して評価する。既定weightsはworktree内ではなくユーザーキャッシュなので、VOICERECOGNIZER_CACHE_DIRや出力パスもworktree内へ固定する。

--no-resumeはfacebook/wav2vec2-baseの事前学習重みからの新規ファインチューニングであり、ランダム初期化学習ではない。
学習は前処理済みWAV、評価は生WAV＋前処理内包INT8 ONNXを優先する。0.6秒の切り出し、音量処理、FeatureExtractor、ONNX内前処理、量子化の一致を確認し、同一モデルのPyTorch/FP32 ONNX/INT8 ONNXを小さな開発集合で比較する。現時点でこの違いを低精度の原因と断定していない。

## 最短の実験順序

1. 固定splitの実使用、speaker継承、テスト由来サンプラーの停止、実験出力先の隔離、モデル/データ/設定のmanifestを整える。モデルのアーキテクチャは変更しない。
2. 生声のみ6,124件で基準runを行う。Val 1,058件を厳守し、できれば女性開発集合を補強する。小規模の前処理・推論一致確認を先に行う。
3. Nanami標準104件＋Keita標準104件の小さなTTS条件で、同じモデル・split・seed・学習条件を比較する。
4. 改善傾向がある場合、Nanami高低208件を追加し、合計416件の条件を比較する。ピッチ・速度拡張はAudioAugmentorに既にあるため、固定TTS3設定を作る効果は別途検証する。
5. 合成Test104件は実声620件と独立集計する。実声の改善と男性側の悪化を確認し、モデル採用を判断する。合成音声のAccuracy上昇だけでは採用しない。
6. データ効果の確認後、必要ならストラテジーブランチのphoneme_multi等を同じ評価条件で比較する。ブランチ全体のmergeや新モデルの同時追加を最初の実験に混ぜない。

全416件追加時のTrainは6,540件、全104件合成Test追加時のTestは724件。既存生声を削除しない。
TTS生成では16kHz mono PCM_16、生成設定とテキストの記録、無音・空ファイル・欠損・生成失敗の検出、104ラベルのcoverageを必須とする。サービスの読みがラベルの単音節発音に一致するかを小規模な試聴で確認してから全件生成する。失敗をゼロ波形で埋めない。
再現性には学習seedだけでなくAudioAugmentorの乱数やDataLoader workerのseedも含める。

## 外部一次資料

- edge-tts公式リポジトリ: rate/volume/pitchがサポートされ、出力例はMP3。WAVと偽って保存せずデコードする。
  https://github.com/rany2/edge-tts
- Casanova et al., Interspeech 2023: 多話者合成音声とvoice conversionによる低資源ASRデータ拡張の研究。TTS活用の根拠にはなるが、本リポジトリの単音節分類での改善を保証するものではない。
  https://www.isca-archive.org/interspeech_2023/casanova23_interspeech.html
- Rosenberg et al., 2019, Speech Recognition with Augmented Synthesized Speech: 合成音声拡張とdomain transferの検証。
  https://arxiv.org/abs/1909.11699

## 調査の限界

保存済みJSONとソースの静的確認、CSV/音声の読み取り集計、既存分割関数のメモリ上での再現まで実施。新規推論、GPU学習、TTS通信、Google Drive/Colab認証、共有キャッシュのモデル同一性は確認していない。
現段階の主張は、確認したコード経路と保存済みデータに基づく。過去実験の因果関係やTTSの有効性を断定しない。
