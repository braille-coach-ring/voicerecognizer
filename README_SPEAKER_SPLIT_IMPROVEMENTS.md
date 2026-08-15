# Speaker Split Improvements README

このREADMEは、`feat-test-improve` で入れた未知話者対応まわりの変更点と、実行方法をまとめたものです。

目的は、従来の音声単位ランダム分割ではなく、話者単位で train / validation / test を分離し、「学習時に聞いていない話者」に対する性能を見やすくすることです。

## 変更点

### 1. speaker_id をデータセットに持たせる

`merged_dataset/index.csv` と `processed_dataset/index.csv` に `speaker_id` 列を追加しました。

推定ルールは以下です。

- `dataset\<speaker_id>\<label>\*.wav` は `<speaker_id>` を話者IDにする
- `dataset\collected\<pc_id>\metadata.csv` は `<pc_id>` を話者IDにする
- 旧形式の `machine_id` があるCSVも評価時に読み取る
- 話者IDが取れない場合は `unknown` にする

関連ファイル:

- `src\voicerecognizer\utils\speaker.py`
- `src\voicerecognizer\preprocessing\dataset_builder.py`
- `src\voicerecognizer\dataset\hiragana_dataset.py`

### 2. 学習splitのデフォルトを speaker split に変更

CNN / Wav2Vec2 の学習で、デフォルトの validation split を話者単位にしました。

つまり、同じ `speaker_id` の音声が train と validation の両方に混ざらないようになります。

```powershell
uv run python train.py --model cnn
uv run python train.py --model wav2vec2
```

従来のラベル層化ランダム分割で比較したい場合は、明示的に指定します。

```powershell
uv run python train.py --model cnn --split-mode stratified
uv run python train.py --model wav2vec2 --split-mode stratified
```

話者IDが不足している場合、たとえば全件 `unknown` の場合は、安全のため従来の stratified split にフォールバックします。

関連ファイル:

- `src\voicerecognizer\utils\split_helper.py`
- `src\voicerecognizer\models\cnn\train.py`
- `src\voicerecognizer\models\wav2vec2\train.py`

### 3. Wav2Vec2 の augmentation 適用漏れを修正

Wav2Vec2 学習では `AugmentedSubset` を作っていましたが、DataLoader に渡していたのが元の `train_dataset` だったため、augmentation が実際には効いていませんでした。

今回、DataLoader に `train_augmented_subset` を渡すように修正しました。

```powershell
uv run python train.py --model wav2vec2
```

augmentation を切って比較する場合:

```powershell
uv run python train.py --model wav2vec2 --no-augment
```

### 4. 評価結果に per_speaker を追加

評価JSONに、話者ごとの成績を出す `per_speaker` を追加しました。

出力例:

```json
"per_speaker": {
  "speaker1": {
    "accuracy": 0.8125,
    "total_samples": 160,
    "correct_samples": 130
  }
}
```

また、誤認識サンプルにも `speaker_id` を含めるようにしました。

関連ファイル:

- `src\voicerecognizer\evaluation\evaluator.py`

### 5. 事前学習モデルベンチも speaker split に対応

`script\benchmark_base_models.py` も、デフォルトで speaker split にしました。

```powershell
uv run python script\benchmark_base_models.py --split-mode speaker
```

従来比較:

```powershell
uv run python script\benchmark_base_models.py --split-mode stratified --output-json evaluation_results\base_model_benchmark_stratified.json
```

## まだ未実装のもの

今回の変更は、まず評価と学習分割の土台を正すためのものです。

以下はまだ未実装です。

- Supervised Contrastive Learning
- Hard Negative Mining
- Speaker-Adversarial Learning / Gradient Reversal Layer
- 母音・子音・有声無声などの Multi-task Learning
- 105クラス直接分類から階層分類への変更
- STFT / Mel 設定の本格見直し
- VTLP / RIR / SpecAugment の追加実装

## 実行手順

基本的に `C:\voicelibrary` で実行します。

```powershell
cd C:\voicelibrary
```

### 1. 環境準備

```powershell
uv sync
```

`uv` が使えない環境では、既存の仮想環境を直接使います。

```powershell
.\.venv\Scripts\python.exe --version
```

### 2. データ統合と前処理

speaker_id 付きの index を作り直します。

```powershell
uv run python script\merge_data.py
uv run python script\preprocess.py
```

学習コマンドは通常、自動で統合と前処理を行います。すでに前処理済みデータを使う場合だけ `--skip-prep` を付けます。

```powershell
uv run python train.py --model cnn --skip-prep
```

### 3. CNN を speaker split で学習

```powershell
uv run python train.py --model cnn --from-scratch
```

従来splitとの比較:

```powershell
uv run python train.py --model cnn --from-scratch --split-mode stratified
```

### 4. Wav2Vec2 を speaker split で学習

```powershell
uv run python train.py --model wav2vec2 --from-scratch --epochs 30 --batch-size 4 --learning-rate 0.00003
```

従来splitとの比較:

```powershell
uv run python train.py --model wav2vec2 --from-scratch --split-mode stratified --epochs 30 --batch-size 4 --learning-rate 0.00003
```

augmentation なしとの比較:

```powershell
uv run python train.py --model wav2vec2 --from-scratch --no-augment --epochs 30 --batch-size 4 --learning-rate 0.00003
```

### 5. 評価を実行

CNN:

```powershell
uv run python script\evaluate.py --model-type cnn --output-json evaluation_results\cnn_evaluation_result.json --output-html evaluation_results\cnn_evaluation_report.html
```

Wav2Vec2:

```powershell
uv run python script\evaluate.py --model-type wav2vec2 --output-json evaluation_results\wav2vec2_evaluation_result.json --output-html evaluation_results\wav2vec2_evaluation_report.html
```

`index.csv` の `predicted_text` だけで再評価する場合:

```powershell
uv run python script\evaluate.py --from-dataset-only --output-json evaluation_results\dataset_only_evaluation_result.json
```

### 6. 話者別成績を見る

PowerShell で `per_speaker` を確認します。

```powershell
uv run python -c "import json; p='evaluation_results\\cnn_evaluation_result.json'; d=json.load(open(p, encoding='utf-8')); print(json.dumps(d.get('per_speaker', {}), ensure_ascii=False, indent=2))"
```

最低話者accuracyをざっくり見る場合:

```powershell
uv run python -c "import json; p='evaluation_results\\cnn_evaluation_result.json'; s=json.load(open(p, encoding='utf-8')).get('per_speaker', {}); print(sorted((v['accuracy'], k, v['total_samples']) for k, v in s.items())[:10])"
```

### 7. 事前学習モデルのベンチ

speaker split:

```powershell
uv run python script\benchmark_base_models.py --split-mode speaker --output-json evaluation_results\base_model_benchmark_speaker.json
```

従来split:

```powershell
uv run python script\benchmark_base_models.py --split-mode stratified --output-json evaluation_results\base_model_benchmark_stratified.json
```

この2つの差が大きい場合、ランダム分割の精度は未知話者性能をかなり楽観的に見ていた可能性があります。

## レビュー画面で間違いデータを削除する流れ

評価でレビュー用HTMLを作ります。

```powershell
uv run python script\evaluate.py --model-type cnn
```

レビューサーバーを起動します。

```powershell
uv run python script\review_server.py --port 8765
```

ブラウザで開きます。

```text
http://127.0.0.1:8765/evaluation_results/review_report.html
```

削除したい音声を `delete_candidate` にします。判断結果は `evaluation_results\review_decisions.json` に保存されます。

まず dry-run で削除対象を確認します。

```powershell
uv run python script\delete_review_candidates.py
```

問題なければ実削除します。

```powershell
uv run python script\delete_review_candidates.py --execute
```

削除ログは以下に出ます。

```text
evaluation_results\delete_review_candidates_log.json
```

## 推奨する実験順

### 実験1: speaker split と stratified split の差を見る

目的:

未知話者性能がどれくらい落ちるかを確認します。

実行:

```powershell
uv run python train.py --model cnn --from-scratch --split-mode speaker
uv run python train.py --model cnn --from-scratch --split-mode stratified
```

見るもの:

- Accuracy
- Macro F1
- `per_speaker` の最低accuracy
- `ri`, `pi`, `nyu`, `di` など recall 0% クラス

判断:

- speaker split だけ大きく落ちるなら、データ数より話者数の不足が主因
- stratified split でも落ちるなら、特徴量・クラス不均衡・ラベル品質も疑う

### 実験2: Wav2Vec2 の augmentation あり/なし比較

目的:

今回修正した augmentation が未知話者性能に効くか確認します。

実行:

```powershell
uv run python train.py --model wav2vec2 --from-scratch --split-mode speaker
uv run python train.py --model wav2vec2 --from-scratch --split-mode speaker --no-augment
```

判断:

- augmentation ありで speaker split の Macro F1 が上がるなら継続
- 半濁音や拗音だけ悪化するなら、augmentation が短時間特徴を壊している可能性あり

### 実験3: 事前学習モデルの線形プローブ比較

目的:

自作特徴量より Wav2Vec2 系の表現が未知話者に強いか確認します。

実行:

```powershell
uv run python script\benchmark_base_models.py --split-mode speaker
```

判断:

- 線形プローブで現行モデルより高いなら、WavLM / HuBERT / Wav2Vec2 方向を優先
- 低いなら、前処理・短音声切り出し・ラベル設計の影響が大きい可能性

### 実験4: 低成績話者と混同クラスを確認してデータ収集

目的:

ただサンプル数を増やすのではなく、未知話者性能に効く収集へ寄せます。

見るもの:

- `per_speaker` で低い話者
- confusion matrix の `ri -> bi`, `pi -> bi`, `nyu -> myu`, `chu -> shu`
- confidence が高い誤認識

判断:

- 特定話者だけ低いなら、話者数追加を優先
- 特定クラスだけ低いなら、その混同ペアを重点収集
- confidence 高めの誤認識が多いなら、hard negative mining や calibration の候補

## 検証コマンド

構文チェック:

```powershell
.\.venv\Scripts\python.exe -m compileall src\voicerecognizer\utils\speaker.py src\voicerecognizer\utils\split_helper.py src\voicerecognizer\dataset\hiragana_dataset.py src\voicerecognizer\preprocessing\dataset_builder.py src\voicerecognizer\models\wav2vec2\train.py src\voicerecognizer\models\cnn\train.py src\voicerecognizer\evaluation\evaluator.py script\benchmark_base_models.py tests\test_split_helper.py tests\test_dataset_builder.py tests\test_wav2vec2_lazy_dataset.py tests\test_class_weight_and_augmentation.py tests\test_evaluator.py
```

pytest:

```powershell
uv run pytest tests\test_split_helper.py tests\test_dataset_builder.py tests\test_wav2vec2_lazy_dataset.py tests\test_class_weight_and_augmentation.py tests\test_evaluator.py
```

この環境では、`voicerecognizer` import 時に `.venv\Lib\site-packages\librosa\__init__.pyi` の `PermissionError` が出て、pytest / unittest は完走できませんでした。構文チェックと `git diff --check` は通っています。

