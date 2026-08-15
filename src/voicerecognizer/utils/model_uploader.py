import hashlib
import logging
from pathlib import Path
from typing import Literal

from huggingface_hub import HfApi, hf_hub_download, login

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG, HuggingFaceConfig, load_env

logger = logging.getLogger(__name__)

ModelType = Literal["cnn", "wav2vec2"]


def calculate_file_sha256(file_path: Path) -> str:
    """ローカルファイルの SHA-256 ハッシュ値を計算します。"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def calculate_file_git_sha1(file_path: Path) -> str:
    """ローカルファイルの Git blob SHA-1 ハッシュ値を計算します。"""
    sha1 = hashlib.sha1()
    file_size = file_path.stat().st_size
    sha1.update(f"blob {file_size}\0".encode())
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha1.update(chunk)
    return sha1.hexdigest()


def is_file_identical_to_remote(local_file: Path, remote_sha: str) -> bool:
    """ローカルファイルがリモートハッシュ（SHA-256またはgit_sha1）と一致しているかチェックします。"""
    if remote_sha.startswith("git_sha1:"):
        expected_git_sha1 = remote_sha.split("git_sha1:", 1)[1]
        return calculate_file_git_sha1(local_file).lower() == expected_git_sha1.lower()
    local_sha = calculate_file_sha256(local_file)
    return local_sha.lower() == remote_sha.lower()


def get_remote_file_sha256_map(api: HfApi, repo_id: str, paths: list[str]) -> dict[str, str]:
    """Hugging Face リモートリポジトリ内の指定ファイルの SHA-256 (LFS) または Git SHA-1 (blob_id) を取得します。"""
    remote_map = {}
    try:
        paths_info = api.get_paths_info(repo_id=repo_id, repo_type="model", paths=paths)
        for info in paths_info:
            rpath = info.path
            # LFS ファイルの場合は lfs.sha256、通常ファイルの場合は sha256
            sha = (
                getattr(info.lfs, "sha256", None)
                if hasattr(info, "lfs") and info.lfs
                else getattr(info, "sha256", None)
            )
            if sha:
                remote_map[rpath] = sha
            elif hasattr(info, "blob_id") and info.blob_id:
                remote_map[rpath] = f"git_sha1:{info.blob_id}"
    except Exception as e:
        logger.debug("リモートファイルメタデータの取得をスキップしました: %s", e)
    return remote_map


def download_latest_team_weights_if_needed(
    model_type: ModelType = "cnn",
    hf_config: HuggingFaceConfig | None = None,
    weights_dir: Path | None = None,
) -> bool:
    """
    学習開始時等に呼ぶことで、Hugging Face リモートのチーム共有最新モデルと手元のモデルの SHA-256 を比較し、
    手元が古い場合や未存在の場合のみ高速ダウンロードして同期します。
    """
    cfg = hf_config or HuggingFaceConfig()
    target_dir = weights_dir or DEFAULT_RECOGNITION_CONFIG.weights_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    token = cfg.token or None
    api = HfApi()

    files_to_sync = []
    if model_type == "cnn":
        files_to_sync.extend(DEFAULT_RECOGNITION_CONFIG.cnn_essential_filenames)
    elif model_type == "wav2vec2":
        files_to_sync.extend(
            [
                f"wav2vec2_best/{fname}"
                for fname in DEFAULT_RECOGNITION_CONFIG.wav2vec2_essential_filenames
            ]
        )

    if not files_to_sync:
        return True

    logger.info("Hugging Face Hub (%s) より最新モデル重み情報を確認中...", cfg.repo_id)
    remote_sha_map = get_remote_file_sha256_map(api, cfg.repo_id, files_to_sync)
    downloaded_any = False

    any_failed = False
    for rel_path in files_to_sync:
        local_file = target_dir / rel_path
        remote_sha = remote_sha_map.get(rel_path)

        if local_file.exists():
            if remote_sha:
                if is_file_identical_to_remote(local_file, remote_sha):
                    logger.debug(
                        "[INFO] 手元の %s はチーム共有の最新モデルと一致しています。(ダウンロード不要)",
                        rel_path,
                    )
                    continue
            else:
                # リモートメタデータが取れなかったがローカルに既に存在する場合はローカルを使用
                logger.debug(
                    "[INFO] リモートのメタデータを取得できませんでしたが、ローカルの %s を利用します。",
                    local_file,
                )
                continue

        # リモートからダウンロード
        logger.info("モデルファイル (%s) を Hugging Face Hub よりダウンロード中...", rel_path)
        try:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=cfg.repo_id,
                    filename=rel_path,
                    repo_type="model",
                    token=token,
                )
            except Exception as first_exc:
                err_lower = str(first_exc).lower()
                if token and ("401" in err_lower or "403" in err_lower or "unauthorized" in err_lower or "invalid" in err_lower):
                    logger.warning(
                        "設定された Hugging Face トークンが無効です。公開モデルのためトークンなしでもダウンロード可能ですが、設定を確認・修正してください: %s",
                        first_exc,
                    )
                    # トークンなしで再試行
                    downloaded_path = hf_hub_download(
                        repo_id=cfg.repo_id,
                        filename=rel_path,
                        repo_type="model",
                        token=None,
                    )
                else:
                    raise first_exc

            local_file.parent.mkdir(parents=True, exist_ok=True)
            # ダウンロードしたファイルを target_dir に配置
            with open(downloaded_path, "rb") as src, open(local_file, "wb") as dst:
                dst.write(src.read())
            logger.info("%s をローカルキャッシュ (%s) に保存しました。", rel_path, local_file)
            downloaded_any = True
        except Exception as e:
            err_str = str(e).lower()
            if "entrynotfounderror" in err_str or "404" in err_str:
                logger.debug(
                    "Hugging Face Hub 上に %s は存在しませんでした (スキップ): %s", rel_path, e
                )
            else:
                logger.warning("モデルファイル (%s) のダウンロードに失敗しました: %s", rel_path, e)
                any_failed = True

    # Wav2Vec2 の場合: ダウンロード完了後、全バリエーションの ONNX がローカルに存在しない、またはダウンロードがあった場合に自動生成
    if model_type == "wav2vec2":
        w2v_best_dir = target_dir / "wav2vec2_best"
        safetensors_path = w2v_best_dir / "model.safetensors"
        required_onnx_files = [
            w2v_best_dir / DEFAULT_RECOGNITION_CONFIG.wav2vec2_mel_int8_onnx_filename,
            w2v_best_dir / DEFAULT_RECOGNITION_CONFIG.wav2vec2_mel_fp32_onnx_filename,
            w2v_best_dir / DEFAULT_RECOGNITION_CONFIG.wav2vec2_int8_onnx_filename,
            w2v_best_dir / DEFAULT_RECOGNITION_CONFIG.wav2vec2_fp32_onnx_filename,
        ]
        is_any_onnx_missing = any(not f.exists() for f in required_onnx_files)

        if safetensors_path.exists() and (is_any_onnx_missing or downloaded_any):
            try:
                logger.info(
                    "ダウンロードした Wav2Vec2 モデルから全バリエーションのローカル ONNX (model_mel_int8 / model_mel_fp32 / model_int8 / model_fp32) を自動生成中..."
                )
                from voicerecognizer.models.wav2vec2.export_onnx import export_and_benchmark

                export_and_benchmark(model_dir=w2v_best_dir, skip_benchmark=True)
            except Exception as e:
                logger.warning("ダウンロード後の ONNX 自動生成中にエラーが発生しました: %s", e)

    return not any_failed


def upload_weights_to_hf(
    model_type: ModelType = "cnn",
    hf_config: HuggingFaceConfig | None = None,
    weights_dir: Path | None = None,
    force_upload: bool = False,
) -> bool:
    """
    config の設定（.env からロードされたトークンおよびリポジトリID）に基づいて、
    指定されたモデルのベスト成果物を Hugging Face Hub へスマートにアップロードします。
    ローカルとリモートで差分がないファイルは送信を自動スキップします。
    """
    if hf_config is not None:
        cfg = hf_config
    else:
        load_env()
        cfg = HuggingFaceConfig()

    if weights_dir is not None:
        target_dir = Path(weights_dir)
    elif Path("weights").exists():
        target_dir = Path("weights")
    else:
        target_dir = DEFAULT_RECOGNITION_CONFIG.weights_dir

    if not target_dir.exists():
        logger.error("アップロード対象の weights ディレクトリが存在しません: %s", target_dir)
        return False

    token = cfg.token
    if not token:
        logger.warning(
            "モデルのアップロードには認証トークンが必要です。環境変数 VOICERECOGNIZER_HF_TOKEN (または HF_TOKEN) を設定してください。"
        )
        return False

    try:
        login(token=token)
        api = HfApi()

        if model_type == "cnn":
            cnn_best_path = target_dir / "best_model.pth"
            labels_path = target_dir / "labels.json"
            files_to_check = []
            if cnn_best_path.exists():
                files_to_check.append(("best_model.pth", cnn_best_path))
            if labels_path.exists():
                files_to_check.append(("labels.json", labels_path))

            if not files_to_check:
                logger.warning("CNN モデルのアップロード対象ファイルが見つかりません。")
                return True

            # 事前差分判定（リモートの SHA-256 / Git SHA-1 と比較）
            remote_sha_map = (
                {}
                if force_upload
                else get_remote_file_sha256_map(api, cfg.repo_id, [r for r, _ in files_to_check])
            )
            files_to_upload = []

            for rel_path, local_file in files_to_check:
                remote_sha = remote_sha_map.get(rel_path)

                if (
                    not force_upload
                    and remote_sha
                    and is_file_identical_to_remote(local_file, remote_sha)
                ):
                    logger.debug(
                        "[INFO] %s はリモートと一致しているため送信をスキップします。", rel_path
                    )
                    continue
                files_to_upload.append(rel_path)

            if not files_to_upload:
                logger.info(
                    "すべての CNN ベストモデルファイルは既にリモートと最新同期されています。送信をスキップします。"
                )
                return True

            logger.info(
                "Hugging Face Hub (%s) へ CNN ベストモデル成果物 (%d 件) を一括アップロード中...",
                cfg.repo_id,
                len(files_to_upload),
            )
            allow_patterns = [Path(rel_path).name for rel_path in files_to_upload]
            api.upload_folder(
                folder_path=str(target_dir),
                repo_id=cfg.repo_id,
                repo_type="model",
                allow_patterns=allow_patterns,
                commit_message=f"Update CNN best model weights ({len(files_to_upload)} files)",
            )
            logger.info(
                "CNN ベストモデルのアップロードが完了しました: https://huggingface.co/%s",
                cfg.repo_id,
            )

        elif model_type == "wav2vec2":
            wav2vec2_best_dir = target_dir / "wav2vec2_best"
            if not wav2vec2_best_dir.exists():
                logger.warning(
                    "Wav2Vec2 ベストモデルディレクトリ (%s) が存在しません。", wav2vec2_best_dir
                )
                return True

            # HF容量節約のため、ONNX はアップロードせず、オリジナルの model.safetensors と設定 JSON のみをアップロード
            essential_filenames = [
                "model.safetensors",
                "labels.json",
                "config.json",
                "preprocessor_config.json",
                "vocab.json",
                "tokenizer_config.json",
            ]
            files_to_check = []
            for fname in essential_filenames:
                fpath = wav2vec2_best_dir / fname
                if fpath.exists():
                    files_to_check.append((f"wav2vec2_best/{fname}", fpath))

            if not files_to_check:
                logger.warning("Wav2Vec2 モデルのアップロード対象ファイルが見つかりません。")
                return True

            remote_sha_map = (
                {}
                if force_upload
                else get_remote_file_sha256_map(api, cfg.repo_id, [r for r, _ in files_to_check])
            )
            files_to_upload = []

            for rel_path, local_file in files_to_check:
                remote_sha = remote_sha_map.get(rel_path)

                if (
                    not force_upload
                    and remote_sha
                    and is_file_identical_to_remote(local_file, remote_sha)
                ):
                    logger.debug(
                        "[INFO] %s はリモートと一致しているため送信をスキップします。", rel_path
                    )
                    continue
                files_to_upload.append(rel_path)

            if not files_to_upload:
                logger.info(
                    "すべての Wav2Vec2 ベストモデルファイルは既にリモートと最新同期されています。送信をスキップします。"
                )
                return True

            logger.info(
                "Hugging Face Hub (%s) へ Wav2Vec2 ベストモデル成果物 (%d 件) を一括アップロード中...",
                cfg.repo_id,
                len(files_to_upload),
            )
            allow_patterns = [Path(rel_path).name for rel_path in files_to_upload]
            api.upload_folder(
                folder_path=str(wav2vec2_best_dir),
                path_in_repo="wav2vec2_best",
                repo_id=cfg.repo_id,
                repo_type="model",
                allow_patterns=allow_patterns,
                commit_message=f"Update Wav2Vec2 best model weights ({len(files_to_upload)} files)",
            )
            logger.info(
                "Wav2Vec2 ベストモデルのアップロードが完了しました: https://huggingface.co/%s",
                cfg.repo_id,
            )

        else:
            logger.warning(
                "未対応の model_type: %s (cnn または wav2vec2 を指定してください)", model_type
            )

        return True

    except Exception as e:
        logger.error("Hugging Face へのアップロード中にエラーが発生しました: %s", e)
        return False
