import hashlib
import logging
from pathlib import Path
from typing import Literal, Optional
from huggingface_hub import HfApi, hf_hub_download, login

from config import DEFAULT_HUGGINGFACE_CONFIG, DEFAULT_RECOGNITION_CONFIG, HuggingFaceConfig

logger = logging.getLogger(__name__)

ModelType = Literal["cnn", "wav2vec2", "best_only", "all"]


def calculate_file_sha256(file_path: Path) -> str:
    """ローカルファイルの SHA-256 ハッシュ値を計算します。"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_remote_file_sha256_map(api: HfApi, repo_id: str, paths: list[str]) -> dict[str, str]:
    """Hugging Face リモートリポジトリ内の指定ファイルの SHA-256 (または LFS oid) を取得します。"""
    remote_map = {}
    try:
        paths_info = api.get_paths_info(repo_id=repo_id, repo_type="model", paths=paths)
        for info in paths_info:
            rpath = info.path
            # LFS ファイルの場合は lfs.sha256、通常ファイルの場合は sha256
            sha = getattr(info.lfs, "sha256", None) if hasattr(info, "lfs") and info.lfs else getattr(info, "sha256", None)
            if sha:
                remote_map[rpath] = sha
    except Exception as e:
        logger.debug("リモートファイルメタデータの取得をスキップしました: %s", e)
    return remote_map


def download_latest_team_weights_if_needed(
    model_type: ModelType = "cnn",
    hf_config: Optional[HuggingFaceConfig] = None,
    weights_dir: Optional[Path] = None,
) -> bool:
    """
    学習開始時等に呼ぶことで、Hugging Face リモートのチーム共有最新モデルと手元のモデルの SHA-256 を比較し、
    手元が古い場合や未存在の場合のみ高速ダウンロードして同期します。
    """
    cfg = hf_config or DEFAULT_HUGGINGFACE_CONFIG
    target_dir = weights_dir or DEFAULT_RECOGNITION_CONFIG.weights_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    token = cfg.token or None
    api = HfApi()

    files_to_sync = []
    if model_type in ["cnn", "best_only", "all"]:
        files_to_sync.extend(["best_model.pth", "labels.json"])

    if not files_to_sync:
        return True

    remote_sha_map = get_remote_file_sha256_map(api, cfg.repo_id, files_to_sync)
    if not remote_sha_map:
        logger.info("ℹ️ リモートリポジトリ %s に事前モデルが見つからないため、ローカルのまま続行します。", cfg.repo_id)
        return True

    for rel_path in files_to_sync:
        local_file = target_dir / rel_path
        remote_sha = remote_sha_map.get(rel_path)

        if local_file.exists() and remote_sha:
            local_sha = calculate_file_sha256(local_file)
            if local_sha.lower() == remote_sha.lower():
                logger.info("ℹ️ 手元の %s はチーム共有の最新モデルと一致しています。(ダウンロード不要)", rel_path)
                continue

        # リモートからダウンロード
        logger.info("📥 チーム共有の最新モデル (%s) をダウンロード中...", rel_path)
        try:
            downloaded_path = hf_hub_download(
                repo_id=cfg.repo_id,
                filename=rel_path,
                repo_type="model",
                token=token,
            )
            # ダウンロードしたファイルを target_dir に配置
            with open(downloaded_path, "rb") as src, open(local_file, "wb") as dst:
                dst.write(src.read())
            logger.info("✨ %s をチーム共有最新版に更新しました。", rel_path)
        except Exception as e:
            logger.warning("チーム最新モデル (%s) の取得をスキップしました: %s", rel_path, e)

    return True


def upload_weights_to_hf(
    model_type: ModelType = "cnn",
    hf_config: Optional[HuggingFaceConfig] = None,
    weights_dir: Optional[Path] = None,
    force_upload: bool = False,
) -> bool:
    """
    config の設定（.env からロードされたトークンおよびリポジトリID）に基づいて、
    指定されたモデルのベスト成果物を Hugging Face Hub へスマートにアップロードします。
    ローカルとリモートで差分がないファイルは送信を自動スキップします。
    """
    cfg = hf_config or DEFAULT_HUGGINGFACE_CONFIG
    target_dir = weights_dir or DEFAULT_RECOGNITION_CONFIG.weights_dir

    if not target_dir.exists():
        logger.error("アップロード対象の weights ディレクトリが存在しません: %s", target_dir)
        return False

    token = cfg.token
    if not token:
        logger.warning(
            "⚠️ HF_TOKEN が設定されていません。.env ファイルまたは環境変数に HF_TOKEN を設定してください。"
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

            # 事前差分判定（リモートの SHA-256 と比較）
            remote_sha_map = {} if force_upload else get_remote_file_sha256_map(api, cfg.repo_id, [r for r, _ in files_to_check])
            uploaded_any = False

            for rel_path, local_file in files_to_check:
                local_sha = calculate_file_sha256(local_file)
                remote_sha = remote_sha_map.get(rel_path)

                if not force_upload and remote_sha and remote_sha.lower() == local_sha.lower():
                    logger.info("ℹ️ %s はリモートと一致しているため送信をスキップします。", rel_path)
                    continue

                logger.info("🚀 Hugging Face Hub (%s) へ %s をアップロード中...", cfg.repo_id, rel_path)
                api.upload_file(
                    path_or_fileobj=str(local_file),
                    path_in_repo=rel_path,
                    repo_id=cfg.repo_id,
                    repo_type="model",
                )
                uploaded_any = True

            if not uploaded_any:
                logger.info("✨ すべての CNN ベストモデルファイルは既にリモートと最新同期されています。")
            else:
                logger.info("✨ CNN ベストモデルのアップロードが完了しました: https://huggingface.co/%s", cfg.repo_id)

        elif model_type == "wav2vec2":
            wav2vec2_best_dir = target_dir / "wav2vec2_best"
            if not wav2vec2_best_dir.exists():
                logger.warning("Wav2Vec2 ベストモデルディレクトリ (%s) が存在しません。", wav2vec2_best_dir)
                return True

            logger.info("🚀 Hugging Face Hub (%s) へ Wav2Vec2 ベストモデルをアップロード中...", cfg.repo_id)
            api.upload_folder(
                folder_path=str(wav2vec2_best_dir),
                path_in_repo="wav2vec2_best",
                repo_id=cfg.repo_id,
                repo_type="model",
                ignore_patterns=["*.log", "*.tmp", ".DS_Store"],
            )
            logger.info("✨ Wav2Vec2 ベストモデルのアップロードが完了しました: https://huggingface.co/%s", cfg.repo_id)

        elif model_type == "best_only":
            logger.info("🚀 Hugging Face Hub (%s) へ全モデルの Best 成果物をアップロード中...", cfg.repo_id)
            api.upload_folder(
                folder_path=str(target_dir),
                repo_id=cfg.repo_id,
                repo_type="model",
                allow_patterns=["best_model.pth", "labels.json", "wav2vec2_best/*"],
                ignore_patterns=["*.log", "*.tmp", ".DS_Store"],
            )
            logger.info("✨ 全 Best 成果物のアップロードが完了しました: https://huggingface.co/%s", cfg.repo_id)

        else:  # "all"
            logger.info("🚀 Hugging Face Hub (%s) へ weights 全ファイルをアップロード中...", cfg.repo_id)
            api.upload_folder(
                folder_path=str(target_dir),
                repo_id=cfg.repo_id,
                repo_type="model",
                ignore_patterns=["*.log", "*.tmp", ".DS_Store"],
            )
            logger.info("✨ 全 weights のアップロードが完了しました: https://huggingface.co/%s", cfg.repo_id)

        return True

    except Exception as e:
        logger.error("❌ Hugging Face へのアップロード中にエラーが発生しました: %s", e)
        return False
