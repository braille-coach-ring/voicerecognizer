"""
Autonomous Colab GPU Training Runner for VoiceRecognizer
Executes Wav2Vec2 adaptation fine-tuning on Colab GPU via WSL Colab CLI.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Windows環境での cp932 エンコーディングエラー防止
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_NAME = "voicerecognizer-gpu"


def run_wsl_stream(args_list: list[str], input_text: str | None = None, check: bool = True) -> int:
    cmd = ["wsl", "-u", "root", "--", *args_list]
    cmd_str = " ".join(cmd)
    if input_text:
        print(f"\n[EXEC] {cmd_str} (piped script: {len(input_text)} chars)", flush=True)
    else:
        print(f"\n[EXEC] {cmd_str}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input_text else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if input_text and proc.stdin is not None:
        proc.stdin.write(input_text)
        proc.stdin.close()

    if proc.stdout is not None:
        for line in iter(proc.stdout.readline, ""):
            try:
                print(line, end="", flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)
        proc.stdout.close()
    return_code = proc.wait()

    if check and return_code != 0:
        raise RuntimeError(f"Command failed with code {return_code}: {cmd_str}")
    return return_code


def main():
    parser = argparse.ArgumentParser(description="VoiceRecognizer Colab GPU Training Runner")
    parser.add_argument(
        "--epochs", type=int, default=12, help="Epochs for Wav2Vec2 fine-tuning (default: 12)"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument(
        "--freeze-layers", type=int, default=4, help="Transformer layers to freeze (default: 4)"
    )
    parser.add_argument("--gpu", type=str, default="T4", help="Colab GPU type (T4, L4, A100)")
    parser.add_argument(
        "--branch",
        type=str,
        default="",
        help="Git branch to clone on Colab (default: current git branch)",
    )
    parser.add_argument(
        "--keep-session", action="store_true", help="Keep Colab VM alive after training"
    )
    args = parser.parse_args()

    # 対象ブランチの特定
    target_branch = args.branch
    if not target_branch:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            target_branch = res.stdout.strip()
        except Exception:
            target_branch = "feat/male-to-female-voice-augmentation"

    # .env から HF_TOKEN を取得
    env_path = PROJECT_ROOT / ".env"
    hf_token = ""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") or line.startswith("VOICERECOGNIZER_HF_TOKEN="):
                hf_token = line.split("=", 1)[1].strip()

    colab_bin = ["/root/.local/bin/colab", "--auth=oauth2"]

    print("=" * 70, flush=True)
    print("   VOICERECOGNIZER COLAB GPU TRAINING RUNNER")
    print(f"   Target GPU: {args.gpu} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"   HF Token Present: {bool(hf_token)}")
    print("=" * 70, flush=True)

    start_time = time.time()

    try:
        # 1. GPU インスタンスの確保
        print(f"\n[Step 1] Provisioning Fresh Colab {args.gpu} GPU Instance...", flush=True)
        run_wsl_stream([*colab_bin, "new", "-s", SESSION_NAME, "--gpu", args.gpu])

        print("Waiting 25 seconds for Colab VM & Jupyter kernel to fully stabilize...", flush=True)
        time.sleep(25)
        run_wsl_stream([*colab_bin, "status", "-s", SESSION_NAME])

        # 2. リモート環境のセットアップ (クローンと依存導入)
        print(
            "\n[Step 2] Cloning repository and installing dependencies on Colab GPU...", flush=True
        )
        setup_script = (
            "import os, subprocess\n"
            "subprocess.run(['apt-get', 'update'], check=True)\n"
            "subprocess.run(['apt-get', 'install', '-y', 'libportaudio2', 'ffmpeg'], check=True)\n"
            "os.chdir('/content')\n"
            "subprocess.run(['rm', '-rf', 'voicerecognizer'], check=False)\n"
            f"subprocess.run(['git', 'clone', '-b', '{target_branch}', 'https://github.com/braille-coach-ring/voicerecognizer.git'], check=True)\n"
            "os.chdir('/content/voicerecognizer')\n"
            "subprocess.run(['pip', 'install', 'uv'], check=True)\n"
            "subprocess.run(['uv', 'pip', 'install', '--system', '-e', '.', 'soundfile', 'librosa', 'onnx', 'onnxruntime', 'tqdm', 'transformers', 'accelerate', 'huggingface_hub', 'scikit-learn'], check=True)\n"
            "print('Environment setup complete on Colab.', flush=True)\n"
        )

        # リトライ付きでセットアップ実行
        for attempt in range(1, 4):
            try:
                run_wsl_stream([*colab_bin, "exec", "-s", SESSION_NAME], input_text=setup_script)
                break
            except Exception as e:
                print(f"Setup attempt {attempt} failed: {e}. Retrying in 15 seconds...", flush=True)
                time.sleep(15)
                if attempt == 3:
                    raise

        # 3. Colab 統合パイプラインの実行 (前処理 -> GPU学習 -> ONNXエクスポート -> 未見テスト評価 -> HFアップロード)
        print(
            f"\n[Step 3] Running Full GPU Training & Evaluation Pipeline ({args.epochs} epochs)...",
            flush=True,
        )
        hf_env_str = (
            f"os.environ['HF_TOKEN'] = '{hf_token}'\nos.environ['VOICERECOGNIZER_HF_TOKEN'] = '{hf_token}'\n"
            if hf_token
            else ""
        )
        run_script = (
            f"import os, subprocess, sys\n"
            f"os.chdir('/content/voicerecognizer')\n"
            f"{hf_env_str}"
            f"p = subprocess.Popen(['python', '-u', 'script/colab_train_pipeline.py', '{args.epochs}', '{args.batch_size}', '{args.lr}', '{args.freeze_layers}'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
            f"for line in p.stdout:\n"
            f"    print(line, end='', flush=True)\n"
            f"p.wait()\n"
            f"if p.returncode != 0:\n"
            f"    raise RuntimeError(f'colab_train_pipeline failed with exit code {{p.returncode}}')\n"
        )
        run_wsl_stream(
            [*colab_bin, "exec", "-s", SESSION_NAME, "--timeout", "7200"], input_text=run_script
        )

        # 4. 評価結果 JSON と HTML をローカルに回収
        print("\n[Step 4] Downloading evaluation results to local...", flush=True)
        local_results_dir = PROJECT_ROOT / "evaluation_results"
        local_results_dir.mkdir(exist_ok=True)
        local_json_dest = "/mnt/c/Users/yamadarikuto/Mycode/voicerecognizer/evaluation_results/speakerphone_test_wav2vec2_colab_after.json"
        local_html_dest = "/mnt/c/Users/yamadarikuto/Mycode/voicerecognizer/evaluation_results/speakerphone_test_wav2vec2_colab_after.html"
        run_wsl_stream(
            [
                *colab_bin,
                "download",
                "-s",
                SESSION_NAME,
                "/content/voicerecognizer/evaluation_results/speakerphone_test_wav2vec2_colab_after.json",
                local_json_dest,
            ]
        )
        try:
            run_wsl_stream(
                [
                    *colab_bin,
                    "download",
                    "-s",
                    SESSION_NAME,
                    "/content/voicerecognizer/evaluation_results/speakerphone_test_wav2vec2_colab_after.html",
                    local_html_dest,
                ]
            )
        except Exception as e:
            print(f"HTML download skipped: {e}")

    finally:
        if not args.keep_session:
            print("\n[Step 5] Releasing Colab GPU VM to Prevent Idle Charges...", flush=True)
            run_wsl_stream([*colab_bin, "stop", "-s", SESSION_NAME], check=False)
            print("\nColab VM released. Current balance:", flush=True)
            run_wsl_stream([*colab_bin, "usage"], check=False)
        else:
            print(f"\n[Info] Session '{SESSION_NAME}' kept alive.", flush=True)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70, flush=True)
    print(f" COLAB GPU TRAINING RUN COMPLETED in {elapsed:.1f}s", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
