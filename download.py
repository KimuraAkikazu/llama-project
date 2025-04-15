from huggingface_hub import hf_hub_download
from pathlib import Path
import os

def download_model(repo_id, filename, save_path=None):
    """
    モデルをHugging Face Hubからダウンロードし、指定されたパスに保存します。
    """
    # 保存先ディレクトリの作成（存在しない場合）
    save_path = Path(save_path) if save_path else Path.cwd() / "models"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 明示的にキャッシュを使用しない設定
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=save_path,
        local_dir_use_symlinks=False,  # シンボリックリンクではなく実ファイルを保存
        force_download=True,
        resume_download=False
    )
    
    print(f"ダウンロード先: {model_path}")
    print(f"保存先ディレクトリの内容: {os.listdir(save_path)}")
    return model_path

if __name__ == "__main__":
    REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    FILENAME = "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    SAVE_PATH = "./models"  # カレントディレクトリからの相対パス

    # 絶対パスに変換して渡す
    absolute_path = Path(SAVE_PATH).resolve()
    download_model(REPO_ID, FILENAME, absolute_path)

    from llama_cpp import Llama
