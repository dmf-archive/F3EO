import subprocess
import os
from pathlib import Path

def resumable_download(url: str, download_root: str, filename: str):
    """
    使用 PowerShell BitsTransfer 进行可断点续传的下载。
    """
    download_root = Path(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    
    file_path = download_root / filename
    
    # 如果文件已存在，则跳过下载
    if os.path.exists(file_path):
        print(f"File {filename} already exists. Skipping download.")
        return

    print(f"Starting resumable download for {filename}...")
    
    # 使用 PowerShell BitsTransfer
    command = [
        "powershell",
        "-Command",
        f"Start-BitsTransfer -Source {url} -Destination {file_path}"
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded {filename}.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {filename}: {e.stderr}")
        # 如果下载失败，删除可能已创建的不完整文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e
