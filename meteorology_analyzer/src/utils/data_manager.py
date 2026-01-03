import os
import shutil
import pandas as pd
from datetime import datetime
from src.config import logger

class DataManager:
    def __init__(self, upload_dir="data/uploads"):
        """初始化数据管理器"""
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(current_dir))
        
        self.upload_dir = os.path.join(base_dir, upload_dir)
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_uploaded_file(self, uploaded_file) -> bool:
        """保存上传的文件"""
        try:
            file_path = os.path.join(self.upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"File saved: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            return False

    def list_files(self) -> pd.DataFrame:
        """列出所有已上传的文件"""
        files = []
        if os.path.exists(self.upload_dir):
            for f in os.listdir(self.upload_dir):
                if f.startswith('.'): continue # 跳过隐藏文件
                
                file_path = os.path.join(self.upload_dir, f)
                if os.path.isfile(file_path):
                    stats = os.stat(file_path)
                    files.append({
                        "id": f,  # 使用文件名作为ID
                        "filename": f,
                        "size": stats.st_size,
                        "upload_time": datetime.fromtimestamp(stats.st_mtime)
                    })
        
        if not files:
            return pd.DataFrame(columns=["id", "filename", "size", "upload_time"])
            
        return pd.DataFrame(files).sort_values("upload_time", ascending=False)

    def get_file_path(self, file_id: str) -> str:
        """根据ID获取文件路径"""
        return os.path.join(self.upload_dir, file_id)
