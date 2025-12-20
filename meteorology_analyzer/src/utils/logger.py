import logging
import os
import sys
from datetime import datetime

def setup_logger(name="meteorology_analyzer", log_level=logging.INFO):
    """
    配置并返回一个日志记录器。
    
    Args:
        name: 日志记录器名称
        log_level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件名包含日期
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 获取logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# 全局默认 Logger
logger = setup_logger()
