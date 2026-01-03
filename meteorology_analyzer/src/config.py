import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("dynamic_meteor")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

# 默认输出目录
OUTPUT_DIR = os.getenv("OUTPUT_DIR", r"c:\Users\15731\PycharmProjects\pythonProject3\meteorology_analyzer\outputs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

for d in [OUTPUT_DIR, LOG_DIR, DATA_DIR, CHART_DIR]:
    os.makedirs(d, exist_ok=True)

# 配置日志
import time
log_filename = f"log_{time.strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding='utf-8')
file_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.addHandler(file_handler)
logger.setLevel(LOG_LEVEL)


# 编码顺序配置
ENCODING_ORDER = os.getenv("ENCODING_ORDER")
if ENCODING_ORDER:
    ENCODING_ORDER = ENCODING_ORDER.split(",")
else:
    ENCODING_ORDER = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']

# LLM配置
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL")  # 新增LLM_MODEL配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0"))
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "60"))

# 如果没有设置LLM相关配置，默认使用DEEPSEEK配置
if LLM_MODEL is None:
    LLM_MODEL = DEEPSEEK_MODEL

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
if LLM_BASE_URL is None:
    LLM_BASE_URL = DEEPSEEK_BASE_URL

LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE")
if LLM_TEMPERATURE is None:
    LLM_TEMPERATURE = DEEPSEEK_TEMPERATURE
else:
    LLM_TEMPERATURE = float(LLM_TEMPERATURE)

LLM_TIMEOUT = os.getenv("LLM_TIMEOUT")
if LLM_TIMEOUT is None:
    LLM_TIMEOUT = DEEPSEEK_TIMEOUT
else:
    LLM_TIMEOUT = int(LLM_TIMEOUT)

# 导出文件配置
EXPORT_FILE_NAME_TEMPLATE = os.getenv("EXPORT_FILE_NAME_TEMPLATE", "气象数据_{timestamp}.csv")

# 用户命令配置
USER_COMMANDS = {
    "帮助": os.getenv("USER_COMMANDS_HELP", "帮助,help,?,功能").split(","),
    "工具列表": os.getenv("USER_COMMANDS_TOOLS", "工具列表,list tools,工具").split(","),
    "导出": os.getenv("USER_COMMANDS_EXPORT", "导出,导出数据,export").split(","),
    "统计": os.getenv("USER_COMMANDS_STATS", "统计,查看统计,stats").split(","),
    "查看结果": os.getenv("USER_COMMANDS_RESULT", "查看结果,结果,show").split(","),
    "可视化": os.getenv("USER_COMMANDS_VISUALIZATION", "可视化,图表,绘图,可视化图表,生成图表,绘制").split(","),
    "计算": os.getenv("USER_COMMANDS_CALCULATION", "计算,计算数据,calculate,计算结果").split(","),
    "退出": os.getenv("USER_COMMANDS_EXIT", "退出,q,quit,exit").split(",")
}

# 结果预览配置
RESULT_PREVIEW_MAX_ITEMS = int(os.getenv("RESULT_PREVIEW_MAX_ITEMS", "10"))

# 响应消息配置
RESPONSE_MESSAGES = {
    "NO_CALCULATION_RESULT": os.getenv("MSG_NO_CALCULATION_RESULT", "⚠️ 尚无最近计算结果。请先执行一次‘计算’。"),
    "NO_VALID_RESULTS": os.getenv("MSG_NO_VALID_RESULTS", "🔎 结果预览：字段 `{field}` 未找到有效值"),
    "LLM_DISABLED": os.getenv("MSG_LLM_DISABLED", "抱歉，暂时无法使用 LLM"),
    "UNKNOWN_COMMAND": os.getenv("MSG_UNKNOWN_COMMAND", "我没有识别出命令。输入 '工具列表' 查看可用工具，或 '帮助' 查看说明。")
}