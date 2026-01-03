import matplotlib
# Force Agg backend to prevent GUI issues in server environments
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import platform
from matplotlib import font_manager
from typing import List, Optional, Tuple

# Import logger from config
from src.config import logger

# --- Matplotlib 中文配置 ---
def configure_chinese_font():
    """
    配置 Matplotlib 以支持中文显示。
    根据操作系统自动选择合适的字体。
    """
    system_name = platform.system()
    font_path = None
    
    if system_name == "Windows":
        # 尝试常见的中文字体
        fonts = ["SimHei", "Microsoft YaHei", "SimSun"]
        for font in fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
                break
            except:
                continue
    elif system_name == "Darwin": # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    else: # Linux
        # 尝试常见的 Linux 中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

# 初始化配置
configure_chinese_font()

def plot_time_series(df: pd.DataFrame, x_col: str = 'datetime', y_cols: Optional[List[str]] = None, title: str = "时间序列图", query: str = None) -> plt.Figure:
    """
    生成时间序列折线图。
    
    Args:
        df: 数据框
        x_col: X轴列名 (通常是时间列)
        y_cols: Y轴列名列表
        title: 图表标题
        query: Pandas 查询字符串，用于在绘图前筛选数据 (例如: "站名 == '兰州'")
        
    Returns:
        plt.Figure: Matplotlib 图表对象
    """
    try:
        # Apply query filter if provided
        if query:
            df = df.query(query)
            if df.empty:
                raise ValueError(f"筛选条件 '{query}' 导致数据为空，无法绘图")

    except Exception as e:
        logger.error(f"数据筛选失败 [plot_time_series]: {str(e)}")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"数据筛选失败: {str(e)}", ha='center', va='center')
        return fig

    if y_cols is None:
        # 如果未指定，选择所有数值列
        y_cols = df.select_dtypes(include=['number']).columns.tolist()
        # 排除一些非指标列
        exclude = ['站点号', '经度', '纬度', '年', '月', '日', '时']
        y_cols = [c for c in y_cols if c not in exclude]

    # 创建 Figure 和 Axes，显式控制大小
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 使用 Seaborn 绘图
    # 将数据转换为长格式以便于 Seaborn 处理多列
    try:
        if x_col not in df.columns:
             # 尝试自动寻找时间列
             datetime_cols = df.select_dtypes(include=['datetime']).columns
             if not datetime_cols.empty:
                 x_col = datetime_cols[0]
             else:
                 # 构造一个简单的索引作为X轴
                 x_col = '_index'
                 df['_index'] = range(len(df))

        # 确保时间列是 datetime 类型
        if pd.api.types.is_object_dtype(df[x_col]) or pd.api.types.is_string_dtype(df[x_col]):
             try:
                 df[x_col] = pd.to_datetime(df[x_col])
             except:
                 pass

        # 绘图
        for col in y_cols:
            if col in df.columns:
                sns.lineplot(data=df, x=x_col, y=col, label=col, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel("时间")
        ax.set_ylabel("数值")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.close(fig)
        return fig
        
    except Exception as e:
        logger.error(f"绘图失败 [plot_time_series]: {str(e)}", exc_info=True)
        # 发生错误时返回一个包含错误信息的空图
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"绘图失败: {str(e)}", ha='center', va='center')
        return fig

def plot_correlation_heatmap(df: pd.DataFrame, cols: Optional[List[str]] = None) -> plt.Figure:
    """
    生成相关性热力图。
    
    Args:
        df: 数据框
        cols: 要计算相关性的列名列表
        
    Returns:
        plt.Figure: Matplotlib 图表对象
    """
    try:
        if cols is None:
            # 选择所有数值列
            cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # 过滤存在的列
        cols = [c for c in cols if c in df.columns]
        
        if len(cols) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "数据列不足，无法生成相关性图", ha='center', va='center')
            return fig

        # 计算相关性矩阵
        corr = df[cols].corr()
        
        # 创建 Figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制热力图
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
        
        ax.set_title("相关性热力图")
        plt.tight_layout()
        
        return fig
    except Exception as e:
        logger.error(f"绘图失败 [plot_correlation_heatmap]: {str(e)}", exc_info=True)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"绘图失败: {str(e)}", ha='center', va='center')
        return fig

def plot_station_distribution(df: pd.DataFrame, value_col: str) -> plt.Figure:
    """
    生成站点分布散点图（简单的经纬度分布）。
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 尝试寻找经纬度列
        lat_col = next((c for c in df.columns if '纬' in c or 'lat' in c.lower()), None)
        lon_col = next((c for c in df.columns if '经' in c or 'lon' in c.lower()), None)
        
        if lat_col and lon_col and value_col in df.columns:
            scatter = ax.scatter(df[lon_col], df[lat_col], c=df[value_col], cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(scatter, label=value_col)
            ax.set_xlabel("经度")
            ax.set_ylabel("纬度")
            ax.set_title(f"站点分布 - {value_col}")
            
            # 标注站点名
            if '站名' in df.columns:
                for idx, row in df.iterrows():
                    ax.annotate(row['站名'], (row[lon_col], row[lat_col]), fontsize=8, alpha=0.7)
        else:
            ax.text(0.5, 0.5, "缺少经纬度或数值数据", ha='center', va='center')
            
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"绘图失败 [plot_station_distribution]: {str(e)}", exc_info=True)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"绘图失败: {str(e)}", ha='center', va='center')
        return fig
