import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import contextlib
import traceback
import re
from typing import Dict, List, Any, Optional, Union

from src.config import logger
from src.core.llm_service import LLMService
# 保留 MetPy 供生成的代码调用
import metpy.calc as mpcalc
from metpy.units import units
import src.tools.plotting as plot_tools
class MeteorologyAgent:
    """
    全能气象智能体：支持 CSV/NetCDF，代码解释器模式，Plotly 交互式绘图。
    """

    def __init__(self, api_key: Optional[str] = None):
        self.llm = LLMService(api_key=api_key)
        plot_tools.configure_chinese_font()

    def run(self, query: str, data: Union[pd.DataFrame, xr.Dataset], history: List[Dict] = None) -> Dict[str, Any]:
        """
        执行分析任务。
        注意参数名为 data，以支持 DataFrame 和 Dataset 两种类型。
        """
        if data is None:
            return {"result": "数据未加载。", "code": "", "figure": None}

        # 1. 构建 Prompt (根据数据类型自动调整)
        prompt = self._build_code_prompt(query, data, history)
        
        # 2. 调用 LLM
        try:
            response_text = self.llm.query(prompt)
        except Exception as e:
            return {"result": f"LLM 调用失败: {e}", "code": "", "figure": None}

        # 3. 提取代码
        code = self._extract_code(response_text)
        if not code:
            return {"thought": response_text, "result": response_text, "code": "", "figure": None}

        # 4. 执行代码 (将 data 传入沙箱)
        execution_result = self._execute_code(code, data)
        
        return {
            "thought": "代码生成与执行成功",
            "code": code,
            "result": execution_result["output"],
            "figure": execution_result["figure"],
            "new_df": execution_result.get("new_df")
        }

    def _build_code_prompt(self, query: str, data: Union[pd.DataFrame, xr.Dataset], history: List[Dict]) -> str:
        """根据数据类型构建 Prompt"""
        
        if isinstance(data, xr.Dataset):
            # --- NetCDF/Grid 模式 ---
            data_info = str(data) 
            data_env = f"""
### 数据环境 (Xarray Dataset)
- 变量名为 `ds`
- 数据结构:
{data_info}

**格点数据处理提示**:
- 绘图前通常需要降维 (切片)，例如 `sel(time=..., level=...)`。
- 简单的空间分布图可用 `ds['var'].plot()` (Matplotlib) 或转 DataFrame 后用 `px.scatter_mapbox`。
"""
        else:
            # --- CSV/Table 模式 ---
            # 兼容性处理：防止 data 为空时报错
            try:
                columns = ", ".join(str(c) for c in data.columns[:50])
                head = data.head(3).to_markdown(index=False)
            except Exception:
                columns = "Unknown"
                head = "Empty"

            data_env = f"""
### 数据环境 (Pandas DataFrame)
- 变量名为 `df`
- 列名: {columns}
- 样例:
{head}

**处理提示**:
- 时间列 `df['时间']` 为 datetime 类型。
"""

        return f"""
你是一位精通 Python 的气象数据科学家。请编写代码回答用户问题。

{data_env}

### 可用库 (已预导入)
- **可视化**: `plotly.express as px`, `plotly.graph_objects as go` (首选交互式), `matplotlib.pyplot as plt`
- **数据**: `pandas (pd)`, `xarray (xr)`, `numpy (np)`
- **气象**: `metpy.calc (mpcalc)`, `metpy.units (units)`

### 规则
1. **优先交互式绘图**: 尽可能使用 Plotly (`px` 或 `go`)。
   - 必须将图表对象赋值给变量 `fig` (例如 `fig = px.line(...)`)。
   - 地图可视化请使用 `px.scatter_mapbox(..., mapbox_style="open-street-map")`。
2. **文本输出**: 使用 `print()` 输出最终答案。
3. **不要显示**: 不要调用 `fig.show()` 或 `plt.show()`。

### 用户请求
{query}

### 输出格式
直接输出 Python 代码块 (```python ... ```)。
"""

    def _extract_code(self, text: str) -> str:
        match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1)
        if "import " in text or "=" in text:
            return text
        return ""

    def _execute_code(self, code: str, data: Any) -> Dict[str, Any]:
        logger.info("Executing generated code...")
        output_capture = io.StringIO()
        
        # 注入环境
        exec_globals = {
            "pd": pd, "np": np, "xr": xr,
            "plt": plt, "px": px, "go": go,
            "mpcalc": mpcalc, "units": units
        }
        
        # 动态注入数据变量：这里解决了 'df' 或 'ds' 的定义问题
        if isinstance(data, xr.Dataset):
            exec_globals["ds"] = data
            # 宽容模式：如果 LLM 错误地使用了 df，尝试将其指向 ds (虽然类型不同，但可能部分兼容或至少能跑通部分逻辑，
            # 或者让报错信息变成 AttributeError 而不是 NameError，更易调试)
            exec_globals["df"] = data
            exec_globals["data"] = data 
        else:
            exec_globals["df"] = data
            exec_globals["data"] = data # 通用别名
            
        figure = None
        new_df = None
        
        try:
            plt.clf() # 清理旧图
            with contextlib.redirect_stdout(output_capture):
                exec(code, exec_globals)
            
            # 捕获图表对象
            if "fig" in exec_globals:
                temp_fig = exec_globals["fig"]
                # 1. Plotly Figure
                if isinstance(temp_fig, (go.Figure,)):
                    figure = temp_fig
                # 2. Matplotlib Figure
                elif isinstance(temp_fig, plt.Figure):
                    figure = temp_fig
                # 3. 误判情况：如果 fig 是列表（例如 plt.plot 的返回值），则尝试获取全局 Figure
                elif plt.get_fignums():
                    figure = plt.gcf()
            # 4. 兜底：如果没有 fig 变量但有活动的 Matplotlib Figure
            elif plt.get_fignums():
                figure = plt.gcf()
                
            # 捕获新生成的数据集
            if "result_df" in exec_globals and isinstance(exec_globals["result_df"], pd.DataFrame):
                new_df = exec_globals["result_df"]
                
            output_text = output_capture.getvalue()
            if not output_text and not figure:
                output_text = "代码执行完毕，无输出。"
                
            return {"success": True, "output": output_text, "figure": figure, "new_df": new_df}
            
        except Exception as e:
            return {"success": False, "output": f"代码执行错误: {traceback.format_exc()}", "figure": None}