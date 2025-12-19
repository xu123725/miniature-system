# -*- coding: utf-8 -*-
"""
Meteorology Agent Core Logic
Refactored to be stateless and strictly follow the architecture.
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import inspect
from typing import Dict, List, Any, Optional, Union

from src.config import logger
from src.core.llm_service import LLMService
from src.tools.metpy_calcs import generate_all_tools
from src.tools import plotting

class MeteorologyAgent:
    """
    Stateless Meteorology Agent.
    Orchestrates the flow: Query -> LLM (Plan) -> Tool (Execute) -> Result.
    """

    def __init__(self):
        logger.info("Initializing MeteorologyAgent...")
        self.llm = LLMService()
        self.tools = self._register_tools()
        logger.info(f"Loaded {len(self.tools)} tools.")

    def _register_tools(self) -> Dict[str, Any]:
        """
        Register all available tools (Calculation + Plotting).
        """
        # 1. Calculation Tools (from metpy_calcs)
        tools = generate_all_tools()
        
        # 2. Plotting Tools (from plotting)
        # Manually register plotting functions to ensure they are available
        tools['plot_time_series'] = plotting.plot_time_series
        tools['plot_correlation_heatmap'] = plotting.plot_correlation_heatmap
        tools['plot_station_distribution'] = plotting.plot_station_distribution
        
        return tools

    def run(self, query: str, df: pd.DataFrame, history: List[Dict] = None) -> Dict[str, Any]:
        """
        Execute the agent on the user query with the provided dataframe.
        
        Args:
            query: User's natural language query.
            df: The active pandas DataFrame.
            history: Conversation history (optional).
            
        Returns:
            Dict containing 'thought', 'action', 'result', and optionally 'figure'.
        """
        if df is None or df.empty:
            return {
                "thought": "用户未提供数据。",
                "action": "None",
                "result": "请先加载数据文件 (CSV)。",
                "figure": None
            }

        # 1. Build Prompt
        prompt = self._build_prompt(query, df, history)
        
        # 2. Call LLM
        try:
            llm_response = self.llm.query(prompt)
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            return {
                "thought": "LLM 调用失败。",
                "action": "None",
                "result": f"AI 服务暂时不可用: {str(e)}",
                "figure": None
            }

        # 3. Parse Response
        parsed_plan = self._parse_llm_response(llm_response)
        
        # 4. Execute Action (if any)
        execution_result = self._execute_plan(parsed_plan, df)
        
        return execution_result

    def _build_prompt(self, query: str, df: pd.DataFrame, history: List[Dict]) -> str:
        """
        Construct the system prompt for the LLM.
        """
        # Data Profile
        columns_info = ", ".join(df.columns.tolist())
        dtypes_info = str(df.dtypes.to_dict())
        try:
            head_info = df.head(3).to_markdown(index=False)
        except:
            head_info = str(df.head(3))

        # Tool Descriptions
        tools_desc = self._get_tools_description()

        # History Format
        history_str = ""
        if history:
            # Take last 5 turns to save context window
            recent = history[-5:] 
            for msg in recent:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                history_str += f"{role}: {content}\n"

        prompt = f"""
你是一位资深气象学家和数据科学家。
你的目标是通过分析提供的气象数据来回答用户的请求。

### 数据集信息
- 列名: {columns_info}
- 数据类型: {dtypes_info}
- 样例数据:
{head_info}

### 可用工具
你可以使用以下 Python 工具。
选择最合适的工具来解决问题。
如果用户请求可视化，优先使用 'plot_time_series'、'plot_correlation_heatmap' 或 'plot_station_distribution'。
如果用户请求计算，查找相关的计算工具。

{tools_desc}

### 指令
1. 分析用户的请求。
2. 确定是否需要使用工具，或者直接回答。
3. 如果使用工具，根据数据集列名精确指定工具名称和参数。
4. 输出必须是严格有效的 JSON 对象。不要包含 markdown 代码块 (```json ... ```)。

### 输出格式 (JSON)
{{
    "thought": "你的思考推理过程（请用中文）...",
    "tool": "工具名称_或_None",
    "args": {{ "参数名": "值", ... }},
    "response": "如果不需要工具，在此给出最终回答（请用中文）"
}}

### 用户请求
{query}

### 对话历史
{history_str}
"""
        return prompt

    def _get_tools_description(self) -> str:
        """
        Generate descriptions for all tools.
        """
        desc_lines = []
        for name, func in self.tools.items():
            try:
                sig = inspect.signature(func)
                doc = inspect.getdoc(func) or "No description."
                # Truncate doc to first line to save tokens
                doc_summary = doc.split('\n')[0]
                desc_lines.append(f"- {name}{sig}: {doc_summary}")
            except Exception:
                desc_lines.append(f"- {name}: (Dynamic Tool)")
        return "\n".join(desc_lines)

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the JSON response from LLM.
        """
        logger.info(f"Raw LLM Response: {response_text}")
        
        # Clean up markdown code blocks if present
        cleaned_text = re.sub(r'^```json\s*', '', response_text.strip(), flags=re.MULTILINE)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.MULTILINE)
        
        try:
            data = json.loads(cleaned_text)
            return data
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON directly. Attempting regex extraction.")
            # Fallback: Try to find a JSON-like structure
            try:
                match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except:
                pass
            
            # If all fails, treat as text response
            return {
                "thought": "无法解析结构化响应。",
                "tool": "None",
                "response": response_text
            }

    def _execute_plan(self, plan: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the tool specified in the plan.
        """
        thought = plan.get("thought", "正在处理...")
        tool_name = plan.get("tool", "None")
        args = plan.get("args", {})
        text_response = plan.get("response", "")

        if tool_name == "None" or not tool_name:
            return {
                "thought": thought,
                "action": "None",
                "result": text_response,
                "figure": None
            }

        if tool_name not in self.tools:
            return {
                "thought": thought,
                "action": f"尝试调用未知工具: {tool_name}",
                "result": f"错误: 未找到工具 '{tool_name}'。",
                "figure": None
            }

        # Execute Tool
        tool_func = self.tools[tool_name]
        try:
            logger.info(f"Executing tool {tool_name} with args {args}")
            
            # Check if it's a plotting tool (based on module or name)
            is_plotting = tool_name.startswith("plot_")
            
            if is_plotting:
                # Plotting tools expect DataFrame as first arg
                # Ensure args don't contain 'df' as it's passed positionally
                if 'df' in args: del args['df']
                
                fig = tool_func(df, **args)
                return {
                    "thought": thought,
                    "action": f"调用 {tool_name}({args})",
                    "result": "图表生成成功。",
                    "figure": fig
                }
            else:
                # Calculation tools (MetPy) expect records (List[Dict]) usually
                # But let's check if we should pass df or records
                # Based on metpy_calcs.py, they take (records, extra_kwargs) or similar
                # The wrapper `compute_with_metpy` takes `records`.
                # So we convert DF to records.
                records = df.to_dict('records')
                
                # The dynamic tools in metpy_calcs.py have signature: tool(records, extra_kwargs=None)
                # But the generated signature in _get_tools_description might show metpy signature
                # We need to bridge this.
                
                # If the tool is one of the manually wrapped ones in metpy_calcs.py, it expects records.
                # Let's inspect the function to be sure.
                # Actually, in metpy_calcs.py:
                # def tool(records: list, extra_kwargs: Dict[str, Any] = None)
                
                # So we pass records. The args from LLM should go into extra_kwargs?
                # Or does the tool expect args as kwargs?
                # The wrapper: return compute_with_metpy(records, name, extra_kwargs=extra_kwargs)
                # compute_with_metpy likely uses extra_kwargs to find parameters.
                
                # We will pass args as extra_kwargs.
                result = tool_func(records, extra_kwargs=args)
                
                # Format result
                if isinstance(result, dict) and "message" in result:
                    msg = result["message"]
                    # Try to find the result value
                    return {
                        "thought": thought,
                        "action": f"调用 {tool_name}({args})",
                        "result": f"计算结果: {msg}",
                        "figure": None
                    }
                else:
                     return {
                        "thought": thought,
                        "action": f"调用 {tool_name}({args})",
                        "result": str(result),
                        "figure": None
                    }

        except Exception as e:
            logger.exception(f"Tool Execution Failed: {tool_name}")
            return {
                "thought": thought,
                "action": f"调用 {tool_name} 失败",
                "result": f"执行工具时出错: {str(e)}",
                "figure": None
            }
