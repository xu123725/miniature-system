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
from src.utils.knowledge_base import KnowledgeBaseManager

class MeteorologyAgent:
    """
    Stateless Meteorology Agent.
    Orchestrates the flow: Query -> LLM (Plan) -> Tool (Execute) -> Result.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Agent.
        
        Args:
            api_key: Optional API Key for LLM service injection.
        """
        logger.info(f"Initializing MeteorologyAgent with api_key type: {type(api_key)}")
        try:
            self.llm = LLMService(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize LLMService: {e}")
            # Fallback to a dummy LLM service or handle gracefully
            self.llm = None
            
        # Initialize Knowledge Base Manager
        try:
            self.kb_manager = KnowledgeBaseManager()
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeBaseManager: {e}")
            self.kb_manager = None

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
        
        # === 新增：数据查询工具 ===
        def query_dataframe(df: pd.DataFrame, query_expr: str = None, n: int = 5) -> Dict[str, Any]:
            """
            根据条件查询数据。
            Args:
                query_expr: pandas query 字符串 (例如: "Temperature > 30" 或 "Date == '2023-01-01'")
                n: 返回的行数限制
            """
            try:
                # 如果有查询条件，先筛选
                if query_expr:
                    result = df.query(query_expr)
                else:
                    result = df
                
                # 即使筛选后数据可能还是很多，只返回前 n 行和统计信息
                summary = f"共找到 {len(result)} 条记录。前 {n} 条如下：\n"
                summary += result.head(n).to_markdown(index=False)
                return {"text": summary, "data": result}
            except Exception as e:
                return {"text": f"查询失败: {str(e)}", "data": None}

        tools['query_dataframe'] = query_dataframe
        
        # === 新增：知识库查询工具 ===
        def consult_knowledge_base(query: str) -> str:
            """
            查询气象专业知识库。
            当用户询问气象学术语定义、物理原理、公式或行业标准规范时，必须使用此工具。
            Args:
                query: 具体的查询问题（例如"什么是露点温度"或"气象观测规范"）
            Returns:
                相关文档片段的合并文本。
            """
            if not self.kb_manager:
                return "知识库管理器未初始化，无法查询。"
            
            try:
                results = self.kb_manager.search(query, k=3)
                if not results or (len(results) == 1 and "not found" in results[0]):
                    return "知识库中未找到相关信息。"
                
                combined_text = "\n\n".join([f"--- 片段 {i+1} ---\n{text}" for i, text in enumerate(results)])
                return f"从知识库中检索到以下相关信息：\n\n{combined_text}"
            except Exception as e:
                return f"知识库查询失败: {str(e)}"

        tools['consult_knowledge_base'] = consult_knowledge_base
        # ==========================
        
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
        all_columns = df.columns.tolist()
        
        # Prevent token explosion if too many columns
        if len(all_columns) > 50:
            columns_info = ", ".join(all_columns[:50]) + f"...(等{len(all_columns)}列)"
        else:
            columns_info = ", ".join(all_columns)

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
**重要**：'plot_time_series' 支持 'query' 参数。如果用户只想要特定站点或时间段的数据（例如“兰州的气温”），**请直接在绘图工具中使用 query 参数进行筛选**，而不要先调用 query_dataframe。
示例：`plot_time_series(x_col='时间', y_cols=['气温_C'], query="站名.str.contains('兰州')")`
如果用户请求计算，查找相关的计算工具。

{tools_desc}

### 指令
1. 分析用户的请求。
2. 确定是否需要使用工具，或者直接回答。
3. **关键规则**：不要在 args 中传递 'df'、'data' 或 'records' 参数。系统会自动将当前数据集注入给工具。你只需要指定列名（如 x_col, y_cols）和其他配置参数。
4. **查询规则**：
   - 数据加载时已自动生成名为 '时间' 的 datetime 列，绘图时 x_col 优先使用该列。
   - 数据加载时已进行列名清洗，特殊符号（如 ℃, %, 括号）已被替换或移除。
   - 严禁在 query 语句中使用反引号（backticks），直接使用清洗后的列名（例如使用 气温_C 而不是 气温(℃)）。
   - 请务必参考“数据集信息”中展示的实际列名。
5. 遇到气象专业概念、定义或标准规范类问题，优先调用 consult_knowledge_base 工具查询，根据检索结果回答，不要仅凭记忆编造。
6. 输出必须是严格有效的 JSON 对象。请使用 markdown 代码块 (```json ... ```)。

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
                "figure": None,
                "new_df": None
            }

        if tool_name not in self.tools:
            return {
                "thought": thought,
                "action": f"尝试调用未知工具: {tool_name}",
                "result": f"错误: 未找到工具 '{tool_name}'。",
                "figure": None,
                "new_df": None
            }

        # Execute Tool
        tool_func = self.tools[tool_name]
        try:
            logger.info(f"Executing tool {tool_name} with args {args}")
            
            # --- 修复核心逻辑开始 ---
            
            # 1. 如果是绘图工具 (plot_xxx) 或数据查询工具 (query_dataframe)：自动注入 DataFrame
            if tool_name.startswith("plot_") or tool_name == 'query_dataframe':
                # 强制删除 LLM 可能错误生成的 df/data 参数，防止干扰
                args.pop('df', None)
                args.pop('data', None)
                
                # 【关键】直接将内存中的真实 df 对象传给函数
                result_obj = tool_func(df, **args)
                
                # 根据返回类型封装结果
                if tool_name.startswith("plot_"):
                     return {
                        "thought": thought,
                        "action": f"调用 {tool_name}",
                        "result": "图表生成成功。",
                        "figure": result_obj,
                        "new_df": None
                    }
                else:
                    # query_dataframe 返回的是字典 {"text": ..., "data": ...}
                    # 我们取 text 展示给用户，data 赋值给 new_df
                    text_result = result_obj.get("text", str(result_obj))
                    new_generated_df = result_obj.get("data", None)
                    
                    return {
                        "thought": thought,
                        "action": f"调用 {tool_name}",
                        "result": text_result,
                        "figure": None,
                        "new_df": new_generated_df
                    }
            
            # 2. 如果是计算工具 (MetPy)：自动注入 Records
            else:
                args.pop('records', None)
                # 将 DataFrame 转为记录列表传给计算工具
                records = df.to_dict('records')
                result = tool_func(records, extra_kwargs=args)
                
                # Format result
                if isinstance(result, dict) and "message" in result:
                    msg = result["message"]
                    # Try to find the result value
                    return {
                        "thought": thought,
                        "action": f"调用 {tool_name}({args})",
                        "result": f"计算结果: {msg}",
                        "figure": None,
                        "new_df": None
                    }
                else:
                     return {
                        "thought": thought,
                        "action": f"调用 {tool_name}({args})",
                        "result": str(result),
                        "figure": None,
                        "new_df": None
                    }
            
            # --- 修复核心逻辑结束 ---

        except Exception as e:
            logger.exception(f"Tool Execution Failed: {tool_name}")
            return {
                "thought": thought,
                "action": f"调用 {tool_name} 失败",
                "result": f"执行工具时出错: {str(e)}",
                "figure": None,
                "new_df": None
            }
