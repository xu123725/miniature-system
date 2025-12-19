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
                "thought": "User provided no data.",
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
                "thought": "LLM call failed.",
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
You are a senior Meteorologist and Data Scientist. 
Your goal is to answer the user's request by analyzing the provided weather data.

### Dataset Info
- Columns: {columns_info}
- Data Types: {dtypes_info}
- Sample Data:
{head_info}

### Available Tools
You have access to the following python tools. 
Choose the most appropriate tool to solve the problem.
If the user asks for visualization, prefer 'plot_time_series' or 'plot_correlation_heatmap' or 'plot_station_distribution'.
If the user asks for calculation, look for relevant calculation tools.

{tools_desc}

### Instructions
1. Analyze the user's request.
2. Determine if you need to use a tool or just answer directly.
3. If using a tool, specify the tool name and arguments precisely based on the dataset columns.
4. Output MUST be a strictly valid JSON object. Do not include markdown code blocks (```json ... ```).

### Output Format (JSON)
{{
    "thought": "Your reasoning process here...",
    "tool": "tool_name_or_None",
    "args": {{ "arg_name": "value", ... }},
    "response": "Final answer if no tool is needed"
}}

### User Request
{query}

### Conversation History
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
                "thought": "Failed to parse structured response.",
                "tool": "None",
                "response": response_text
            }

    def _execute_plan(self, plan: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the tool specified in the plan.
        """
        thought = plan.get("thought", "Processing...")
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
                "action": f"Attempted to call unknown tool: {tool_name}",
                "result": f"Error: Tool '{tool_name}' not found.",
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
                    "action": f"Called {tool_name}({args})",
                    "result": "Chart generated successfully.",
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
                        "action": f"Called {tool_name}({args})",
                        "result": f"Calculation Result: {msg}",
                        "figure": None
                    }
                else:
                     return {
                        "thought": thought,
                        "action": f"Called {tool_name}({args})",
                        "result": str(result),
                        "figure": None
                    }

        except Exception as e:
            logger.exception(f"Tool Execution Failed: {tool_name}")
            return {
                "thought": thought,
                "action": f"Failed call to {tool_name}",
                "result": f"Error executing tool: {str(e)}",
                "figure": None
            }
