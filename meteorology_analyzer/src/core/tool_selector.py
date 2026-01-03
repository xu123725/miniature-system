import os
import sys
from typing import Optional, Dict, Callable, List, Any

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 使用绝对导入
from config import logger

class ToolSelector:
    """智能选择要使用的工具"""

    KEYWORD_TOOL_MAPPING = {
        # 露点温度相关
        '露点': 'dewpoint_from_relative_humidity', 'dew': 'dewpoint_from_relative_humidity',
        '露点温度': 'dewpoint_from_relative_humidity', 'dewpoint': 'dewpoint_from_relative_humidity',
        '相对湿度算露点': 'dewpoint_from_relative_humidity',
        
        # 湿球温度相关
        '湿球': 'wet_bulb_temperature', 'wet bulb': 'wet_bulb_temperature',
        '湿球温度': 'wet_bulb_temperature',
        
        # 位温相关
        '位温': 'potential_temperature', 'potential': 'potential_temperature',
        '位势温度': 'potential_temperature', 'theta': 'potential_temperature',
        
        # 虚温相关
        '虚温': 'virtual_temperature', 'virtual': 'virtual_temperature',
        '虚拟温度': 'virtual_temperature',
        
        # 水汽压相关
        '水汽压': 'vapor_pressure', 'vapor pressure': 'vapor_pressure',
        '水汽压力': 'vapor_pressure', '水汽压强': 'vapor_pressure',
        
        # 混合比相关
        '混合比': 'mixing_ratio_from_relative_humidity', 'mixing ratio': 'mixing_ratio_from_relative_humidity',
        '混合比率': 'mixing_ratio_from_relative_humidity',
        
        # 热指数相关
        '热指数': 'heat_index', '体感': 'heat_index', 'heat': 'heat_index',
        '体感温度': 'heat_index', '热浪': 'heat_index',
        
        # 风寒相关
        '风寒': 'windchill', 'windchill': 'windchill',
        '风寒指数': 'windchill', '冷风': 'windchill',
        
        # 相对湿度相关
        '相对湿度': 'relative_humidity_from_dewpoint', 'humidity': 'relative_humidity_from_dewpoint',
        '湿度': 'relative_humidity_from_dewpoint', '相对湿度计算': 'relative_humidity_from_dewpoint',
        
        # 比湿相关
        '比湿': 'specific_humidity_from_dewpoint', 'specific': 'specific_humidity_from_dewpoint',
        '比湿度': 'specific_humidity_from_dewpoint', 'specific humidity': 'specific_humidity_from_dewpoint',
        
        # 气压高度相关
        '气压高度': 'pressure_to_height_std', 'pressure height': 'pressure_to_height_std',
        '气压转高度': 'pressure_to_height_std', '高度计算': 'pressure_to_height_std',
        '标准大气': 'pressure_to_height_std', '气压高度转换': 'pressure_to_height_std',
        
        # 风分量相关
        '分量': 'wind_components', 'component': 'wind_components',
        '风分量': 'wind_components', '风速分量': 'wind_components',
        '风向风速分量': 'wind_components',
        
        # 风向相关
        '风向': 'wind_direction', 'wind direction': 'wind_direction',
        '风的方向': 'wind_direction', '计算风向': 'wind_direction',
        
        # 对流相关
        '对流': 'cape_cin', 'cape': 'cape_cin', 'cin': 'cape_cin',
        '对流有效位能': 'cape_cin', '对流抑制能量': 'cape_cin',
        '大气不稳定': 'cape_cin',
    }

    @classmethod
    def select_tool(cls, user_input: str, available_tools: Dict[str, Callable], context: Optional[Dict[str, Any]] = None) -> Optional[Callable]:
        """根据用户输入和上下文选择最合适的工具"""
        if not user_input or not available_tools:
            logger.warning("用户输入或可用工具为空")
            return None

        user_input_lower = user_input.lower()
        
        # 1. 优先考虑最近使用的工具（基于上下文）
        if context and context.get("last_tool_used") and context.get("last_tool_used") in available_tools:
            last_tool_name = context.get("last_tool_used")
            
            # 检查用户输入是否与最近使用的工具相关
            # 如果包含与工具相关的关键词，或者包含通用操作词（如"再次"、"继续"、"分析"等）
            tool_related_keywords = [keyword for keyword, tool_name in cls.KEYWORD_TOOL_MAPPING.items() if tool_name == last_tool_name]
            general_action_words = ['再次', '继续', '重新', '分析', '处理', '计算']
            
            if any(keyword in user_input_lower for keyword in tool_related_keywords) or any(word in user_input_lower for word in general_action_words):
                logger.info(f"基于上下文，优先使用最近工具: {last_tool_name}")
                return available_tools[last_tool_name]
        
        # 首先尝试精确匹配工具名称
        for tool_name, tool_func in available_tools.items():
            if tool_name.lower() in user_input_lower:
                logger.info(f"检测到工具名称 '{tool_name}'，选择工具")
                return tool_func
        
        # 然后尝试关键词匹配
        # 先收集所有匹配的工具
        matched_tools = []
        for keyword, tool_name in cls.KEYWORD_TOOL_MAPPING.items():
            if keyword in user_input_lower:
                if tool_name in available_tools:
                    matched_tools.append((len(keyword), tool_name))
                    logger.debug(f"检测到关键词 '{keyword}'，匹配工具: {tool_name}")
        
        # 如果有多个匹配，选择最长的关键词对应的工具（更具体的匹配）
        if matched_tools:
            # 按关键词长度降序排序，选择最长的匹配
            matched_tools.sort(key=lambda x: x[0], reverse=True)
            selected_tool_name = matched_tools[0][1]
            logger.info(f"选择工具: {selected_tool_name}")
            return available_tools[selected_tool_name]
        
        # 最后检查是否包含"计算"或"分析"等通用指令
        # 移除过于激进的模糊匹配逻辑，避免将通用问题错误映射到随机工具
        # 只有当用户明确提到工具名称的一部分且长度足够时才匹配
        general_keywords = ['计算', '分析', '求', '算', '计算一下', '分析一下']
        for keyword in general_keywords:
            if keyword in user_input_lower:
                # 尝试匹配工具名称的一部分，但要求匹配的词长度至少为3，且不能是通用词
                for tool_name, tool_func in available_tools.items():
                    terms = tool_name.split('_')
                    # 过滤掉短词和常见词
                    # 增加更多过滤词，确保只有真正独特的术语才会被匹配
                    stop_words = ['get', 'set', 'calc', 'from', 'to', 'std', 'new', 'and', 'the', 'for', 'with', 'value', 'data']
                    valid_terms = [t for t in terms if len(t) >= 4 and t not in stop_words]  # 长度限制提高到4
                    
                    if any(term in user_input_lower for term in valid_terms):
                        logger.info(f"检测到通用计算指令及具体工具关键词，选择工具: {tool_name}")
                        return tool_func
        
        logger.info("没有找到匹配的工具")
        return None