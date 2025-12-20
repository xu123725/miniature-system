import sys
import os

import inspect
from typing import List, Dict, Any, Callable, Optional

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 使用绝对导入
from src.config import logger

# 懒加载metpy.calc模块
_mpcalc = None

def _get_mpcalc():
    global _mpcalc
    if _mpcalc is None:
        import metpy.calc as mpcalc
        _mpcalc = mpcalc
    return _mpcalc

# 创建全局缓存变量
_TOOLS_CACHE = None

# 懒加载metpy.calc模块
_metpy_calc = None

def _get_mpcalc():
    global _metpy_calc
    if _metpy_calc is None:
        import metpy.calc
        _metpy_calc = metpy.calc
    return _metpy_calc

# 手动定义的工具元数据
METPY_FUNCTIONS_MANUAL = {
    "dewpoint": {
        "description": "计算露点温度",
        "required_params": ["temperature", "relative_humidity"],
        "example": "dewpoint(temperature=25, relative_humidity=70)",
        "group": "热力学"
    },
    "wet_bulb_temperature": {
        "description": "计算湿球温度",
        "required_params": ["temperature", "pressure", "relative_humidity"],
        "example": "wet_bulb_temperature(temperature=25, pressure=1013, relative_humidity=70)",
        "group": "热力学"
    },
    "wind_speed": {
        "description": "计算风速",
        "required_params": ["u", "v"],
        "example": "wind_speed(u=10, v=20)",
        "group": "风分析"
    },
    "wind_direction": {
        "description": "计算风向",
        "required_params": ["u", "v"],
        "example": "wind_direction(u=10, v=20)",
        "group": "风分析"
    },
    "advection": {
        "description": "计算平流",
        "required_params": ["field", "u", "v"],
        "example": "advection(temperature=25, u=10, v=20)",
        "group": "动力学"
    },
    "vorticity": {
        "description": "计算涡度",
        "required_params": ["u", "v", "dx", "dy"],
        "example": "vorticity(u=10, v=20, dx=1000, dy=1000)",
        "group": "动力学"
    },
    "divergence": {
        "description": "计算散度",
        "required_params": ["u", "v", "dx", "dy"],
        "example": "divergence(u=10, v=20, dx=1000, dy=1000)",
        "group": "动力学"
    }
}

def generate_all_tools() -> Dict[str, Callable]:
    """
    生成所有可用的气象计算工具
    
    Returns:
        Dict[str, Callable]: 工具名称到工具函数的映射
    """
    global _TOOLS_CACHE
    
    # 如果缓存存在，返回缓存的副本
    if _TOOLS_CACHE is not None:
        return dict(_TOOLS_CACHE)
    
    logger.info("开始生成气象计算工具...")
    
    tools = {}
    
    # 1. 添加手动定义的工具
    for func_name, metadata in METPY_FUNCTIONS_MANUAL.items():
        try:
            tool = _make_tool_from_name(func_name)
            # 为工具添加元数据
            for key, value in metadata.items():
                setattr(tool, key, value)
            tools[func_name] = tool
        except Exception as e:
            logger.error(f"生成工具 {func_name} 失败: {e}")
    
    # 2. 从metpy.calc模块中提取所有公共函数
    mpcalc = _get_mpcalc()
    
    # 批量获取所有公共可调用对象的元数据
    public_callables = []
    for name in dir(mpcalc):
        if _is_public_callable(mpcalc, name):
            public_callables.append(name)
    
    # 批量创建工具
    for name in public_callables:
        if name in tools:
            continue  # 跳过已手动定义的工具
        
        try:
            # 获取函数签名
            func = getattr(mpcalc, name)
            sig = inspect.signature(func)
            
            # 生成默认元数据
            metadata = {
                "description": func.__doc__ or f"计算 {name}",
                "required_params": [param.name for param in sig.parameters.values() 
                                  if param.default == inspect.Parameter.empty],
                "example": f"{name}()",
                "group": "未分类"
            }
            
            # 创建工具
            tool = _make_tool_from_name(name)
            for key, value in metadata.items():
                setattr(tool, key, value)
            tools[name] = tool
        except Exception as e:
            logger.error(f"生成工具 {name} 失败: {e}")
    
    # 缓存结果
    _TOOLS_CACHE = tools.copy()
    logger.info(f"工具生成完成，共生成 {len(tools)} 个工具")
    
    return dict(tools)

def _is_public_callable(module, name):
    """判断是否为公开可调用对象"""
    if name.startswith('_'):
        return False
    obj = getattr(module, name)
    return callable(obj) and not inspect.ismodule(obj) and not isinstance(obj, type)

def _make_tool_from_name(name: str) -> Callable:
    """
    根据名称创建工具函数
    
    Args:
        name: 工具名称
        
    Returns:
        Callable: 工具函数
    """
    try:
        from metpy_wrapper import compute_with_metpy
    except ImportError as e:
        logger.error(f"无法导入compute_with_metpy: {e}")
        
        # 如果导入失败，返回一个错误处理的工具函数
        def error_tool(records: list, extra_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
            return {"status": "fail", "message": f"无法导入metpy_wrapper模块: {e}"}
        
        error_tool.__name__ = name
        return error_tool
    
    def tool(records: list, extra_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        工具函数
        
        Args:
            records: 数据记录列表
            extra_kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 计算结果
        """
        if extra_kwargs is None:
            extra_kwargs = {}
        try:
            return compute_with_metpy(records, name, extra_kwargs=extra_kwargs)
        except Exception as e:
            logger.error(f"工具 {name} 调用失败: {e}")
            return {"status": "fail", "message": str(e)}
    
    tool.__name__ = name
    return tool