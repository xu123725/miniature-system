"""
MetPy 使用包装器（修复并去除所有对未定义 FIELD_MAPPING 的引用）

说明：
- 修复了在逐条回退路径中意外引用未定义符号的问题（已全部改为使用 colmap）。
- 该模块提供 compute_with_metpy(...)，用于在你的 Agent/工具生成器中以稳健的方式调用 metpy.calc 的函数。
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import inspect

from config import logger
import metpy.calc as mpcalc  # type: ignore
from metpy.units import units  # type: ignore

# MetPy is imported at module load; no deferred import

# 示例 schema：可以按需扩展或从外部加载
DEFAULT_SCHEMA: Dict[str, Dict[str, Any]] = {
    "temperature": {"aliases": ["气温", "温度", "气温(℃)"], "unit": "degC", "required": True, "range": (-90, 60)},
    "relative_humidity": {"aliases": ["相对湿度", "湿度", "相对湿度(%)"], "unit": "percent", "required": True, "range": (0, 100)},
    "pressure": {"aliases": ["气压", "气压(hPa)"], "unit": "hPa", "required": False, "range": (100, 2000)},
    "dewpoint": {"aliases": ["露点", "露点温度(℃)"], "unit": "degC", "required": False, "range": (-90, 60)},
    "wind_speed": {"aliases": ["风速", "风速(m/s)"], "unit": "m/s", "required": False, "range": (0, 100)},
    "wind_direction": {"aliases": ["风向", "风向(°)"], "unit": "degree", "required": False, "range": (0, 360)},
    "specific_humidity": {"aliases": ["比湿", "比湿(g/kg)"], "unit": "kg/kg", "required": False, "range": (0, 0.05)},
    "mixing_ratio": {"aliases": ["混合比", "混合比(g/kg)"], "unit": "kg/kg", "required": False, "range": (0, 0.1)},
    "height": {"aliases": ["高度", "海拔", "高度(m)"], "unit": "meter", "required": False, "range": (0, 20000)},
    "geopotential_height": {"aliases": ["位势高度", "Φz"], "unit": "meter", "required": False, "range": (0, 20000)},
}

def build_column_map(columns: List[str], schema: Dict[str, Dict[str, Any]] = DEFAULT_SCHEMA) -> Dict[str, str]:
    """根据列名匹配 schema 中的 canonical 字段名 -> 实际列名"""
    colmap: Dict[str, str] = {}
    lowcols = {c.lower(): c for c in columns}
    for canonical, meta in schema.items():
        aliases = meta.get("aliases", [])
        found = None
        for alias in aliases:
            a = alias.lower()
            for col_lower, col in lowcols.items():
                if a == col_lower or a in col_lower or col_lower in a:
                    found = col
                    break
            if found:
                break
        if found:
            colmap[canonical] = found
    return colmap

def normalize_units(data_dicts: List[Dict[str, Any]], colmap: Dict[str, str],
                    schema: Dict[str, Dict[str, Any]] = DEFAULT_SCHEMA) -> Tuple[Dict[str, Any], List[str]]:
    """
    把数据列转换为 numpy arrays，并返回预期单位字符串信息。
    返回 (mapped_arrays, warnings). mapped_arrays: {canonical: (arr, unit_str)}
    """
    mapped: Dict[str, Tuple[np.ndarray, Optional[str]]] = {}
    warnings: List[str] = []
    n = len(data_dicts)
    for canonical, col in colmap.items():
        vals = []
        for rec in data_dicts:
            v = rec.get(col, None)
            try:
                vals.append(float(v) if v is not None else np.nan)
            except Exception:
                vals.append(np.nan)
        arr = np.array(vals, dtype="float64")
        unit = schema.get(canonical, {}).get("unit", None)
        mapped[canonical] = (arr, unit)
        rng = schema.get(canonical, {}).get("range", None)
        if rng is not None:
            mask_bad = (arr < rng[0]) | (arr > rng[1])
            bad_count = int(np.nansum(mask_bad.astype(float)))
            if bad_count > 0 and (bad_count / max(1, n)) > 0.01:  # 超过1%警告
                warnings.append(f"{canonical}: 大约 {bad_count} 条记录超出合理范围 {rng}")
    return mapped, warnings

def prepare_quantities(mapped_arrays: Dict[str, Tuple[np.ndarray, Optional[str]]], units_module) -> Dict[str, Any]:
    """
    将 (numpy arr, unit_str) 转为 pint Quantity arrays（使用 metpy.units）。
    返回 {param_name: quantity_array}
    """
    qtys: Dict[str, Any] = {}
    for k, (arr, unit_str) in mapped_arrays.items():
        if unit_str is None:
            qtys[k] = arr * units_module.dimensionless
        else:
            try:
                if hasattr(units_module, unit_str):
                    u = getattr(units_module, unit_str)
                else:
                    u = units_module(unit_str)
            except Exception:
                try:
                    mapping = {'degC': units_module.degC, 'm/s': units_module('m/s'), 'hPa': units_module.hPa,
                               'kg/kg': units_module('kg/kg'), 'percent': units_module.percent, 'degree': units_module.degree}
                    u = mapping.get(unit_str, units_module.dimensionless)
                except Exception:
                    u = units_module.dimensionless
            qtys[k] = arr * u
    return qtys

def _fill_missing_params(params: Dict[str, Any]):
    """
    辅助函数：根据已有参数计算缺失的气象参数。
    适用于向量化（pint.Quantity arrays）和标量（float/Quantity）输入。
    原地修改 params 字典。
    """
    # 露点温度计算
    if 'dewpoint' not in params:
        if ('temperature' in params) and ('relative_humidity' in params):
            try:
                params['dewpoint'] = mpcalc.dewpoint_from_relative_humidity(params['temperature'], params['relative_humidity'])
            except Exception:
                pass
        elif ('temperature' in params) and ('specific_humidity' in params) and ('pressure' in params):
            try:
                mr = mpcalc.mixing_ratio_from_specific_humidity(params['specific_humidity'])
                rh = mpcalc.relative_humidity_from_mixing_ratio(params['pressure'], params['temperature'], mr)
                params['dewpoint'] = mpcalc.dewpoint_from_relative_humidity(params['temperature'], rh)
            except Exception:
                pass

    # 混合比计算
    if 'mixing_ratio' not in params:
        if 'specific_humidity' in params:
            try:
                params['mixing_ratio'] = mpcalc.mixing_ratio_from_specific_humidity(params['specific_humidity'])
            except Exception:
                pass
        elif ('pressure' in params) and ('temperature' in params) and ('relative_humidity' in params):
            try:
                params['mixing_ratio'] = mpcalc.mixing_ratio_from_relative_humidity(params['pressure'], params['temperature'], params['relative_humidity'])
            except Exception:
                pass
        elif ('pressure' in params) and ('dewpoint' in params):
            try:
                e = mpcalc.saturation_vapor_pressure(params['dewpoint'])
                params['mixing_ratio'] = mpcalc.mixing_ratio(e, params['pressure'])
            except Exception:
                pass

    # 气压计算
    if 'pressure' not in params:
        if 'height' in params:
            try:
                params['pressure'] = mpcalc.height_to_pressure_std(params['height'])
            except Exception:
                pass
        elif 'geopotential_height' in params:
            try:
                h_geom = mpcalc.geopotential_to_height(params['geopotential_height'])
                params['pressure'] = mpcalc.height_to_pressure_std(h_geom)
            except Exception:
                pass

    # 气块配置计算
    if 'parcel_profile' not in params and ('pressure' in params) and ('temperature' in params) and ('dewpoint' in params):
        try:
            params['parcel_profile'] = mpcalc.parcel_profile(params['pressure'], params['temperature'], params['dewpoint'])
        except Exception:
            pass

def call_metpy_function_vectorized(func_name: str, quantities: Dict[str, Any], extra_kwargs: Dict[str, Any] = None):
    """
    向量化调用 MetPy 函数（mpcalc.<func_name>），quantities: dict of pint.Quantity arrays.
    返回 (result_quantity, error_message)
    """
    if not hasattr(mpcalc, func_name):
        return None, f"MetPy 不包含函数 {func_name}"

    func = getattr(mpcalc, func_name)
    sig = inspect.signature(func)
    
    # 使用辅助函数填充缺失参数
    _fill_missing_params(quantities)
    
    call_args: Dict[str, Any] = {}
    for p in sig.parameters.keys():
        if p in quantities:
            call_args[p] = quantities[p]
        elif extra_kwargs and p in extra_kwargs:
            call_args[p] = extra_kwargs[p]
    try:
        res = func(**call_args)
        return res, None
    except Exception as e:
        logger.exception("向量化调用 MetPy %s 失败", func_name)
        return None, str(e)

def postprocess_result_to_records(result, records: List[Dict[str, Any]], out_col: str, units_module) -> List[Dict[str, Any]]:
    """
    将 MetPy 返回的结果（可能是 Quantity array 或标量）写回 records 列表为 out_col（数值化并做单位友好转换）。
    """
    out_records: List[Dict[str, Any]] = []
    try:
        if hasattr(result, "magnitude"):
            mag = result.magnitude
            try:
                if hasattr(result, "units") and str(result.units).lower().startswith("kelvin"):
                    mag = result.to(units_module.degC).magnitude
            except Exception:
                pass
        else:
            mag = np.asarray(result)
    except Exception:
        mag = np.asarray(result)

    n = len(records)
    try:
        if np.isscalar(mag):
            arr = np.repeat(float(mag), n)
        else:
            arr = np.asarray(mag)
            if arr.shape[0] != n:
                if arr.size == 1:
                    arr = np.repeat(float(arr), n)
                else:
                    new = np.full(n, np.nan, dtype=float)
                    L = min(n, arr.size)
                    new[:L] = arr[:L]
                    arr = new
    except Exception:
        arr = np.full(n, np.nan, dtype=float)

    for i, rec in enumerate(records):
        rec2 = dict(rec)
        try:
            val = arr[i]
            rec2[out_col] = None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
        except Exception:
            rec2[out_col] = None
        out_records.append(rec2)
    return out_records

def compute_with_metpy(records: List[Dict[str, Any]], func_name: str, schema: Dict[str, Dict[str, Any]] = DEFAULT_SCHEMA,
                       out_col: Optional[str] = None, extra_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    从 records（list[dict]）执行 MetPy 计算的完整流程：
      1. 构建列映射（根据 schema.aliases）
      2. 归一化单位（生成 quantity arrays）
      3. 向量化调用 MetPy（失败则回退逐条计算）
      4. 写回结果并返回 stats
    返回 dict: {status, message, processed, stats, result_field}
    """
    if not records:
        return {"status": "fail", "message": "数据为空", "processed": []}

    cols = list(records[0].keys())
    colmap = build_column_map(cols, schema)
    mapped, warnings = normalize_units(records, colmap, schema)
    if warnings:
        logger.warning("数据范围警告：%s", warnings)

    quantities = prepare_quantities(mapped, units)
    res, err = call_metpy_function_vectorized(func_name, quantities, extra_kwargs=extra_kwargs)
    if err:
        logger.warning("向量化调用失败 (%s)，尝试逐条回退：%s", func_name, err)
        processed: List[Dict[str, Any]] = []
        result_vals: List[float] = []
        func = getattr(mpcalc, func_name)
        for rec in records:
            try:
                kwargs: Dict[str, Any] = {}
                # 1) 优先直接使用记录中英文参数名
                for param in inspect.signature(func).parameters:
                    if param in rec:
                        kwargs[param] = rec[param]
                        continue
                    # 2) 如 colmap 中有匹配（canonical->实际列名），使用之
                    if param in colmap:
                        colname = colmap[param]
                        if colname in rec:
                            kwargs[param] = rec[colname]
                            continue
                    # 3) 最后尝试 extra_kwargs
                    if extra_kwargs and param in extra_kwargs:
                        kwargs[param] = extra_kwargs[param]
                # 给标量附加单位
                for k, v in list(kwargs.items()):
                    if isinstance(v, (int, float)):
                        unit = schema.get(k, {}).get("unit")
                        if unit:
                            try:
                                u = getattr(units, unit) if hasattr(units, unit) else units(unit)
                            except Exception:
                                u = units.dimensionless
                            kwargs[k] = v * u
                
                # 使用辅助函数填充缺失参数
                _fill_missing_params(kwargs)
                
                out = func(**kwargs)
                if hasattr(out, "magnitude"):
                    try:
                        if hasattr(out, "units") and str(out.units).lower().startswith("kelvin"):
                            out_v = out.to(units.degC).magnitude
                        else:
                            out_v = out.magnitude
                    except Exception:
                        out_v = out.magnitude
                else:
                    out_v = out
                try:
                    out_v = float(out_v)
                    rec_copy = dict(rec)
                    rec_copy[f"{func_name}_result"] = round(out_v, 4)
                    result_vals.append(out_v)
                except Exception:
                    rec_copy = dict(rec)
                    rec_copy[f"{func_name}_result"] = str(out_v)
                processed.append(rec_copy)
            except Exception as e:
                logger.debug("逐条调用失败：%s", e, exc_info=True)
                rec_copy = dict(rec)
                rec_copy[f"{func_name}_error"] = str(e)
                processed.append(rec_copy)
        stats: Dict[str, Any] = {}
        if result_vals:
            stats = {"mean": sum(result_vals)/len(result_vals), "count": len(result_vals)}
        return {"status": "partial", "message": f"向量化失败，已回退到逐条计算（可能部分失败）：{err}", "processed": processed, "stats": stats}

    if func_name == 'cape_cin' and isinstance(res, tuple) and len(res) >= 2:
        processed = postprocess_result_to_records(res[0], records, 'cape_result', units)
        processed = postprocess_result_to_records(res[1], processed, 'cin_result', units)
        vals = [rec['cape_result'] for rec in processed if rec.get('cape_result') is not None]
        stats: Dict[str, Any] = {}
        if vals:
            stats = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "count": len(vals)}
        return {"status": "success", "message": f"{func_name} 计算完成", "processed": processed, "stats": stats, "result_field": 'cape_result'}
    out_col_name = out_col or f"{func_name}_result"
    processed = postprocess_result_to_records(res, records, out_col_name, units)
    vals = [rec[out_col_name] for rec in processed if rec.get(out_col_name) is not None]
    stats: Dict[str, Any] = {}
    if vals:
        stats = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "count": len(vals)}
    return {"status": "success", "message": f"{func_name} 计算完成", "processed": processed, "stats": stats, "result_field": out_col_name}
