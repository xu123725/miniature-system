import re
import os
import sqlite3
import json
import logging

from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 以下函数为未来功能预留：将气象数据导出到SQLite数据库
# 保留原因：计划在未来版本中支持将分析结果导出到SQLite格式
def export_to_sqlite(data: List[Dict[str, Any]], table: str, db_path: str) -> bool:
    """
    将数据导出到SQLite数据库
    :param data: 要导出的数据列表
    :param table: 目标表名
    :param db_path: 数据库文件路径
    :return: 导出是否成功
    """
    if not data:
        logger.info(f"没有数据可导出到SQLite表 {table}")
        try:
            # 创建空数据库文件
            with open(db_path, 'w') as f:
                f.write('')
            logger.info(f"已创建空数据库文件：{db_path}")
            return True
        except Exception as e:
            logger.error(f"创建空数据库文件失败: {e}")
            return False

    try:
        # 表名验证
        if not re.match(r'^[a-zA-Z0-9_]+$', table):
            logger.error(f"无效的表名: {table}")
            return False

        # 提取字段名
        fields = list(data[0].keys())
        logger.info(f"提取到字段: {fields}")

        # 连接数据库
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 删除旧表（如果存在）
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"已删除旧表 {table}（如果存在）")

            # 创建新表（所有字段都是TEXT类型）
            create_table_sql = f"CREATE TABLE {table} ({', '.join([f'"{field}" TEXT' for field in fields])})"
            cursor.execute(create_table_sql)
            logger.info(f"创建表SQL: {create_table_sql}")

            # 插入数据
            for i, record in enumerate(data):
                # 处理数据类型
                values = []
                for field in fields:
                    value = record.get(field)
                    if value is None:
                        values.append(None)
                    elif isinstance(value, (dict, list)):
                        # 将复杂类型转换为JSON字符串
                        values.append(json.dumps(value))
                    else:
                        # 其他类型转换为字符串
                        values.append(str(value))

                # 构建插入SQL
                placeholders = ', '.join(['?' for _ in values])
                insert_sql = f"INSERT INTO {table} VALUES ({placeholders})"
                cursor.execute(insert_sql, values)
                logger.info(f"插入SQL {i+1}: {insert_sql} 数据: {values}")

            # 提交事务
            conn.commit()
            logger.info(f"成功导出 {len(data)} 条记录到SQLite表 {table}，数据库路径：{db_path}")
            return True

    except sqlite3.Error as e:
        logger.error(f"SQLite错误: {e}")
        return False
    except Exception as e:
        logger.error(f"导出数据到SQLite失败: {e}")
        return False