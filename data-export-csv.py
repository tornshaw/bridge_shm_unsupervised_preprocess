#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Doris数据导出工具
从Doris数据库查询数据，解析JSON字段并导出为CSV格式
"""

import pymysql
import json
import csv
import os
from datetime import datetime, timedelta


class DorisDataExporter:
    def __init__(self, host='localhost', port=9030, user='root', password='', database=''):
        """
        初始化Doris连接参数

        Args:
            host: Doris主机地址
            port: Doris端口
            user: 用户名
            password: 密码
            database: 数据库名
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise

    def disconnect(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("数据库连接已关闭")

    def execute_query(self, sql):
        """
        执行SQL查询

        Args:
            sql: SQL查询语句

        Returns:
            查询结果列表
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
                print(f"查询成功，返回 {len(results)} 条记录")
                return results
        except Exception as e:
            print(f"查询执行失败: {e}")
            raise

    def parse_json_data(self, data_rows):
        """
        解析JSON数据并整理为标准格式

        Args:
            data_rows: 从数据库查询的原始数据

        Returns:
            处理后的数据和字段列表
        """
        processed_data = []
        all_fields = set()

        for row in data_rows:
            data_time = row['data_time']
            datas_json = row['datas']

            try:
                # 解析JSON字符串
                if isinstance(datas_json, str):
                    datas_dict = json.loads(datas_json)
                else:
                    datas_dict = datas_json

                # 收集所有字段名
                all_fields.update(datas_dict.keys())

                # 保存处理后的数据
                processed_data.append({
                    'data_time': data_time,
                    'fields': datas_dict
                })

            except json.JSONDecodeError as e:
                print(f"JSON解析失败 for data_time {data_time}: {e}")
                # 如果JSON解析失败，仍然保存data_time，但fields为空
                processed_data.append({
                    'data_time': data_time,
                    'fields': {}
                })

        # 对字段进行排序：按字段名字符串排序（字段名为传感器实际ID）
        def sort_key(field_name):
            """自定义排序键：直接按字段名排序"""
            return field_name

        sorted_fields = sorted(all_fields, key=sort_key)

        return processed_data, sorted_fields

    def export_to_csv(self, processed_data, sorted_fields, output_path):
        """
        导出数据到CSV文件

        Args:
            processed_data: 处理后的数据
            sorted_fields: 排序后的字段列表
            output_path: 输出文件路径
        """
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # CSV表头：data_time + 排序后的字段名
        headers = ['data_time'] + sorted_fields

        try:
            # 如果目标文件已存在，先尝试删除（避免写入被锁住的旧文件）。
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except PermissionError:
                    # 如果文件被其他程序锁定，继续尝试打开覆盖它。
                    pass

            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # 写入表头
                writer.writerow(headers)

                # 写入数据
                for row in processed_data:
                    data_time = row['data_time']
                    fields = row['fields']

                    # 构建数据行
                    csv_row = [data_time]

                    # 为每个字段添加值，如果不存在则为空
                    for field in sorted_fields:
                        value = fields.get(field, '')
                        csv_row.append(value)

                    writer.writerow(csv_row)

            print(f"数据已成功导出到: {output_path}")
            print(f"总记录数: {len(processed_data)}")
            print(f"总字段数: {len(sorted_fields)}")

        except Exception as e:
            print(f"导出CSV失败: {e}")
            raise

    def export_data(self, sql, output_path):
        """主导出方法

        Args:
            sql: 查询SQL
            output_path: 输出CSV文件路径
        """
        try:
            # 连接数据库
            self.connect()

            # 执行查询
            raw_data = self.execute_query(sql)

            # 处理数据
            processed_data, sorted_fields = self.parse_json_data(raw_data)

            # 导出CSV
            self.export_to_csv(processed_data, sorted_fields, output_path)

        finally:
            # 确保连接关闭
            self.disconnect()

    def _build_query_for_object(self, object_id, start_date='2025-12-01', end_date='2026-03-19'):
        """根据 object_id 生成对应的查询 SQL。

        逻辑：
          1) 从 DIM_MON_POINT_INFO 找出该 object_id 的点位（point_code/point_id）和 MONITOR_TYPE
          2) 通过 DIM_MON_ITEM_INFO 找到对应的 ITEM_KEY
          3) 对每个 ITEM_KEY 拼接对应的 MON_BRIDGE_SP_{ITEM_KEY}_RT 表
          4) 从这些表中取出 data_time、point_code、point_field、mon_value
          5) 使用 MAP_AGG 将每个时间点按 point_code_point_field 聚合为 JSON
        """
        # 1) 从维表获取 point_id 和 monitor_type
        mapping_sql = f"""
        select p.ID as point_id, p.MONITOR_TYPE
        from CITYLL_DW_DIM.DIM_MON_POINT_INFO p
        where p.OBJECT_ID = '{object_id}'
          and p.TENANT_ID = 125
          and p.DEVICE_TYPE in ('DS','SG','HLG','INC','THS')
        """

        rows = self.execute_query(mapping_sql)

        point_ids = {r['point_id'] for r in rows if r.get('point_id') is not None}
        monitor_type_ids = {r['MONITOR_TYPE'] for r in rows if r.get('MONITOR_TYPE')}

        # 2) 从 DIM_MON_ITEM_INFO 中根据 MONITOR_TYPE (ID) 取 ITEM_KEY
        item_keys = set()
        if monitor_type_ids:
            # MONITOR_TYPE 在 DIM_MON_ITEM_INFO 中是 ID 字段，可能是字符串形式
            quoted_ids = ", ".join("'" + str(x).replace("'", "''") + "'" for x in monitor_type_ids)
            item_sql = f"""
            select ID, ITEM_KEY
            from CITYLL_DW_DIM.DIM_MON_ITEM_INFO
            where ID in ({quoted_ids})
            """
            item_rows = self.execute_query(item_sql)
            for r in item_rows:
                if r.get('ITEM_KEY'):
                    item_keys.add(r['ITEM_KEY'])

        # 验证 ITEM_KEY 是否对应的 DWD 表存在，避免拼表失败
        if item_keys:
            show_tables_sql = "SHOW TABLES LIKE 'MON_BRIDGE_SP_%_RT'"
            table_rows = self.execute_query(show_tables_sql)
            existing_keys = set()
            for r in table_rows:
                tbl = list(r.values())[0]
                if tbl.startswith('MON_BRIDGE_SP_') and tbl.endswith('_RT'):
                    existing_keys.add(tbl[len('MON_BRIDGE_SP_'):-len('_RT')])
            item_keys = {k for k in item_keys if k in existing_keys}

        if not item_keys:
            # 没有匹配的数据时返回空结果（不触发聚合错误）
            return "select 1 as data_time, '{}' as datas where 1=0"

        union_parts = []
        for item_key in sorted(item_keys):
            ids_clause = ','.join(str(int(i)) for i in sorted(point_ids)) if point_ids else 'NULL'
            union_parts.append(f"""
            select t.DATA_TIME as data_time,
                   t.POINT_ID as point_id,
                   p.POINT_CODE as point_code,
                   t.POINT_FIELD as point_field,
                   t.MON_VALUE as mon_value
            from CITYLL_DW_DWD.MON_BRIDGE_SP_{item_key}_RT t
            join CITYLL_DW_DIM.DIM_MON_POINT_INFO p on p.ID = t.POINT_ID
            where t.POINT_ID in ({ids_clause})
              and t.DATA_TIME >= '{start_date}'
              and t.DATA_TIME < '{end_date}'
            """)

        union_sql = "\nunion all\n".join(union_parts)

        final_sql = f"""
        select MINUTE_FLOOR(data_time,10) as data_time,
               MAP_AGG(concat(point_code,'_',point_field), mon_value) as datas
        from (
{union_sql}
        ) t
        group by MINUTE_FLOOR(data_time,10)
        order by MINUTE_FLOOR(data_time,10) asc
        """

        return final_sql

    def export_for_object(self, object_id, output_path):
        """基于 object_id 导出对应桥的数据到 CSV。"""
        try:
            self.connect()
            sql = self._build_query_for_object(object_id)
            raw_data = self.execute_query(sql)
            processed_data, sorted_fields = self.parse_json_data(raw_data)
            self.export_to_csv(processed_data, sorted_fields, output_path)
        finally:
            self.disconnect()


def main():
    """
    主函数 - 示例使用
    请根据实际情况修改数据库连接参数和SQL查询
    """

    # 数据库连接配置
    db_config = {
        'host': '10.172.121.139',  # 请替换为实际的Doris主机地址
        'port': 9030,               # Doris默认端口
        'user': 'root',    # 请替换为实际用户名
        'password': '8j2q8nWs0u7ZogoDzMxa', # 请替换为实际密码
        'database': 'CITYLL_DW_DWD'  # 请替换为实际数据库名
    }

    # 创建导出器实例
    exporter = DorisDataExporter(**db_config)

    # 输出目录（导出文件会存到 data/ 目录）
    output_dir = 'data'

    # 时间范围：固定为 2025-12-01 到 2026-03-18（日期格式 YYYYMMDD）
    start_time = '20251201'
    end_time = '20260318'

# 2001252877095403520	大岗沥大桥
# 2001256113965629440	新榄核大桥
# 2001259878131171328	灵岗二桥
# 2005573459857375232	沥心沙大桥
# 2001257803523555328	新浅海大桥
# 2005536136788705280	榄核桥
# 2001257326891237376	浅海大桥
# 2001259207633928192	灵岗一桥
# 2006281284376068096	沙仔二桥
# 2001258258362269696	子沙大桥
# 2005572553875128320	西樵水道特大桥
# 2001254361472172032	高新沙大桥
# 2005528715705974784	民生桥
# 2005545574480216064	蕉门水道特大桥
    object_id_name_pair_list = [
        # ('2001252877095403520', '大岗沥大桥'),
        # ('2001256113965629440', '新榄核大桥'),
        # ('2001259878131171328', '灵岗二桥'),
        # ('2005573459857375232', '沥心沙大桥'),
        # ('2001257803523555328', '新浅海大桥'),
        ('2005536136788705280', '榄核桥'),
        # ('2001257326891237376', '浅海大桥'),
        # ('2001259207633928192', '灵岗一桥'),
        # ('2006281284376068096', '沙仔二桥'),
        # ('2001258258362269696', '子沙大桥'),
        # ('2005572553875128320', '西樵水道特大桥'),
        # ('2001254361472172032', '高新沙大桥'),
        # ('2005528715705974784', '民生桥'),
        # ('2005545574480216064', '蕉门水道特大桥')
    ]

    for object_id, object_name in object_id_name_pair_list:
        safe_name = object_name.replace(' ', '_')
        output_file = f"{output_dir}/{safe_name}_{start_time}_{end_time}.csv"
        exporter.export_for_object(object_id, output_file)


if __name__ == '__main__':
    main()
