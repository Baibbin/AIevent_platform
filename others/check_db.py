import sqlite3
import json

# 连接数据库
conn = sqlite3.connect('ai_detection_events.db')
cursor = conn.cursor()

# 查询所有事件（按时间倒序）
cursor.execute("SELECT * FROM detection_events ORDER BY event_time DESC LIMIT 20;")
rows = cursor.fetchall()

# 打印查询结果
print(f"共查询到 {len(rows)} 条最新事件：")
print("-" * 100)
for row in rows:
    print(f"ID: {row[0]}")
    print(f"事件类型: {row[1]}")
    print(f"事件描述: {row[2]}")
    print(f"时间: {row[3]}")
    print(f"监控源: {row[4]}")
    print(f"检测功能: {row[5]}")
    print(f"额外信息: {json.loads(row[6]) if row[6] else '{}'}")
    print("-" * 100)

conn.close()