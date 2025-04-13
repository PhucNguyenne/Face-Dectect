import sqlite3
import pandas as pd

# Đọc dữ liệu từ CSV
csv_file = "Test_IMG_predictions.csv"
df = pd.read_csv(csv_file)

# Kết nối đến cơ sở dữ liệu SQLite (file "predictions.db" sẽ được tạo nếu chưa tồn tại)
conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()

# Tạo bảng nếu chưa tồn tại
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        image_id TEXT PRIMARY KEY,
        age TEXT,
        gender TEXT,
        emotion TEXT
    )
''')

# Nhập dữ liệu từ DataFrame vào bảng (sử dụng INSERT OR REPLACE để cập nhật nếu đã có)
for index, row in df.iterrows():
    cursor.execute('''
        INSERT OR REPLACE INTO predictions (image_id, age, gender, emotion)
        VALUES (?, ?, ?, ?)
    ''', (row['image_id'], row['age'], row['gender'], row['emotion']))

# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()

print("Dữ liệu đã được nhập vào cơ sở dữ liệu SQLite thành công!")
