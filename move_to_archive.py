"""
将原始文件移动到归档目录的脚本
"""

import os
import shutil

# 定义要移动的文件
files_to_move = [
    '3D Probability Map.py',
    '3class.py',
    'classifier2d.py',
    'classifier_3D_Boundary.py',
    'data_preview.py',
    'readme.txt'
]

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建归档目录（如果不存在）
archive_dir = os.path.join(current_dir, 'archive')
if not os.path.exists(archive_dir):
    os.makedirs(archive_dir)

# 移动文件
for filename in files_to_move:
    src_path = os.path.join(current_dir, filename)
    dst_path = os.path.join(archive_dir, filename)

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"已移动: {filename}")
    else:
        print(f"文件不存在，跳过: {filename}")

print("\n文件移动完成！")
