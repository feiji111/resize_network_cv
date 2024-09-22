import os
import shutil

# 源目录和目标目录
source_directory = '/data/chengqian/cars_classfication/train/cars'
target_directory = '/data/chengqian/cars_classfication/stanford_car'

# 确保目标目录存在
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# 遍历源目录中的所有文件
for filename in os.listdir(source_directory):
    # 检查文件名是否包含 "_jpg.rf"
    if '_jpg.rf' in filename:
        # 构建完整的文件路径
        source_file = os.path.join(source_directory, filename)
        target_file = os.path.join(target_directory, filename)
        
        # 移动文件
        shutil.move(source_file, target_file)
        print(f'Moved {filename} to {target_directory}')

print("All files have been moved.")