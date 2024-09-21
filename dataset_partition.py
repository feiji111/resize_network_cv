import os
import shutil
import random

# 统计 cars 文件夹中的文件数量
cars_dir = '/data/chengqian/cars_classfication/train/cars'
cars_files = os.listdir(cars_dir)
num_cars = len(cars_files)

# 统计 nocars 文件夹中的文件数量
nocars_dir = '/data/chengqian/cars_classfication/train/nocars'
nocars_files = os.listdir(nocars_dir)
num_nocars = len(nocars_files)

# 计算 10% 的文件数量
num_val_cars = int(num_cars * 0.1)
num_val_nocars = int(num_nocars * 0.1)

# 随机抽取 cars 文件夹中的文件
random.shuffle(cars_files)
val_cars_files = cars_files[:num_val_cars]

# 随机抽取 nocars 文件夹中的文件
random.shuffle(nocars_files)
val_nocars_files = nocars_files[:num_val_nocars]

val_dir = '/data/chengqian/cars_classfication/val/'

# 将抽取的文件移动到验证集目录
for file in val_cars_files:
    shutil.move(os.path.join(cars_dir, file), os.path.join(val_dir, 'cars', file))

for file in val_nocars_files:
    shutil.move(os.path.join(nocars_dir, file), os.path.join(val_dir, 'nocars', file))