#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from PIL import Image


# 制作二进制数据
# def change_ont_hot(int_value):
#     lis = []
#     for i in str(int_value):
#         i = int(i)
#         b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         b[i] = 1
#         lis.append(b)
#     return np.array(lis)

path = os.getcwd() + "/tidy_id_code/train/"
writer = tf.python_io.TFRecordWriter("my_dataset/train.tfrecords")
for img_name in os.listdir(path):
    img_path = path + img_name
    int_values = img_name.replace(".jpg", "")
    img = Image.open(img_path)
    img = img.resize((64, 64))
    # 将图片文件转换成比特流
    img_raw = img.tobytes()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(int_values)])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }
        )
    )
    writer.write(example.SerializeToString())
writer.close()
