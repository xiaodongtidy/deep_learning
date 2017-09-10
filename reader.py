#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import re
import tensorflow as tf
import numpy as np
from PIL import Image
from encoder_decoder import num_encoder


IMAGE_HEIGHT = 60
IMAGE_WIDTH = 120


def read_and_decode(filename, batch_size):
    """
    读取tfrecords格式的文件,并生成图像和标签张量
    :param filename:        tfrecords文件所在路径
    :param batch_size:      返回批次数目
    :return:                图像和标签张量
    """
    # 创建文件队列
    file_queue = tf.train.string_input_producer([filename])
    # 创建读取器对象
    reader = tf.TFRecordReader()
    # 从队列读入序列化的样本
    _, serialized_example = reader.read(file_queue)
    # 从序列化示例获得特性
    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.int64),
            "img_raw": tf.FixedLenFeature([], tf.string)
        }
    )
    img = tf.decode_raw(features["img_raw"], tf.uint8)
    img = tf.reshape(img, [64, 64, 1])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features["label"], tf.int32)
    image, labels = tf.train.shuffle_batch([img, label], batch_size=batch_size, num_threads=16,
                                           min_after_dequeue=1, capacity=1024)
    return image, tf.reshape(labels, [batch_size])

# 测试read_and_decode
# img_batch, label_batch = read_and_decode("my_dataset/train.tfrecords")
# print(img_batch, label_batch)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     threads = tf.train.start_queue_runners(sess=sess)
#     for i in range(15):
#         print(img_batch.shape, label_batch)
#         val, l = sess.run([img_batch, label_batch])
#         print(val.shape, l)


def read_jpg_decode(filename, batch_size=None, data_format="channels_last"):
    """
    读取图片,并生成图像和标签张量
    :param filename:        图片文件所在文件夹的路径,必须在本文件所在同一级的目录之内,图片文件必须都以图片标签为文件名
                            filename例:"/tidy_id_code/small/",注意文件夹前后必须加上‘/’
    :param batch_size:      读取的文件数目
    :param data_format:     字符串,为‘channels_first’或‘channels_last’,代表图像的通道维位置
                            默认为‘channels_last’,数据应该组织为(128, 128, 3)
    :return:                按batch_size拼接而成的图像和标签张量
    """
    if data_format != "channels_last" and data_format != "channels_first":
        raise KeyError("data_format must be one of the choice: 1: channels_first 2: channels_last")

    if batch_size is None:
        x_list = []
        y_list = []
        path = os.getcwd() + filename
        for img_name in os.listdir(path):
            re_label = re.compile(".\w+$")
            label = re_label.sub("", img_name)
            label_array = num_encoder(label)
            img_2 = Image.open(path + img_name)
            img_array = np.array(img_2)
            img_array = np.reshape(img_array, (1, 60, 120))
            if data_format == "channels_last":
                img_array = img_array.transpose((1, 2, 0))
            elif data_format == "channels_first":
                pass
            else:
                pass
            x_list.append(img_array)
            y_list.append(label_array)
        return np.array(x_list), np.array(y_list)


# 测试read_jpg_decode
# image_x, image_y = read_jpg_decode("/tidy_id_code/small/")
# print(image_x, image_x.shape)
# print(image_y, image_y.shape)
