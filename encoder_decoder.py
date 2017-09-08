#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np


def num_encoder(text):
    """
    将数字标签文本转化成向量
    :param text:            数字字符
    :return:                转化后的numpy array,数据类型为int
    """
    if len(text) > 4:
        raise ValueError("验证码最长4位!")

    vector = np.zeros(40)

    for index, value in enumerate(text):
        idx = index * 10 + int(value)
        vector[idx] = 1

    return vector


def num_decoder(vector):
    """
    将向量转化为数字字符
    :param vector:          numpy array格式的向量
    :return:                转化后的数字字符,数据类型为string
    """
    text_list = vector.nonzero()[0]
    text = []
    for index, value in enumerate(text_list):
        text.append(str(int(value) - index * 10))
    return "".join(text)

# 测试
# for a in range(1000, 9999):
#     b = num_encoder(str(a))
#     c = num_decoder(b)
#     print(a)
#     print(c)
