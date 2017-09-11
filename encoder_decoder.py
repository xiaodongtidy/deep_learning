#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np


def num_encoder(text):
    """
    将4位数字标签文本转化成向量
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


def num_decoder(vector, vector_type="label"):
    """
    将向量转化为4位数字字符
    :param vector:          numpy array格式的向量,长度必须为40
    :param vector_type:     输入向量的类型,"label"为标签,"pred"为模型预测输出
    :return:                转化后的数字字符,数据类型为string
    """
    if len(vector) != 40:
        raise KeyError("The input vector's length is not equal to 40")
    text = []
    if vector_type == "label":
        # 返回非零元素的索引
        text_list = vector.nonzero()[0]
        for index, value in enumerate(text_list):
            # 根据索引值解出验证码的值
            text.append(str(int(value) - index * 10))
    elif vector_type == "pred":
        # 将输入拆分成四个列表,分别得出最大值所在的索引,即为模型的预测输出
        for i in range(4):
            pred_list = vector[i * 10:(i + 1) * 10].tolist()
            text.append(str(pred_list.index(max(pred_list))))
    else:
        raise KeyError("The vector_type must be label or pred!")

    return "".join(text)
