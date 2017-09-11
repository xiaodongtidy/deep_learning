#!/usr/bin/env python
# -*- coding:utf-8 -*-
from reader import get_code_decode
from keras.models import load_model
from encoder_decoder import num_decoder


# 读取模型
model = load_model("logs/cnn_rmsprop_model.h5")

# 读取模型权重
model.load_weights("logs/cnn_rmsprop_weights.h5")

# 测试模型
error = 0
for i in range(1000):
    img, label = get_code_decode()
    pred = num_decoder(model.predict(img, batch_size=1, verbose=1)[0], vector_type="pred")
    if pred != label:
        print("预测", pred)
        print("实际", label)
        error += 1
print(1 - error / 1000.)
