#!/usr/bin/env python
# -*- coding:utf-8 -*-
from Lin_deep_learning.reader import read_jpg_decode
from keras.models import load_model


# 获取数据
img_test, label_test = read_jpg_decode("/tidy_id_code/small/")
img_test = img_test.astype("float32")

# 读取模型
model = load_model("logs/cnn_rmsprop_model_2.h5")

# 读取模型权重
model.load_weights("logs/cnn_rmsprop_weights_2.h5")

# 测试模型
test_cost = model.predict(img_test, batch_size=2, verbose=1)
print("Test cost is ", test_cost)
