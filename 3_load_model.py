#!/usr/bin/env python
# -*- coding:utf-8 -*-
from reader import read_jpg_decode
from keras.models import load_model
from encoder_decoder import num_decoder


# 获取数据
img_test, label_test = read_jpg_decode("/tidy_id_code/test/")
img_test = img_test.astype("float32")
label_test = label_test.astype("int16")

# 读取模型
model = load_model("logs/cnn_rmsprop_model_2.h5")

# 读取模型权重
model.load_weights("logs/cnn_rmsprop_weights_2.h5")

# 测试模型
pred_test = model.predict(img_test, batch_size=2, verbose=1)

result = zip(pred_test, label_test)
error = 0
for i, l in result:
    pred = num_decoder(i, vector_type="pred")
    label = num_decoder(l)
    if pred != label:
        print("预测: ", pred, i)
        print("实际: ", label, l)
        error += 1
print(error)
