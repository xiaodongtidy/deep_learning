#!/usr/bin/env python
# -*- coding:utf-8 -*-
from keras.models import load_model
from reader import get_batch_code
import time


# 读取模型
model = load_model("logs/cnn_rmsprop_model.h5")

# 编译模型
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 开始训练
for i in range(1000):
    img, label = get_batch_code(batch_size=256)
    start = time.time()
    res = model.train_on_batch(img, label)
    end = time.time()
    print("Epochs {0}:".format(i))
    print("loss:", res[0], "accuracy:", res[1], "use %.2f" % (end - start))
