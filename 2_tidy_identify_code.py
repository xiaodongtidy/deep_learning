#!/usr/bin/env python
# -*- coding:utf-8 -*-
from reader import read_jpg_decode
# from Lin_deep_learning.reader import read_jpg_decode
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from keras.models import Model
# from keras.optimizers import SGD
# from keras import regularizers
from keras.callbacks import TensorBoard
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import Xception
# import tensorflow as tf


# 获取数据
img_train, label_train = read_jpg_decode("/tidy_id_code/train/")
img_test, label_test = read_jpg_decode("/tidy_id_code/test/")
img_validate, label_validate = read_jpg_decode("/tidy_id_code/validate/")
img_train = img_train.astype("float32")
img_test = img_test.astype("float32")
img_validate = img_validate.astype("float32")


# 建立模型
input_img = Input(shape=(60, 120, 1))
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(40, activation="sigmoid")(x)
model = Model(inputs=input_img, outputs=output)

# 编译模型
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 开始训练
model.fit(
    img_train,
    label_train,
    batch_size=100,
    epochs=50,
    validation_data=(img_validate, label_validate),
    callbacks=[TensorBoard(log_dir="logs", write_graph=False, write_images=True)]
)

test_cost = model.evaluate(img_test, label_test, batch_size=100)
print("Test cost is ", test_cost)

# 保存模型结构
model.save("logs/cnn_rmsprop_model.h5")

# 保存模型权重
model.save_weights("logs/cnn_rmsprop_weights.h5")
