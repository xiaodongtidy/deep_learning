#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential


# 准备数据
x = np.linspace(-1, 1, 500)
np.random.shuffle(x)
y = 3 * x + np.random.randn(*x.shape) * 0.03 + 15

x_train, y_train = x[:400], y[:400]
x_test, y_test = x[400:], y[400:]


# 图形显示生成的数据
plt.scatter(x, y)
plt.show()


# 建立模型
model = Sequential()
model.add(Dense(1, input_shape=(1, )))


# 编译模型
model.compile(
    optimizer="sgd",
    loss="mse"
)
# model.fit(
#     x,
#     y,
#     batch_size=500,
#     epochs=1
# )
# print(model.get_weights())


# 训练
print("Training...")
for train in range(1001):
    train_cost = model.train_on_batch(x_train, y_train)
    if train % 50 == 0:
        print("After {0} trains, the cost is {1}".format(train, train_cost))


# 测试
print("Testing...")
test_cost = model.evaluate(x_test, y_test, batch_size=100)
print("Test cost is {:.5}".format(test_cost))
W, b = model.layers[0].get_weights()
print("weight: {}, bias: {}".format(W, b))


# 画出训练结果
y_pred = model.predict(x_test)
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred)
plt.show()
