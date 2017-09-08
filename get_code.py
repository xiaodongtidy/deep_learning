#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import requests


# 进度条百分比函数
def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s>%s]%d%%' % ("=" * num, " " * (100 - num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


# 抓取验证码
get_code_url = "https://h5dev.24tidy.com/action/verifyCode.php"
for s in range(1001):
    res = requests.get(get_code_url, stream=True)
    with open("C:/Users/nengwen.tan/Desktop/train/{0}.jpg".format(res.headers["code"]), "wb") as f:
        for i in res.iter_content(chunk_size=1024):
            if i:
                f.write(i)
    if s % 10 == 0:
        view_bar(int(s / 10), 100)
