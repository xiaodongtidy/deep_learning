#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import requests


def view_bar(num, total):
    """
    打印百分比进度条
    :param num:             当前进度值,0 <= num <= 100
    :param total:           最大进度值,默认100
    :return:
    """
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s>%s]%d%%' % ("=" * num, " " * (100 - num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


def save_code(num):
    """
    抓取验证码,并保存至指定目录
    :param num:             需要保存的验证码个数,如果有重复验证码会直接覆盖,最终生成的验证码个数可能和指定保存的数目不同
    :return:                None,屏幕输出进度
    """
    get_code_url = "https://h5dev.24tidy.com/action/verifyCode.php"
    for s in range(num + 1):
        res = requests.get(get_code_url, stream=True)
        with open("C:/Users/nengwen.tan/Desktop/train/{0}.jpg".format(res.headers["code"]), "wb") as f:
            for i in res.iter_content(chunk_size=1024):
                if i:
                    f.write(i)
        if s % 10 == 0:
            view_bar(int(s / 10), 100)


def get_code():
    get_code_url = "https://h5dev.24tidy.com/action/verifyCode.php"
    res = requests.get(get_code_url, stream=True)
    return res
