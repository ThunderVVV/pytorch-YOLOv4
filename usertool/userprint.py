# -*- coding: utf-8 -*-
'''
@Time          : 2021/03/01 00:51
@Author        : ThunderVVV
@File          : userprint.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import os
import sys
import datetime

def debugPrint(info):
    now = datetime.datetime.now()
    print("{} {}[line:{}] INFO:".format(now.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
                                     os.path.basename(sys._getframe().f_back.f_code.co_filename), 
                                     sys._getframe().f_back.f_lineno), 
                                     end=" ")
    print(info)