#!/usr/bin/env python
# author:AnFany
# datetime:2020/12/30 16:51

# 计算向量之间的距离

import numpy as np

def compare_dis_em(em1, em2, mode='euler'):
    em1 = np.array(em1)
    em2 = np.array(em2)
    if mode == 'euler':
        # 欧式距离
        diff = np.subtract(em1, em2)
        dist = np.sum(np.square(diff))
        # dist = np.sum(np.square(diff), 1)
    elif mode == 'cos':
        # 基于余弦相似度的距离
        dot = np.sum(np.multiply(em1, em2), axis=1)
        norm = np.linalg.norm(em1, axis=1) * np.linalg.norm(em2, axis=1)
        sim = dot / norm
        dist = np.arccos(sim) / np.pi
    return dist