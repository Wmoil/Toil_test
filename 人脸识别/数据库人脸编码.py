#!/usr/bin/env python
# author:AnFany
# datetime:2020/12/30 14:38

# 对数据库中的人脸进行编码并存为jason文件
# facenet是基于tensorflow1.x版本的

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import facenet.src.facenet as facenet
import cv2
import numpy as np
import os
import json

# 人脸图片文件夹
FaceFigPath = r'D:\Object_Detection\facenet_database\face_fig'
# 存储人脸图片编码json字符串的路经
JsonPath = r'D:\Object_Detection\facenet_database'

# 尺寸配置
ImageSize = [140, 169]  # 需要适合预训练模型，本文模型适合[139-170]。
# 下载好的预训练模型
PreTrainModelDir = r'D:\Object_Detection\model\20180408-102900\facenet_model\20180408-102900.pb'

# 人脸数据编码函数
def get_face_db_code(inp=FaceFigPath, imlist=ImageSize, md=PreTrainModelDir, jp=JsonPath):
    face_code_dict = {}
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            # 下载预训练模型参数
            facenet.load_model(md)
            # 根据名称获取相应的张量
            image_input_tensor = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings_tensor = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_tensor = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # 开始遍历每个人的
            for per_fold in os.listdir(inp):
                face_code_dict[per_fold] = {}
                per_fold_path = os.path.join(inp, per_fold)
                for per_fig in os.listdir(per_fold_path):
                    per_fig_path = os.path.join(per_fold_path, per_fig)
                    # 读取图片数据，并转换通道
                    image_data = cv2.cvtColor(cv2.imread(per_fig_path), cv2.COLOR_BGR2RGB)
                    # 正脸尺寸编号
                    right_face_sign = per_fig.split('_')[0]
                    # 不同的尺寸
                    for fs in imlist:
                        if fs not in face_code_dict[per_fold]:
                            face_code_dict[per_fold][fs] = {}
                        image_data2 = cv2.resize(image_data, (fs, fs), interpolation=cv2.INTER_CUBIC)
                        # 图片数据预处理
                        image_data2 = facenet.prewhiten(image_data2)
                        # 数据增加维度
                        image_data2 = image_data2.reshape(-1,fs,fs,3)
                        # 编码数据，长度为512
                        embeddings_data = sess.run(embeddings_tensor,
                                                   feed_dict={image_input_tensor: image_data2,
                                                              phase_train_tensor: False})[0].tolist()
                        if right_face_sign in face_code_dict[per_fold][fs]:
                            face_code_dict[per_fold][fs][right_face_sign].append(embeddings_data)
                        else:
                            face_code_dict[per_fold][fs][right_face_sign] = [embeddings_data]

    # 将字典变为json
    # jsonstr = json.dumps(face_code_dict)
    # 存储
    with open(r'%s/em_face_json.json' % jp, 'w') as j:
        # j.write(jsonstr)
        json.dump(face_code_dict, j)

    return print('人脸图片编码json保存完毕')

get_face_db_code()