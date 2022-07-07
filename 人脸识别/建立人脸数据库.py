#!/usr/bin/env python
# author:AnFany
# datetime:2020/12/30 13:44


# 根据人脸数据库原始图片存储的每个人的照片，建立人脸图片数据
from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
from skimage import transform as trans
import shutil

# 人脸数据库路径
DataBase_Figure_Path = r'D:\Object_Detection\facenet_database'
# 存放每个人的原始图片文件夹
Origin_Name = 'origin_fig'

# 不同尺寸下人脸左眼、右眼、鼻子、左嘴角、右嘴角的坐标的对应关系
Size_KeyPoints = {'1': [(112, 112), np.array([(38.2946, 51.6963),
                                              (73.5318, 51.6963),
                                              (56.0252, 71.7366),
                                              (41.5493, 92.3655),
                                              (70.7299, 92.3655)])],
                  '2': [(96, 96), np.array([(30.2946, 43.6963),
                                            (65.5318, 43.6963),
                                            (48.0252, 63.7366),
                                            (33.5493, 84.3655),
                                            (62.7299, 84.3655)])]}

# 建立每个人的人脸图片
def build_face_person_database(inp=DataBase_Figure_Path, oname=Origin_Name, dst_data=Size_KeyPoints):
    # 首先建立人脸文件夹
    face_path = os.path.join(inp, 'face_fig')
    if 'face_fig' in os.listdir(inp):
        shutil.rmtree(face_path)
        os.mkdir(face_path)
    else:
        os.mkdir(face_path)

    # 人脸检测模型
    detector = MTCNN()

    # 开始遍历原始图片
    originfigpath = os.path.join(inp, oname)
    # 遍历每个人的文件夹
    for fol in os.listdir(originfigpath):
        # 新建文件夹
        new_face_path = os.path.join(face_path, fol)
        if not os.path.exists(new_face_path):
            os.mkdir(new_face_path)
        # 遍历每个图片
        person_file = os.path.join(originfigpath, fol)
        for per in os.listdir(person_file):
            fig_path = os.path.join(person_file, per)
            # 人脸检测模型
            # 因为cv2读取的图片数据是bgr的，模型的输入是rgb的
            img_data = cv2.cvtColor(cv2.imread(fig_path), cv2.COLOR_BGR2RGB)
            # 检测结果
            result_detection = detector.detect_faces(img_data)
            # 判断是否检测出人脸
            name = ''.join(per.split('.')[:-1])
            if result_detection:
                face_count = 1 # 人脸数量
                face_set = []
                for face in result_detection:
                    # 获取框的像素坐标
                    minx, miny, width, height = face['box']
                    maxx, maxy = minx+width, miny + height
                    # 保存人脸框
                    face_set.append([[minx, miny], [maxx, maxy]])
                    # 将检测到的人脸图片截取下来
                    face_detection_data = img_data[miny:maxy,minx:maxx, :]
                    face_detection_data = cv2.cvtColor(face_detection_data, cv2.COLOR_RGB2BGR)

                    # 获得五个关键点的坐标
                    src_data = []
                    for k in ['left_eye','right_eye','nose','mouth_left','mouth_right']:
                        src_data.append(face['keypoints'][k])

                    # 计算放射矩阵
                    tform = trans.SimilarityTransform()
                    # 遍历不同的正脸尺寸
                    for d in dst_data:
                        fsize, kdata = dst_data[d]
                        res = tform.estimate(np.array(src_data), kdata)
                        M = tform.params
                        # 应用仿射矩阵进行人脸对齐
                        align_face_data =  cv2.warpAffine(img_data.copy(), M[:2,:], fsize,
                                                          flags=cv2.INTER_CUBIC, borderValue=(255,255,255))
                        # 转化通道
                        align_face_data = cv2.cvtColor(align_face_data, cv2.COLOR_RGB2BGR)
                        # 保存图片
                        cv2.imwrite(r'%s/%s_%s_%s_%s.png' % (new_face_path, d, fol, face_count,name), align_face_data,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    face_count += 1

    return print('人脸数据库生成完毕')

build_face_person_database()