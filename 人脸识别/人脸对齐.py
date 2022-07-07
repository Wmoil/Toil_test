#!/usr/bin/env python
# author:AnFany
# datetime:2020/12/28 10:03

# 在人脸识别的基础上将输出的人脸进行对齐校正

from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
from skimage import transform as trans


# 输入图片路径
IN_Figure_Path = r'D:\Object_Detection\mtcnn_in'

# 输出图片路径
OUT_Figure_Path = r'D:\Object_Detection\mtcnn_out'

# 正脸按照112*112的尺寸，此时左眼、右眼、鼻子、嘴巴左边、右边的坐标为
KeyPoints = np.array([(38.2946, 51.6963),
                      (73.5318, 51.6963),
                      (56.0252, 71.7366),
                      (41.5493, 92.3655),
                      (70.7299, 92.3655)])

# 进行人脸检测的函数
def mtcnn_face_detection_align(inp=IN_Figure_Path, outp=OUT_Figure_Path,
                               boxcolor=(220, 20, 20), boxw=3, dst_data=KeyPoints, figsize=(112, 112)):
    """
    :param inp: 需要进行人脸检测的图片的路径
    :param outp: 检测后的输出路径
    :param boxcolor: (220, 20, 20) 对应rgb，检测框的颜色
    :param boxw: 检测框的宽度
    :param dst_data: 112*112 的正脸中关键点的坐标
    :param figsize: 图片尺寸112*112
    :return: 人脸对齐后的图片
    """
    # 人脸检测模型
    detector = MTCNN()
    for fig in os.listdir(inp):
        # 因为cv2读取的图片数据是bgr的，模型的输入是rgb的
        img_data = cv2.cvtColor(cv2.imread(r'%s/%s' %(inp, fig)), cv2.COLOR_BGR2RGB)
        # 检测结果
        result_detection = detector.detect_faces(img_data)
        # 判断是否检测出人脸
        name = ''.join(fig.split('.')[:-1])
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
                # 保存截取到的人脸
                # cv2.imwrite(r'%s/%s_%s.png' % (outp, name, face_count), face_detection_data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                # 获得五个关键点的坐标
                src_data = []
                for k in ['left_eye','right_eye','nose','mouth_left','mouth_right']:
                    src_data.append(face['keypoints'][k])

                # 计算放射矩阵
                tform = trans.SimilarityTransform()
                res = tform.estimate(np.array(src_data), dst_data)
                M = tform.params
                # 应用仿射矩阵进行人脸对齐
                align_face_data =  cv2.warpAffine(img_data.copy(), M[:2,:], figsize,
                                                  flags=cv2.INTER_CUBIC, borderValue=(255,255,255))
                align_face_data = cv2.cvtColor(align_face_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(r'%s/%s_%s_align.png' % (outp, name, face_count), align_face_data,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                face_count += 1

            for fb in face_set:
                # 加上人脸框
                face_data = cv2.rectangle(img_data, tuple(fb[0]), tuple(fb[1]), boxcolor, boxw)
            # 数据通道在变回去
            face_data = cv2.cvtColor(face_data, cv2.COLOR_RGB2BGR)
            # 保存带有人脸框的图片
            # cv2.imwrite(r'%s/%s.png' % (outp, name), face_data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return print('人脸检测完毕')

mtcnn_face_detection_align()