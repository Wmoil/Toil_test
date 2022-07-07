#!/usr/bin/env python
# author:AnFany
# datetime:2021/1/4 9:46


# 实现在线视频的人脸检测并标注

# 第一步骤：人脸检测和对齐
from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
from skimage import transform as trans
from PIL import Image, ImageDraw, ImageFont
import shutil

# 第二步骤：人脸编码
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import facenet.src.facenet as facenet
import json

# 第三步骤：人脸识别
# 计算距离
import computer_distance_em as cde

# 数据库人脸图片编码json字符串的路经
JsonPath = r'D:\Object_Detection\facenet_database\em_face_json.json'
# 尺寸配置
ImageSize = 140 # 最好和数据库存储的一样
# 正脸关键点
Size_KeyPoints = {'1': [(112, 112), np.array([(38.2946, 51.6963),
                                              (73.5318, 51.6963),
                                              (56.0252, 71.7366),
                                              (41.5493, 92.3655),
                                              (70.7299, 92.3655)])]}
# 下载好的预训练模型
PreTrainModelDir = r'D:\Object_Detection\model\20180408-102900\facenet_model\20180408-102900.pb'
# 阈值
Threshold = 0.6  # 距离小于阈值就视为是一个人

# 陌生人标注
Strname = 'unKnown'

KN='1'

# 实时视频的输入文件
CAPVideo = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# 图片数据标注函数
def plot_tip(figuredata, boxdata, txt, boxcolor=(255, 97, 0), boxw=2, txtcolor1=(255,0,0),txtcolor2=(0,0,255)):
    # 标注人脸框
    box_data = cv2.rectangle(figuredata, boxdata[0], boxdata[1], boxcolor, boxw) # 注意cv2识别的颜色是BGR
    # 添加文字
    # font = cv2.FONT_HERSHEY_COMPLEX
    x, y = boxdata[0][0], boxdata[0][1]
    if txt == '无法识别':
        txt_data = cv2AddChineseText(box_data, txt, (x, y - 20), textColor=(0,0,255), textSize=20)
    else:
        txt_data = cv2AddChineseText(box_data, txt, (x, y - 20), textColor=(0,255,0), textSize=20)
    # txt_data = cv2.putText(box_data, txt, (x, y), font, fontScale=0.6, color=txtcolor, thickness=1)
    return txt_data



# 视频人脸识别
def face_video_iden(fs=ImageSize,md=PreTrainModelDir, jp=JsonPath, tn=Threshold, keyn=KN,
                     rfs=Size_KeyPoints[KN][0], dstdata=Size_KeyPoints[KN][1], sn=Strname, cap=CAPVideo):
    # WIDTH/HEIGHT必须和摄像头逐帧捕获的分辨率一致，否则会生成1kb视频文件并且无法播放,by Navy 2022-03-31
    # 通过frame.shape获取摄像头逐帧分辨率,by Navy 2022-03-31
    WIDTH = 640
    HEIGHT = 480
    FILENAME = r'D:\Object_Detection\video\myvideo2.MP4'

    FPS = 18
    CAPVideo.set(cv2.CAP_PROP_FPS, 24)
    # 如下fourcc参数必须是小写，用大写会有OpenCV报错,by Navy 2022-03-31
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(FILENAME, fourcc=fourcc, fps=FPS, frameSize=(WIDTH, HEIGHT))

    # 获取数据库编码
    with open(jp, 'r') as g:
        json_str = g.read()
    face_code_dict = dict(json.loads(json_str))
    # 人脸检测模型
    detector = MTCNN()
    # 人脸编码模型
    sess = tf.Session()
    # 下载预训练模型参数
    facenet.load_model(md)
    # 根据名称获取相应的张量
    image_input_tensor = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings_tensor = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_tensor = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    # 开始遍历每个图片
    while 1:
        ret, image_data = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # 读取图片数据，并转换通道
        result_detection = detector.detect_faces(image_data)
        # 存储识别到的人脸
        face_set_signed = {}
        # 没有识别到的人脸
        fce_set_nosign = {}
        fce_set_nosign[sn] = []
        # 判断是否检测出人脸
        if result_detection:
            # 遍历获取到的每一个脸
            for face in result_detection:
                # 获取框的像素坐标
                minx, miny, width, height = face['box']
                maxx, maxy = minx + width, miny + height
                # 将检测到的人脸图片截取下来
                face_detection_data = image_data[miny:maxy, minx:maxx, :]
                face_detection_data = cv2.cvtColor(face_detection_data, cv2.COLOR_RGB2BGR)
                # 获得五个关键点的坐标
                src_data = []
                for k in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']:
                    src_data.append(face['keypoints'][k])
                # 计算仿射矩阵
                tform = trans.SimilarityTransform()
                res = tform.estimate(np.array(src_data), dstdata)
                M = tform.params
                # 应用仿射矩阵进行人脸对齐
                align_face_data = cv2.warpAffine(image_data.copy(), M[:2, :], rfs,
                                                 flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
                # 进入人脸编码模型
                code_face_data = cv2.resize(align_face_data, (fs, fs), interpolation=cv2.INTER_CUBIC)
                # 图片数据预处理
                code_face_data = facenet.prewhiten(code_face_data)
                # 数据增加维度
                code_face_data = code_face_data.reshape(-1, fs, fs, 3)
                # 编码数据，长度为512
                embeddings_data = sess.run(embeddings_tensor,
                                           feed_dict={image_input_tensor: code_face_data,
                                                      phase_train_tensor: False})[0].tolist()

                # 开始遍历数据库编码
                person_dict = {}
                for person in face_code_dict:
                    code_list1 = []
                    code_list2 = []
                    for fs_t in face_code_dict[person]:
                        for size in face_code_dict[person][str(fs_t)]:
                            if fs_t == '140':
                                data_code = face_code_dict[person][str(fs_t)][size]
                                code_list1.extend(data_code)
                            if fs_t == '169':
                                data_code = face_code_dict[person][str(fs_t)][size]
                                code_list2.extend(data_code)

                    all_dis1 = []
                    all_dis2 = []
                    for pco in code_list1:
                        all_dis1.append(cde.compare_dis_em(embeddings_data, pco))
                    for pco1 in code_list2:
                        all_dis2.append(cde.compare_dis_em(embeddings_data, pco1))

                    # 计算均值
                    dis1_mean = sum(all_dis1) / len(code_list1)
                    dis2_mean = sum(all_dis2) / len(code_list2)
                    # person_dict[person] = min(dis1_mean, dis2_mean)
                    person_dict[person] = min(min(all_dis1), min(all_dis2))

                    # person_dict[person] = min(all_dis)

                # 所有的均值都不小于阈值，则为陌生人，否则选择最小的作为身份
                print(sorted(person_dict.items(), key=lambda s:s[1]))
                min_dis = sorted(person_dict.items(), key=lambda s:s[1])[0]

                if min_dis[1] > tn:
                    # 陌生人
                    fce_set_nosign[sn].append([(minx, miny), (maxx, maxy)])
                else:
                    face_set_signed[min_dis[0]] = [(minx, miny), (maxx, maxy)]

        dic_trans = {sn: '无法识别', 'wanghaojie': '王豪杰', 'yangqun': '杨群', 'huangbo': '黄渤', 'lijiaxin': '李嘉欣',
                     'wangxun': '王迅', 'gaierjiaduo': '盖尔-加朵', 'songguomin': '宋国民', 'zhanna': '詹娜', 'linyuner': '林允儿',
                     'madongxi': '马东锡', 'sanjicaihua': '三吉彩花', 'yueyunpeng': '岳云鹏'}

        # 开始进行标注
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        if fce_set_nosign[sn]:
            for data_f in fce_set_nosign[sn]:
                image_data = plot_tip(image_data, data_f, dic_trans[sn])
        if face_set_signed:
            for fkey in face_set_signed:
                if fkey in dic_trans:
                    image_data = plot_tip(image_data, face_set_signed[fkey], dic_trans[fkey])
                else:
                    image_data = plot_tip(image_data, face_set_signed[fkey], fkey)

        cv2.namedWindow('object detection', cv2.WND_PROP_FULLSCREEN)  # 支持全屏,by Navy,2022.04.01
        # 保存为图片


        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        cv2.imshow('object detection', image_data)

        out.write(image_data)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            # cv2.destroyAllWindows()
            break



    out.release()
    cap.release()
    cv2.destroyAllWindows()


face_video_iden()