# {'box': [711, 205, 202, 180], 'confidence': 0.9637635946273804, 'keypoints': {'left_eye': (823, 274), 'right_eye': (834, 327), 'nose': (795, 300), 'mouth_left': (762, 293), 'mouth_right': (767, 337)}}
#!/usr/bin/env python
# author:AnFany
# datetime:2020/12/24 17:40

# 利用mtcnn实现图片中人脸的检测

from mtcnn.mtcnn import MTCNN
import cv2
import os

# 输入图片路径
IN_Figure_Path = r'D:\Object_Detection\mtcnn_in'

# 输出图片路径
OUT_Figure_Path = r'D:\Object_Detection\mtcnn_out'

# 进行人脸检测的函数
def mtcnn_face_detection(inp=IN_Figure_Path, outp=OUT_Figure_Path, boxcolor=(220, 20, 20),
                         keypointcolor=(20, 220, 20),boxw=3):
    """
    :param inp: 需要进行人脸检测的图片的路径
    :param outp: 检测后的输出路径
    :param boxcolor: (255,0,0) 对应rgb，检测框的颜色
    :param keypointcolor: (255,0,0) 对应rgb，检测框的颜色
    :param boxw: 检测框的宽度
    :return: 带有检测框的图片
    """
    # 人脸检测模型
    detector = MTCNN()
    for fig in os.listdir(inp):
        # 因为cv2读取的图片数据是bgr的，模型的输入是rgb的
        img_data = cv2.cvtColor(cv2.imread(r'%s/%s' %(inp, fig)), cv2.COLOR_BGR2RGB)
        # 检测结果
        result_detection = detector.detect_faces(img_data)
        # 判断是否检测出人脸
        if result_detection:
            for face in result_detection:
                # 获取框的像素坐标
                minx, miny, width, height = face['box']
                maxx, maxy = minx+width, miny + height
                # 加上人脸框
                face_data = cv2.rectangle(img_data, (minx, miny), (maxx, maxy), boxcolor, boxw)
                # 加上五关键点
                for k in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']:
                    face_data = cv2.circle(face_data, face['keypoints'][k], radius=0,
                                           color=keypointcolor, thickness=boxw*2)
                # 数据通道在变回去
                face_data = cv2.cvtColor(face_data, cv2.COLOR_RGB2BGR)
            # 保存图片
            name = ''.join(fig.split('.')[:-1])
            cv2.imwrite(r'%s/%s.png' % (outp, name), face_data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return print('人脸检测完毕')

mtcnn_face_detection()