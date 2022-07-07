#!/usr/bin/env python
# author:AnFany
# datetime:2020/12/11 16:00

# 实现实时视频的目标检测

import cv2
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image


# 模型名称
Model_Path = r'E:\odapi_gwt\models\workspace\training_demo_james_kobe\exported-models\Lebron_Kobe_Model'
# 标签文件
PATH_TO_LABELS = r'E:\odapi_gwt\models\workspace\training_demo_james_kobe\annotations\label_map.pbtxt'

# 实时视频的输入文件
CAPVideo = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def real_time_object_detection(modelp=Model_Path, labelsp=PATH_TO_LABELS, cap=CAPVideo):
    # 获取识别物体的标签对应字典
    category_index = label_map_util.create_category_index_from_labelmap(labelsp, use_display_name=True)
    # 加载模型
    detect_fn = tf.saved_model.load(modelp + "/saved_model")

    while 1:
        ret, image_np = cap.read()
        print(image_np)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=3,
            min_score_thresh=0.2,
            agnostic_mode=False)
        #
        cv2.imshow('object detection', image_np_with_detections)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cap.release()
    cv2.destroyAllWindows()


real_time_object_detection()