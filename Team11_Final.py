
#------Made by Ken Huang & Kai Fu---------
#-----Add FPS　calculate------#
import pickle
import pandas as pd
from tensorflow import string
#Load ML Model
with open('body_language.pkl','rb') as f:
    model=pickle.load(f)
#print(model)
import math
import cv2
import datetime
import time
import mediapipe as mp
import numpy as np
import csv
import os


import argparse
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy

#最佳訓練PT檔位置
#model_path ='C:/Users/user/PycharmProjects/finalproject _V2/yolov5/best_K.pt'
model_path ='C:/Users/user/PycharmProjects/finalproject _V2/yolov5/best55.pt'
#model_path ='C:/Users/user/PycharmProjects/finalproject _V2/yolov5/best_yolov5m_traindata_equal_test_mAP096.pt'

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


#載入YOLO模型
model_yolo = torch.hub.load('C:/Users/user/PycharmProjects/finalproject _V2/yolov5', 'custom', path=model_path, source='local')

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測


cap = cv2.VideoCapture(0)
class_name='Shoulder'

shoulder_count=0  #紀錄測飛鳥次數
shoulder_dir=0    #記錄測飛鳥方向

detect_TF=0       #判別有無偵測到啞鈴，0:未偵測  1:偵測到

q_count=0         #紀錄二頭彎舉次數
q_dir=0           #記錄二頭彎舉方向

pTime = 0         #計算FPS

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img  = cv2.resize(img,(920,720))              # 修改尺寸
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB,因opencv讀取影像的順序為BGR,測測pose需要的影像順序為 RGB
        results = pose.process(img2)                  # 取得姿勢偵測結果

        cv2.rectangle(img, (0, 620), (100, 720), (255, 255, 255), -1)


        detect_dumbell=model_yolo(img2)                                   #將YOLO模型套用到圖片上
        detect_data=numpy.array(detect_dumbell.xyxy[0].cpu().numpy())     #將YOLO偵測的數值轉換成array


        #len=0,yolo資料為空->未偵測到啞鈴
        if(len(detect_data)==0):
            cv2.rectangle(img, (720, 0), (920, 60), (105, 105, 105), -1)
            cv2.putText(img, 'Warm up', (740, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            detect_TF=0
            print('No detect dumbell')
        else:

            for i in range(detect_data.shape[0]):
                #偵測到啞鈴
                if(detect_data[i][4]>=0.6):   #預測率60%以上才判定為偵測到
                    cv2.rectangle(img, (720, 0), (920, 60), (255, 0, 0), -1)
                    cv2.putText(img, 'Training', (740, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    detect_TF = 1

                    #繪製啞鈴偵測方塊圖
                    height1 = int(detect_data[i][0])
                    width1  = int(detect_data[i][1])
                    height2 = int(detect_data[i][2])
                    width2  = int(detect_data[i][3])
                   # prob    = (detect_data[i][4])
                    cv2.rectangle(img, (height1, width1), (height2, width2), (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(img,'dumbell',(height1-13, width1),cv2.FONT_HERSHEY_SIMPLEX,1.2,(100,200,200),3)
                    #print(prob)

        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        try:

            # # 使用 try 避免抓不到姿勢時發生錯誤
            # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0
            # # 如果滿足模型判斷條件 ( 表示要換成背景 )，回傳 True
            # img = np.where(condition, img, bg)
            # # 將主體與背景合成，如果滿足背景條件，就更換為 bg 的像素，不然維持原本的 img 的像素


            #print(results.pose_landmarks.landmark)
            pose_t=results.pose_landmarks.landmark
            pose_row=list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility]for landmark in pose_t]).flatten())
            #print(pose_row)

            row=pose_row

            #make detection
            X=pd.DataFrame([row])
            body_language_class=model.predict(X)[0]
            body_language_prob =model.predict_proba(X)[0]
            body_language_prob_1=body_language_prob[1]
            body_language_prob_0=body_language_prob[0]

            #角度變化偵測點
            x11 = int(pose_t[11].x*imgWidth)
            y11 = int(pose_t[11].y*imgHeight)

            x23 = int(pose_t[23].x * imgWidth)
            y23 = int(pose_t[23].y * imgHeight)

            x13 = int(pose_t[13].x * imgWidth)
            y13 = int(pose_t[13].y * imgHeight)

            x15 = int(pose_t[15].x * imgWidth)
            y15 = int(pose_t[15].y * imgHeight)

            #cal angle
            angle_shoulder=-1*math.degrees(math.atan2(y13-y11,x13-x11)-
                                        math.atan2(y23-y11,x23-x11))

            angle_q = math.degrees(math.atan2(y11 - y13, x11 - x13) -
                                               math.atan2(y15 - y13, x15 - x13))

            #cal %
            per_shoulder=np.interp(angle_shoulder,(60,90),(0,100))

            per_q = np.interp(angle_q, (-178, -20), (0, 100))

            #bar
            bar=np.interp(angle_shoulder,(60,90),(600,150))

            q_bar=np.interp(angle_q,(-178,-20),(600,150))

            #check curls
            color=(255,0,255)
            if(detect_TF==1):
                if per_shoulder==100:
                    color=(0,255,0)
                    if shoulder_dir ==0:
                        shoulder_count+=0.5
                        shoulder_dir=1
                if per_shoulder==0:
                    if shoulder_dir ==1:
                        shoulder_count+=0.5
                        shoulder_dir=0

                q_color = (160, 82, 45)
                if per_q == 100:
                    q_color = (0, 128, 255)
                    if q_dir == 0:
                        q_count += 0.5
                        q_dir = 1
                if per_q == 0:
                    if q_dir == 1:
                        q_count += 0.5
                        q_dir = 0

            #若預測機率大於50%時，開始作動
            if(body_language_prob_0>0.55 or body_language_prob_1>0.55):

                #shoulder
                if(body_language_prob_0>body_language_prob_1):

                    #draw bar
                    cv2.rectangle(img, (800, 150), (860, 600), color, 3)
                    cv2.rectangle(img, (800, int(bar)), (860, 600), color, cv2.FILLED)

                    #draw per
                    cv2.putText(img, f'{int(per_shoulder)}%', (800, 120), cv2.FONT_HERSHEY_PLAIN, 3, color, 4)
                    #draw count
                    cv2.putText(img, str(int(shoulder_count)), (30, 700), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,0), 5)

                    #---------------------------------------------------------#
                    cv2.rectangle(img, (0, 0), (250, 60), (0,255,0), -1)
                    # Display class
                    cv2.putText(img, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)

                    # Display prob
                    cv2.putText(img, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                                , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    print("test1")
 #*******************************************************************************************************#
                #q
                if (body_language_prob_0 < body_language_prob_1):

                    #draw bar
                    cv2.rectangle(img, (800, 150), (860, 600), q_color, 3)
                    cv2.rectangle(img, (800, int(q_bar)), (860, 600), q_color, cv2.FILLED)

                    #draw per
                    cv2.putText(img, f'{int(per_q)}%', (800, 120), cv2.FONT_HERSHEY_PLAIN, 3, q_color, 4)
                    #draw count
                    cv2.putText(img, str(int(q_count)), (30, 700), cv2.FONT_HERSHEY_PLAIN, 5, (0, 128, 255),
                                    5)
                    #-------------------------------------------------------#
                    cv2.rectangle(img, (0, 0), (250, 60), (0, 128, 255), -1)
                    # Display class
                    cv2.putText(img, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)

                    # Display prob
                    cv2.putText(img, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                                , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    print("test2")
        except:
            pass
        #cv2.circle(img, (300, 400), 50, (0, 0, 255), -1)
            # show FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        fps_str = "fps:"+ str(int(fps))
        cv2.putText(img, fps_str, (10, 100), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0), 3)
        cv2.imshow('Team11_Final', img)


        #鍵盤按下'q' 即紀錄訓練CSV檔
        if cv2.waitKey(5) == ord('q'):
            #currentDateAndTime = datetime.now()
            strday=str(datetime.date.today())
            localtime = time.localtime()
            result = time.strftime("%I:%M:%S%p", localtime)
            str1=str(q_count)
            str2=str(shoulder_count)
            table = [
                [result, ''],
                [strday, '二頭', '肩膀'],
                ['次數', str1, str2],
                ['-------------------------------------']
            ]
            with open('train_record.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 寫入二維表格
                writer.writerows(table)
            #print(q_count)
            break     # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()



