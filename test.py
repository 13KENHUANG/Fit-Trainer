import cv2
import mediapipe as mp
import numpy as np
import csv
import os
mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測


cap = cv2.VideoCapture(0)
class_name='Shoulder'  #動作一
#class_name='biceps'   #動作二
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
        img  = cv2.resize(img,(520,320))               # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(img2)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        try:
            #print(results.pose_landmarks.landmark)
            pose_t=results.pose_landmarks.landmark
            pose_row=list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility]for landmark in pose_t]).flatten())
            #print(pose_row)
            row=pose_row
            row.insert(0,class_name)
            print(row)
            #export to csv
            with open('coords.csv','a',newline='') as csvfile:
                #csv_writer=csv_writer(csvfile,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(row)
        except:
            pass
        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()