import pandas as pd
import numpy as np
import cv2
import glob
import os
import sys
import shutil
import torch
from PIL import Image
from torchvision import transforms

model_path = 'yolov5ml.pt'  # YOLO 모델 파일 경로
model = torch.load(model_path)
model.eval()  # 모델을 추론 모드로 설정

####################### initial setting #################################
print(cv2.__version__)

treshold_data = [] # all data
index_data = [] # index data
timestamp_data = [] # timestamp data
timestamp_data_final = [] #  timestamp data
slicing_data = [] #100 = 1s

start_anomaly_data = []

csv_file_path = '/workspace/CSV_IMAGE/data2/item_export_10-19-2023-00-58-59.csv.csv' # 입력 CSV 파일 경로
output_csv_file_path = '/workspace/CSV_IMAGE/output' # 결과를 저장할 CSV 파일 경로
video_file_path_R = '/workspace/CSV_IMAGE/video/R' # video path


# 결과를 저장할 DataFrame 생성
output_df1 = pd.DataFrame()

# CSV 파일 읽기
df = pd.read_csv(csv_file_path)



################## IMU Sensor Treshold data and Timestamp extract #############################

for index, row in df.iterrows():
    rms = row['rms']
    
    # rms 값이 11 이상인 경우 처리
    if rms >= 11:
        # 선택된 인덱스를 중심으로 앞뒤 150개의 데이터 인덱스 계산
        start_index = max(0, index - 150)  # 시작 인덱스가 데이터프레임 범위를 벗어나지 않도록 조정
        end_index = min(len(df), index + 151)  # 종료 인덱스가 데이터프레임 범위를 벗어나지 않도록 조정

        # 해당 범위의 데이터를 추출하여 `treshold_data` 리스트에 추가
        for sub_index in range(start_index, end_index):
            row = df.iloc[sub_index]
            acceleration_x = row['acc-x']
            acceleration_y = row['acc-y']
            acceleration_z = row['acc-z']
            gyro_x = row['gyro-x']
            gyro_y = row['gyro-y']
            gyro_z = row['gyro-z']
            timestamp = row['timestamp']
            rms_value = row['rms']
            treshold_data.append([timestamp, acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z, rms_value])
        
        # 첫 번째 조건을 만족하는 순간에 대한 데이터만 처리하고 반복문 종료
        break

# 최종 데이터를 DataFrame으로 변환 및 CSV 파일로 저장
output_df1 = pd.DataFrame(treshold_data)
output_df1.to_csv(output_csv_file_path + 'output2.csv', index=False, header=None)


################### video file name change ##################
file_list_R = os.listdir(video_file_path_R)

for file in file_list_R:
    src = os.path.join(video_file_path_R, file) # 기존 파일 경로
    dst_name = file.replace("REC_", "") # 이름 수정 
    dst = os.path.join(video_file_path_R, dst_name) # 바뀐 이름으로 저장할 경로
    os.rename(src, dst) # rename 수행  

for file in file_list_R:
    src = os.path.join(video_file_path_R, file) # 기존 파일 경로
    dst_name = file.replace("_R", "") # 이름 수정 
    dst = os.path.join(video_file_path_R, dst_name) # 바뀐 이름으로 저장할 경로
    os.rename(src, dst) # rename 수행

for file in file_list_R:
    src = os.path.join(video_file_path_R, file) # 기존 파일 경로
    dst_name = file.replace("_", ":") # 이름 수정 
    dst = os.path.join(video_file_path_R, dst_name) # 바뀐 이름으로 저장할 경로
    os.rename(src, dst) # rename 수행    


cap = cv2.VideoCapture('/workspace/CSV_IMAGE/video/R/2023:10:17:20:57:00.MP4') # time change

if not cap.isOpened():
    print("Camera open failed!") # 열리지 않았으면 문자열 출력
    sys.exit()

print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

fps = cap.get(cv2.CAP_PROP_FPS)
total_time = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
print('FPS:', fps)
print("total time:",total_time)


##############video capture#################hour and minute change#####
count = 1
time = 18 #seconds
while(cap.isOpened()):
    ret, image = cap.read()
    if(int(cap.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
        second = time + count
        if time < 10 :
            if second < 10 :
                cv2.imwrite(video_file_path_R + "/capture/00:58:0%d.jpg" % second, image)
                print('Saved frame number :', str(int(cap.get(1))))
            if second >= 10 and second < 60:
                cv2.imwrite(video_file_path_R + "/capture/00:58:%d.jpg" % second, image)
                print('Saved frame number :', str(int(cap.get(1))))         
            if second >= 60 :
                remain = second % 60
                if remain < 10 :
                    cv2.imwrite(video_file_path_R + "/capture/00:59:0%d.jpg" % remain, image)
                    print('Saved frame number :', str(int(cap.get(1))))    
                if remain >= 10 :
                    cv2.imwrite(video_file_path_R + "/capture/00:59:%d.jpg" % remain, image)
                    print('Saved frame number :', str(int(cap.get(1))))     

        if time >= 10 :
            if second < 60 :
                cv2.imwrite(video_file_path_R + "/capture/00:58:%d.jpg" % second, image)
                print('Saved frame number :', str(int(cap.get(1))))
            if second >= 60 :
                remain = second % 60
                if remain < 10 :
                    cv2.imwrite(video_file_path_R + "/capture/00:59:0%d.jpg" % remain, image)
                    print('Saved frame number :', str(int(cap.get(1))))    
                if remain >= 10 :
                    cv2.imwrite(video_file_path_R + "/capture/00:59:%d.jpg" % remain, image)
                    print('Saved frame number :', str(int(cap.get(1))))    

        count += 1
        if count == 61:
            break
cap.release()


#################### capture file == treshold_timestamp ##########################

capture_list_R = os.listdir(video_file_path_R + "/capture")

filenames = [os.path.splitext(file)[0] for file in capture_list_R]

for i in filenames:
    for j in timestamp_data:
        if i in j:
            for k in capture_list_R:
                if i in k:
                    shutil.copy(os.path.join(video_file_path_R + "/capture", k),
                                os.path.join("/workspace/CSV_IMAGE/video/threshold_image", k))
                                    

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # YOLO 모델에 맞는 크기 조정
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    return image

# 이미지 추론 실행 및 결과 처리 함수
def infer_image(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        predictions = model(image)
    # 추론 결과를 처리하는 코드 추가
    print(predictions)

capture_images_directory = '/workspace/CSV_IMAGE/video/R/capture/'

# 캡처된 이미지 파일들에 대해 추론 실행
for image_file in os.listdir(capture_images_directory):
    image_path = os.path.join(capture_images_directory, image_file)
    infer_image(model, image_path)


######## txt label insert in original data  ##########################
label_list_R = os.listdir("/workspace/CSV_IMAGE/video/label")
labelnames = [os.path.splitext(file)[0] for file in label_list_R]

count = 0

for index, row in df.iterrows():
    acceleration_x = row['acc-x']
    acceleration_y = row['acc-y']
    acceleration_z = row['acc-z']
    gyro_x = row['gyro-x']
    gyro_y = row['gyro-y']
    gyro_z = row['gyro-z']
    timestamp = row['timestamp'] 
    ############ label time == csv time #############
    for names in labelnames:
        file = open("/workspace/CSV_IMAGE/video/label/" + names +".txt","r")
        label = file.readline()
        if names in timestamp:
            treshold_data.append([timestamp, acceleration_x,acceleration_y,acceleration_z,gyro_x,gyro_y,gyro_z,int(label)])
            
            output_df2 = pd.DataFrame(treshold_data)
            output_df2.to_csv(output_csv_file_path + 'final.csv', index=False,header=None)
            print("save csv")
        else:
            break


file.close()


