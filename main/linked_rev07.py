# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:13:40 2023

@author: nyj

"""

##############################################################
# 'data' 디렉토리 > yolo모델 > 'crop' 디렉토리 > Resnet 모델 > 리스트 반환

# Team SinLeeNaJoe (신이나조)
##############################################################

# 총 소요시간 체크
import time
start= time.time()

# custom .py 불러오기
from mobilenet_v3_1st_rev01 import mobilenet_1st_floor
from mobilenet_v3_2nd import mobilenet_2nd_floor
from yolo_v8s_beta import yolo_test
from mobilenet_v3_3rd_rev01 import mobilenet_v3_3rd_floor_isfolder, mobilenet_v3_3rd_floor_isfile

# 라이브러리 임포트
import os
import shutil
import cv2
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
# from torchvision.datasets import ImageFolder
import warnings
warnings.filterwarnings('ignore')
import traceback

def main():
    # 'crop' 디렉토리가 없으면 생성
    crop_dir = 'crop/'
    if not os.path.exists(crop_dir):
        os.mkdir(crop_dir)
        
    result_dir = 'B_result/'

    # B_result 폴더 생성
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        # print(f"{result_dir} 폴더 생성됨")

    # Clean, Little, Curve, Dirty, Thick, nodetect폴더 생성
    subfolders = ['Clean', 'Little', 'Curve', 'Dirty', 'Thick', 'nodetect']
    for folder in subfolders:
        folder_path = os.path.join(result_dir, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            # print(f"{folder_path} 폴더 생성됨")

    # 'crop' 디렉토리 내의 모든 파일 삭제
    for filename in os.listdir(crop_dir):
        file_path = os.path.join(crop_dir, filename)
        # print('crop 디렉토리 내의 모든 파일 삭제 :', file_path)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

    # 'data' 폴더에서 원본 이미지 가져오기
    input_data_folder = 'data'  
    image_files = [file for file in os.listdir(input_data_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [os.path.join(input_data_folder, file) for file in image_files]
    
    #################### modeling ####################

    # 원본이미지 path 하나씩
    # for image in images:
        
    # cmd 창에서 입력한 이미지 path를 input
    # input_data_folder= sys.argv[1]
    #input_data_folder= r"C:\Users\user\Desktop\Burr_Classification\data"
    # YOLO모델로 예측하여 'crop' 디렉토리에 단면이미지 저장
    # print('input_=', input_data_folder, 'output_=', crop_dir)
    
    # for i in range(0, len(images), batch_size):
    #     batch = images[i:i+batch_size]
        # yolo_test(input_= batch, output_= crop_dir, height=1080, width=1920)
    yolo_test(input_= input_data_folder, output_= crop_dir, height=1080, width=1920)

    # 첫번째 분류모델
    # ('이미지 경로', 클래스) 반환
    floor_1= mobilenet_1st_floor(crop_dir)
    # print(floor_1, len(floor_1))
    # 첫번째 분류모델 이미지 저장
    for i in range(len(floor_1)):
        crop_path= floor_1[i][0]
        crop_class= floor_1[i][1]
        crop_name= crop_path.split('/')[-1]
        crop= cv2.imread(crop_path)
        cv2.imwrite(f'AB_CE_D/{crop_class}/{crop_name}',crop)

    # print(f"crop 저장완료 : {time.time() - start:.5f} sec")
    # ############## classification 1 clear #####################
    def classification_2():
        CLASS = {0 : 'Clean', 1 : 'Little', 2 : 'Curve', 3 : 'Dirty', 4 : 'Thick'}
        result=[]
        # print(f"{time.time() - start:.5f} sec")
        ##### CE 등급 #####
        c_path ='AB_CE_D/1/'
        if len(os.listdir(c_path)) == 0:
            # print('Dirty 없음')
            pass
        else:
            # print('Dirty 있음')
            CE = mobilenet_v3_3rd_floor_isfolder(c_path)
            for i in CE:
                file_path = i[0]
                image_name = file_path.replace('\\', '/').split('/')[-1]
                if i[1] == 2:
                    shutil.move(file_path, f'B_result/{CLASS[2]}/{image_name}')
                    i[1]= 'Curve'
                elif i[1] == 4:
                    shutil.move(file_path, f'B_result/{CLASS[4]}/{image_name}')
                    i[1]= 'Thick'
                else:
                    shutil.move(file_path, f'B_result/{CLASS[3]}/{image_name}')
                    i[1]= 'Dirty'
                result.append(i)
            

        ##### D 등급 #####
        d_path ='AB_CE_D/2/'
        if len(os.listdir(d_path)) == 0:
            # print('Thick 없음')
            pass
        else:
            # print('Thick 있음')
            D = mobilenet_v3_3rd_floor_isfolder(d_path)
            for i in D:
                file_path = i[0]
                image_name = file_path.replace('\\', '/').split('/')[-1]

                if i[1] == 2:
                    shutil.move(file_path, f'B_result/{CLASS[2]}/{image_name}')
                    i[1]= 'Curve'
                elif i[1] == 3:
                    shutil.move(file_path, f'B_result/{CLASS[3]}/{image_name}')
                    i[1]= 'Dirty'
                elif i[1] == 5:
                    shutil.move(file_path, f'B_result/{CLASS[3]}/{image_name}')
                    i[1]= 'Dirty'
                else:
                    shutil.move(file_path, f'B_result/{CLASS[4]}/{image_name}')
                    i[1]= 'Thick'
                result.append(i)
                
        ##### AB 등급 #####
        ab_path ='AB_CE_D/0/' 
        if len(os.listdir(ab_path)) == 0:
            print('Clean or Curve 없음', os.listdir(ab_path))
            pass
        # AB 분류모델 : 튜플('이미지 경로', 클래스) 반환
        # print('AB',AB)
        # 1폴더 내에 사진들을 하나씩 순회
        else:
            # print('Clean or Curve 있음')
            ######## A(Clean) + A'(Little) vs B(Curve)분류 모델 ########    
            # AB= resnet_2nd_floor(ab_path)
            AB= mobilenet_2nd_floor(ab_path)
            for i in AB:
                # print('AB: ', i)
                # 튜플(사진 하나에 대한 정보)을 하나씩 append
                if i[1] == 0:
                    file_path = i[0]
                    # print(file_path)
                    # image = cv2.imread(file_path)
                    image_name = file_path.replace('\\', '/').split('/')[-1]
                    # print(file_path.replace('\\', '/').split('/')[-1])
                    
                    ######## A(Clean) vs A'(Little) 분류 모델 ########                   
                    # i = mobilenet_3rd_floor(file_path)
                    # i = efficientnet_3rd_floor(file_path)
                    i = mobilenet_v3_3rd_floor_isfile(file_path)
                    # print('mobilenet_v3_3rd_floor: ', i)
                    if i[1] == 0:
                        shutil.move(file_path, f'B_result/{CLASS[0]}/{image_name}')
                        i[1]= 'Clean'
                    elif i[1] == 1:
                        shutil.move(file_path, f'B_result/{CLASS[1]}/{image_name}')
                        i[1]= 'Little'
                    elif i[1] == 2:
                        shutil.move(file_path, f'B_result/{CLASS[2]}/{image_name}')
                        i[1]= 'Curve'
                    elif i[1] == 3:
                        shutil.move(file_path, f'B_result/{CLASS[3]}/{image_name}')
                        i[1]= 'Dirty'
                    elif i[1] == 4:
                        shutil.move(file_path, f'B_result/{CLASS[4]}/{image_name}')
                        i[1]= 'Thick'
                    elif i[1] == 5:
                        shutil.move(file_path, f'B_result/{CLASS[3]}/{image_name}')
                        i[1]= 'Dirty'
                else:
                    file_path = i[0]
                    # print(file_path)
                    # image = cv2.imread(file_path)
                    image_name = file_path.replace('\\', '/').split('/')[-1]
                     ######## B(Curve) vs 분류 모델 ########  
                    i = mobilenet_v3_3rd_floor_isfile(file_path)
                    # print('mobilenet_v3_3rd_floor: ', i)
                    if i[1] == 3:
                        shutil.move(file_path, f'B_result/{CLASS[3]}/{image_name}')
                        i[1]= 'Dirty'
                    elif i[1] == 4:
                        shutil.move(file_path, f'B_result/{CLASS[4]}/{image_name}')
                        i[1]= 'Thick'
                    elif i[1] == 5:
                        shutil.move(file_path, f'B_result/{CLASS[3]}/{image_name}')
                        i[1]= 'Dirty'
                    else:
                        shutil.move(file_path, f'B_result/{CLASS[2]}/{image_name}')
                        i[1]= 'Curve'
                result.append(i)
                # print('be_i',new_i)
        return result
    
    def get_image_paths(base_folder):
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_paths = []

        for root, dirs, files in os.walk(base_folder):
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                if subdir.isdigit() and int(subdir) in [0, 1, 2]:
                    for file in os.listdir(subdir_path):
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            file_path = os.path.join(subdir_path, file)
                            image_paths.append(file_path)

        return image_paths

    # resnet 판정이 계속 일부분 누락되므로 while문으로 지속 반복함
    root_dir = 'AB_CE_D/'  
    image_paths = get_image_paths(root_dir)

    total_result = []
    while len(image_paths) != 0:
        try:
            result = classification_2() 
            # print(result)
            total_result.extend(result)
            image_paths = get_image_paths(root_dir)
        except Exception as e:
            traceback.print_exc()
            print(f"An error occurred: {e}")

    # print(len(total_result), total_result)
         
    # 이미지 파일 명으로 sorted - x는 ('경로',클래스)
    sorted_left = sorted(total_result, key=lambda x: int(x[0].split('_')[-1].strip('.jpg')))
    # print('기준:', result[0].split('_')[-1].strip('.jpg'))
    # print('sorted : ', sorted_left)        
  
    class_result=[]
    # 클래스 리스트에 append
    for i in range(len(sorted_left)):
        # crop_path= sorted_left[i][0]
        crop_class= sorted_left[i][1]
        # crop_name= crop_path.split('/')[-1]
        # crop= cv2.imread(crop_path)
        # cv2.imwrite(f'crop{crop_class}/{crop_name}',crop)
        class_result.append(crop_class)

    print('class_result', class_result)
            
    ################ 2floor clear ####################
    
    # 'data' 디렉토리 내의 처리가 완료된 파일 원본이미지 이동
    for file_path in sorted_left:
        source_filename = file_path[0].replace('\\', '/').split('/')[-1]
        # print(file_path)
        source_path = os.path.join(input_data_folder, source_filename)
        source_path = source_path[:-6] + ".jpg"
        
        destination_filename = file_path[0].replace('\\', '/').split('/')[-1]
        destination_path = os.path.join(result_dir, destination_filename)
        destination_path = destination_path[:-6] + ".jpg"
        
        # 파일을 옮기기 전에 소스 파일이 존재하는지 확인합니다.
        if os.path.exists(source_path):
            try:
                shutil.move(source_path, destination_path)
                # print(f"{source_path} 파일을 {destination_path}로 이동했습니다.")
            except FileNotFoundError as e:
                print(f"오류: {e}")
        else:
            pass
            # print(f"{source_path} 파일이 존재하지 않습니다.")
    
    # data 폴더 재탐색하여 nodetection사진 확인 후 B_result에 nodetect로 이동
    image_files = [file for file in os.listdir(input_data_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [os.path.join(input_data_folder, file) for file in image_files]
    if images is not None:
        for file_path in images:
            move_path = file_path.replace('\\', '/').split('/')[-1]
            # print(move_path)
            shutil.move(file_path, f'B_result/nodetect/{move_path}')
            
    # 'data' 디렉토리 내의 처리가 완료된 파일 삭제    
    # try:
    #     os.remove(image)
    #     # print(f"{image} 삭제 완료")
    # except FileNotFoundError:
    #     print(f"{image} 파일을 찾을 수 없습니다.")
    # except Exception as e:
    #     print(f"에러 발생: {e}")
    end= time.time()
    print(f"{end - start:.5f} sec")

if __name__ == "__main__":
    main()

