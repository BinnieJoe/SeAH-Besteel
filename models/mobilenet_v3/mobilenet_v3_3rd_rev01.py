# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:26:03 2023

@author: nyj
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
import traceback

class CustomDataset(Dataset):
    def __init__(self, image_folder='crop/', transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    
    def is_image_path(self):  # 인스턴스 메서드로 사용하여 인스턴스 속성에 접근
        if os.path.isfile(self.image_folder):  # 수정: 폴더인지 아닌지 여부만 확인하면 됩니다.
            print(os.path.isfile(self.image_folder))
            return True  # 이미지 폴더를 가리킴
        else:
            return False  # 이미지 폴더가 아님
    
    # 데이터셋의 전체 길이
    def __len__(self):
        return len(self.image_paths)

    # 인덱스에 해당하는 이미지 불러와서 변환
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        if os.path.isfile(img_path) and any(img_path.lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
            # 이미지인 경우에만 처리
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, img_path
        else:
            # 이미지가 아닌 경우 pass
            pass

def mobilenet_v3_3rd_floor_isfile(image_path):
    predictions = []
    
    # 입력 이미지 변환 (이미지 크기, 텐서형태로 변환)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 이미지 불러오기
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # gpu 사용시
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cpu 사용시
    device = torch.device("cpu")

    # 모델 불러오기
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(device)
    num_classes = 6
    model.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1280, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    )
    model = model.to(device)
    # print(model.classifier)
    # 저장된 모델 가중치 불러오기
    pth_path = 'weight/mobilenet_v3/3rd_floor/mobilenet_v3_rev07.pth'
    
    try:
    
        # gpu 사용시, 사용가능
        # model.load_state_dict(torch.load(pth_path))
        # cpu 사용시, 사용가능
        # model.load_state_dict(torch.load(pth_path, map_location=device)['net'] )
        model.load_state_dict(torch.load(pth_path, map_location=device))
        # print('model',model)
    
        # 모델을 평가모드로 전환
        model.eval()
        
        # 연산시 기울기 계산을 하지 않도록 함 (평가모드이기 때문)
        with torch.no_grad():
            ################# 단일 이미지 판정 #################
            # 이미지 분류
            image = image.to(device)
            output = model(image)
            # 각 클래스 중 가장 높은 값(확률)을 가진 인덱스 선택
            pred = output.argmax(dim=1, keepdim=True)
            
            # 반환 형식 list로 변경
            predictions = [image_path, pred.item()]
            
    except Exception as e:
        traceback.print_exc()
        print('mobilenet_v3_3rd 에러발생')
        print(f'{e}')
        # pass
    # print(predictions)
    return predictions


def mobilenet_v3_3rd_floor_isfolder(image_folder= 'crop/'):
    # def mobilenet_v3_3rd_floor(image_path):
    predictions = []
    
    # 입력 이미지 변환 (이미지 크기, 텐서형태로 변환)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 이미지 불러오기
    # image = Image.open(image_path).convert('RGB')
    # image = transform(image).unsqueeze(0)  # Add batch dimension


    # # 커스텀 데이터셋을 생성하여 데이터를 로드
    test_dataset = CustomDataset(image_folder=image_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # gpu 사용시
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cpu 사용시
    device = torch.device("cpu")

    # 모델 불러오기
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(device)
    num_classes = 6
    model.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1280, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    )
    model = model.to(device)
    # print(model.classifier)
    # 저장된 모델 가중치 불러오기
    pth_path = 'weight/mobilenet_v3/3rd_floor/mobilenet_v3_rev07.pth'
    
    try:
    
        # gpu 사용시, 사용가능
        # model.load_state_dict(torch.load(pth_path))
        # cpu 사용시, 사용가능
        # model.load_state_dict(torch.load(pth_path, map_location=device)['net'] )
        model.load_state_dict(torch.load(pth_path, map_location=device))
        # print('model',model)
    
        # 모델을 평가모드로 전환
        model.eval()
        
        # 연산시 기울기 계산을 하지 않도록 함 (평가모드이기 때문)
        with torch.no_grad():
            ################# 단일 이미지 판정 #################
            # 이미지 분류
            # image = image.to(device)
            # output = model(image)
            # # 각 클래스 중 가장 높은 값(확률)을 가진 인덱스 선택
            # pred = output.argmax(dim=1, keepdim=True)
            
            # # 반환 형식 list로 변경
            # predictions = [image_path, pred.item()]
            
            ################# 특정 폴더 전체 판정 #################
            for data, img_path in test_loader:
                data = data.to(device)
                
                # 이미지 분류
                output = model(data)
                # print(output)
                # 각 클래스 중 가장높은 값(확률)을 가진 인덱스 선택
                pred = output.argmax(dim=1, keepdim=True)
                # print(pred)

                # 반환 형식 변경
                # prediction = (image_path, pred.item())
                
                # 특정 폴더 전체 판정
                for i in range(len(pred)):
                    # print('pred',pred[i].tolist()[0], type(pred[i].tolist()[0]))
                    # print('img_path', img_path[i])
                    predictions.append([img_path[i], pred[i].tolist()[0]])
    
        # # GPU 메모리 사용 최적화 (gpu 사용시)
        # torch.cuda.empty_cache()
    
        # print(pred.flatten().tolist())
        # print('mobilenet_v3 완료, :', predictions)
    except Exception as e:
        traceback.print_exc()
        print('mobilenet_v3_3rd 에러발생')
        print(f'{e}')
        # pass
    # print(predictions)
    return predictions

# mobilenet_v3_test_3()