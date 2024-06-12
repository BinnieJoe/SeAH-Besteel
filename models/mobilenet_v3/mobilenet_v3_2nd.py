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
# from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet, mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

class CustomDataset(Dataset):
    # 데이터셋의 경로와 변환함수 초기화
    def __init__(self, image_folder='crop/', transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    
    # 데이터셋의 전체 길이
    def __len__(self):
        return len(self.image_paths)

    # 인덱스에 해당하는 이미지 불러와서 변환
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_path

def mobilenet_2nd_floor(image_folder= 'crop/'):
    predictions = []
    
    # 입력 이미지 변환 (이미지 크기, 텐서형태로 변환)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 커스텀 데이터셋을 생성하여 데이터를 로드
    test_dataset = CustomDataset(image_folder=image_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # gpu 사용시
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cpu 사용시
    device = torch.device("cpu")

    # 모델 불러오기
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(device)
    num_classes = 2
    model.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1280, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    )
    model = model.to(device)
    
    # 저장된 모델 가중치 불러오기
    # pth_path = 'weight/mobilenet_v3/2nd_floor/mobilenet_v3_2nd_rev00.pth'
    pth_path = 'weight/mobilenet_v3/2nd_floor/mobilenet_v3_2nd_rev02.pth'
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
            for data, img_path in test_loader:
                data = data.to(device)
                
                # 이미지 분류
                output = model(data)
                # print(output)
                # 각 클래스 중 가장높은 값(확률)을 가진 인덱스 선택
                pred = output.argmax(dim=1, keepdim=True)
                # print(pred)
                
                for i in range(len(pred)):
                    # print('pred',pred[i].tolist()[0], type(pred[i].tolist()[0]))
                    # print('img_path', img_path[i])
                    predictions.append([img_path[i], pred[i].tolist()[0]])
    
        # # GPU 메모리 사용 최적화 (gpu 사용시)
        # torch.cuda.empty_cache()
    
        # print(pred.flatten().tolist())
        # print('mobilenet_1 완료, :', predictions)
    except Exception as e:
        print('mobilenet_1st 에러발생')
        print(f'{e}')
        # pass
    # print(predictions)
    return predictions

# mobilenet_test_1()