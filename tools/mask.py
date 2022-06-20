# -----------------------------------------
# python modules
# 使用torchvision自带的mask r-cnn进行语义分割，提供给SLAM进行动态物体剔除
# 建议配置一下anaconda环境，在安装相关依赖
# -----------------------------------------
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
from PIL import Image
import cv2
from tqdm import tqdm

coco_part = ["person", "bicycle","car","motorbike",'aeroplane', "bus", "train","truck"]#对应coco序号：1~8

def main():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # 使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    root_path = '/home/hai/Data_D/BaiduNetdiskDownload/kittiOdom/color/03'
    img_path = root_path + '/image_2/'
    save_path = root_path + '/demo/'
    imgs = []
    for i in range(60, 65, 1):
        # print(str(i).zfill(6)+'.png')
        imgs.append(str(i).zfill(6)+'.png')
    for name in tqdm(imgs):
        t_start = time.time()
        img = img_path+name
        image = cv2.imread(img)    
        # gamma = 0.7
        # lookUpTable = np.empty((1,256), np.uint8)
        # for i in range(256):
        #     lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        # frame = cv2.LUT(image, lookUpTable)
        # alpha = 3
        # beta = 20

        # frame = np.zeros(image.shape, image.dtype)
        # for y in range(image.shape[0]):
        #     for x in range(image.shape[1]):
        #         for c in range(image.shape[2]):
        #             frame[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
        blob = transform(image)
        c,h,w = blob.shape

        input_x = blob.view(1,c,h,w)
        output = model(input_x.to(device))[0]
        boxes = output['boxes'].cpu().detach().numpy()
        scores = output['scores'].cpu().detach().numpy()
        labels = output['labels'].cpu().detach().numpy()
        masks = output['masks'].cpu().detach().numpy()

        # 对推理预测得到四个输出结果，分别进行解析，其中score阈值为0.5，
        # mask采用soft版本，对大于0.5分割为当前对象像素，这部分的代码实现如下：
        index = 0
        color_mask = np.zeros((h,w,c),dtype=np.uint8)
        mask_gray = np.zeros((h,w), dtype=np.uint8)
        mv = cv2.split(color_mask)
        file = open(save_path+ name.replace('png','txt'), 'w')
        text_to_write = ''
        for x1,y1,x2,y2 in boxes:
            if scores[index] > 0.8 and labels[index]<9:
                text_to_write += ('{} {:.2f} {:.2f} {:.2f} {:.2f} \n').format(coco_part[labels[index]-1], x1, y1, x2, y2)
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                (0,255,255), 2, 8, 0)
                cv2.rectangle(image, (int(x1-8+index*2), int(y1-3)), (int(x2-5+index*2), int(y2-2)),
                (255,0,0), 2, 8, 0)
                mask = np.squeeze(masks[index]>0.5)
                mask_gray+=mask
                np.random.randint(0,256)
                # 对实例分割的对象像素进行随机颜色填充，完成彩色mask图像生成
                mv[2][mask == 1], mv[1][mask == 1], mv[0][mask == 1] = \
                [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
                lable_txt = coco_part[labels[index]-1]
                cv2.putText(image, str(6-index), (np.int32(x1), np.int32(y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            index += 1
        file.write(text_to_write)
        file.close()
        maskImg = Image.fromarray(255*mask_gray)
        maskImg.save(save_path+'mask_'+name)
        color_mask = cv2.merge(mv)
        # 在原图中标注
        result = cv2.addWeighted(image, 0.5, color_mask,0.5, 0)
        cv2.imwrite(save_path+name, result)

        print('inference time: %.2f'%(time.time() - t_start))

def test_one():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # 使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img_path = '/home/hai/mydata/odom/left/000712.png'

    image = cv2.imread(img_path)
    brightness = get_lightness(image)
    print(brightness)
    alpha = 4
    beta = 25

    frame = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                frame[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    # gamma = 0.7
    # lookUpTable = np.empty((1,256), np.uint8)
    # for i in range(256):
    #     lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    # frame = cv2.LUT(image, lookUpTable)
    blob = transform(frame)
    c,h,w = blob.shape

    input_x = blob.view(1,c,h,w)
    output = model(input_x.to(device))[0]
    boxes = output['boxes'].cpu().detach().numpy()
    scores = output['scores'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()
    masks = output['masks'].cpu().detach().numpy()

    # 对推理预测得到四个输出结果，分别进行解析，其中score阈值为0.5，
    # mask采用soft版本，对大于0.5分割为当前对象像素，这部分的代码实现如下：
    index = 0
    color_mask = np.zeros((h,w,c),dtype=np.uint8)
    mask_gray = np.zeros((h,w), dtype=np.uint8)
    mv = cv2.split(color_mask)
    # file = open(save_path+ name.replace('png','txt'), 'w')
    for x1,y1,x2,y2 in boxes:
        if scores[index] > 0.5 and labels[index]<9:
            # text_to_write += ('{} {:.2f} {:.2f} {:.2f} {:.2f} \n').format(coco_part[labels[index]-1], x1, y1, x2, y2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
            (0,255,255), 1, 8, 0)
            mask = np.squeeze(masks[index]>0.5)
            mask_gray+=mask
            np.random.randint(0,256)
            # 对实例分割的对象像素进行随机颜色填充，完成彩色mask图像生成
            mv[2][mask == 1], mv[1][mask == 1], mv[0][mask == 1] = \
            [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
            # lable_txt = coco_part[labels[index]-1]
            cv2.putText(frame, str(labels[index]), (np.int32(x1), np.int32(y1)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        index += 1
    # file.write(text_to_write)
    # file.close()
    # maskImg = Image.fromarray(255*mask_gray)
    # maskImg.save(save_path+'mask_'+name)
    color_mask = cv2.merge(mv)
    result = cv2.addWeighted(frame, 0.6, color_mask,0.4, 0)
    result = np.concatenate((image, frame, result),1)
    cv2.imshow('result',result)
    cv2.waitKey()

def get_lightness(src):
	# 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:,:,2].mean()
    
    return  lightness

if __name__ == '__main__':

    main()
    # test_one()