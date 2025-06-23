#coding:utf8
import os
import torch
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
from mypath import Path
from opts import parse_opts
args = parse_opts()
import pickle
import csv
from torchvision import transforms
from PIL import Image
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # 对图片中心进行裁剪
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
## 视频数据集类定义
class VideoDataset(Dataset):
    def __init__(self, dataset='ucf101', split='train', clip_len=16):#clip_len每一个视频片段的长度
        self.root_dir = Path.db_dir(dataset)
        folder = os.path.join(self.root_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.dataset = dataset

        if not self.check_integrity():#对要处理的视频路径检查
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        ## 遍历所有的图片文件夹
        self.fnames, self.labels = [], [] ##存储所有的文件夹及其标签
        ## folder：大类文件夹
        ## fnames：每一个视频对应的文件夹，存储所有数据，标签等于其上层目录的label
        for label in os.listdir(folder):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                self.labels.append(int(label))

        assert len(self.labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # 直接用类别名做label
        # self.label_array = np.array(labels)

        # self.resize_height = 224  # 128
        # self.resize_width = 224  # 171
        self.crop_size = 224  # 112

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # num_items = len(self.labels)#创造正负样本
        # while 1:
        #     index_s = random.randrange(num_items)
        #     label_s = self.labels[index_s]
        #     if abs(label_s - self.labels[index]) <= 2:
        #         break
        # while 1:
        #     index_d = random.randrange(num_items)
        #     label_d = self.labels[index_d]
        #     if abs(label_d - self.labels[index]) >= 15:
        #         break
        samples = self.load_samples(index)
        # samples_s = self.load_samples(index_s)
        # samples_d = self.load_samples(index_d)
        # print(self.labels[index],label_s,label_d)#
        # print(samples)
        return samples


    def load_samples(self,index):
        labels = np.array(self.labels[index])

        buffer = self.get_frames(self.fnames[index])  # 所有图片
        buffer, attn_mask_A, time_index = self.crop(buffer, self.clip_len, self.crop_size,self.fnames[index])  # 选取一定数量图片，如果不够报错

        # print('time_index',time_index)

        # if self.split == 'dev':#随机翻转
        #     buffer = self.randomflip(buffer)
        # tensor可以增加随机翻转，代码自己增加
        # print(torch.from_numpy(buffer).size())#torch.Size([3, 16, 172, 172])
        ####head pose
        path_p = self.fnames[index] + '/point.csv'#point.csv point_norm.csv head_pose
        p =self.get_p(path_p,time_index)
        # print('p',np.array(p).shape)
        # print(path_p)

        ####audio
        path_a = self.fnames[index] + '/mel.pkl'
        a = self.get_a(path_a, time_index)
        # print('a', np.array(a).shape)

        return {'vision':torch.from_numpy(buffer),
                'audio':torch.from_numpy(np.array(a).astype(np.float32)),
                'pose':torch.from_numpy(np.array(p).astype(np.float32)),
                'label':torch.from_numpy(labels),
                'vision_mask': torch.from_numpy(attn_mask_A.astype(np.int64))}

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def get_frames(self, file_dir):
        # load image
        frames = os.listdir(file_dir)
        frames_ = []
        for i in frames:
            if i[-3:] == "jpg":
                frames_.append(i)

        frames_.sort(key=lambda x: int(x[:-4]))  # zw排序，去掉.jpg后缀再排序
        framess = [os.path.join(file_dir, img) for img in frames_]
        frame_count = len(framess)  ##取得某一个视频对应的所有图片

        buffer = np.empty((frame_count, 3 ,224, 224), np.dtype('float32'))
        for i, frame_name in enumerate(framess):
            img = Image.open(frame_name).convert('RGB')  # 黑白照片转4通道
            img = data_transform(img)##标准化处理
            buffer[i] = img
        return buffer  # 对应某一个视频所有图片[240,3,224,224]


    def crop(self, buffer, clip_len, crop_size, fnames_index):#clip_len抽16帧， crop_size裁剪尺寸
        ## 随机选择时间切片参数
        if (buffer.shape[0] < clip_len):#
            print("该视频没有足够的帧数可供选择",fnames_index)
            # print('buffer',len(buffer))
            attn_mask0 = np.ones(len(buffer))
            # print('attn_mask0',len(attn_mask0))
            attn_mask1 = np.zeros(self.clip_len - len(buffer))#0被遮掩
            mask = np.append(attn_mask0, attn_mask1)

            time_index = 0
            # print('buffer',torch.tensor(buffer).size())#[27, 3, 224, 224])
            #补0
            b = np.zeros((clip_len-buffer.shape[0], 3, 224, 224), np.dtype('float32'))
            # print('b', torch.tensor(b).size())  # [27, 3, 224, 224])
            buffer = np.append(buffer, b, axis=0)
            # 补最后一帧图像
            # b = [buffer[-1]]
            # for i in range(clip_len-buffer.shape[0]):
            #     buffer = np.append(buffer, b, axis=0)
        else:
            # print(buffer.shape[0])
            time_index = np.random.randint(buffer.shape[0] - clip_len+1)#47-16
            # time_index = 0#不随机选，从0开始，保障数据一致性
            mask = np.ones(self.clip_len)

        ## 随机选择空间裁剪参数
        # height_index = np.random.randint(buffer.shape[1] - crop_size)
        # width_index = np.random.randint(buffer.shape[2] - crop_size)

        # buffer = buffer[time_index:time_index + clip_len,#时间偏移量、空间偏移量
        #          height_index:height_index + crop_size,
        #          width_index:width_index + crop_size, :]
        buffer = buffer[time_index:time_index + clip_len,:, :, :]#时间偏移量
        return buffer,mask,time_index

    def get_p(self, filename,time_index):
        P = []
        with open(filename, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)
            # 遍历csvreader对象的每一行内容并输出
            for row in csvreader:
                P.append(row)
        if (len(P) > self.clip_len):
            P = P[time_index:time_index + self.clip_len]
        else:
            for i in range(self.clip_len - len(P)):
                P.append([0.0] * 12)

        return P


    def get_a(self, filename, time_index):
        
        with open(filename, 'rb') as f:
            B = pickle.load(f, encoding='bytes')  # 读取 (B,N,D)
        #判断
        if time_index+self.clip_len > B.shape[0]:
            print(B.shape,time_index)
            time_index = time_index -1

        if (B.shape[0] > self.clip_len):  # 大于512就随机选取
            B = B[time_index:time_index+self.clip_len,:,:]
        else:
            padding_needed = self.clip_len - B.shape[0]
            B = np.pad(B, ((0, padding_needed), (0, 0), (0, 0)), mode='constant', constant_values=0.0)

        return B


    def randomflip(self, buffer):
        if random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)#对图像随机水平翻转
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    val_dataloader = DataLoader(VideoDataset(dataset='avec', split='train', clip_len=16), batch_size=4, num_workers=1)
    # print(val_dataloader)
    for i, sample in enumerate(val_dataloader):
        print(i)
        vision = sample['vision']
        labels = sample['label']

