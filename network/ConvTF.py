#coding:utf8
import torch.nn as nn
from torchvision import models
from network import resnet
import torch
from network import functions,transformer
from torch.nn import functional as F
from opts import parse_opts
import os

args = parse_opts()

# python train.py --clip_len 16  --dataset avec --modelName ConvTF --batch_size 8 --lr 1e-4 --pretrained True --GD True
# --GD True 可能会影响结果

if torch.cuda.device_count() > 1:
    print("多卡")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # '0,1'指定GPU编号
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 创建GPU对象
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## 模型定义
class R2D_TF(nn.Module):
    def __init__(self, pretrained=False):#n_class分类类别
        super(R2D_TF, self).__init__()
        self.conv_model = Pretrained_conv(pretrained=pretrained)
        self.cnn_a = functions.Cnn_a(inpt_dim=128, embed_dim=512)#1024

        self.tf_encoder = transformer.Transformer_encoder(d_model=512)#768
        # self.tf_encoder = TF_model_.TF_decoder(d_model=256+256+256)

        # self.tf_decoder = transform_.Transformer_decoder()
        # self.fusion= TF_model_.LMF(512,(512,512,512))

        self.attention = functions.AttentionBlock()#注意力

        self.embedding_v = functions.Embedding(inpt_dim=2048,embed_dim =512)#cat->256
        self.embedding_p = functions.Embedding(inpt_dim=12,embed_dim =512)#128

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 512))
        self.fc1 = nn.Linear(512, 512 // 4, bias=False) # channel, channel // reduction分别为in out bias=False偏置为0  cat-> 768
        self.fc2 = nn.Linear(512 // 4, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # self.Loss = My_HuberLoss()
        # self.Loss_Contrastive = Contrastive_loss()
    def forward(self, samples):#attn_mask
        v = samples['vision'].to(device)
        a = samples['audio'].to(device)
        p = samples['pose'].to(device)

        mask_v = samples['vision_mask'].to(device)

        batch_size, timesteps, channel_x, h_x, w_x = v.shape
        conv_input = v.reshape(batch_size * timesteps, channel_x, h_x, w_x)#
        conv_output = self.conv_model(conv_input)
        v = conv_output.view(batch_size, timesteps, -1)#torch.Size([b, t, 512])
        v = self.embedding_v(v)  # 2048->512

        a = self.cnn_a(a)#
        p = self.embedding_p(p)

        # x = self.attention(v, a, p)
        # x = torch.cat((v, a, p), dim=-1)#cat
        x = a #单模态

        mask_x = mask_v
        x = self.tf_encoder(x, mask_x)

        # x = self.tf_decoder(x, mask_x)
        # x = torch.stack((v, a, p), 1)#([16, 3, 512])
        # x = self.se(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)  # 将张量拉成一维的向量（B，2048）

        # x = self.fusion(v,a,p)

        outputs = self.fc2(self.relu(self.fc1(x)))

        return outputs#,loss,label

## 预训练的ResNet模型
class Pretrained_conv(nn.Module):
    def __init__(self, pretrained=False):
        super(Pretrained_conv, self).__init__()
        # self.conv_model = models.resnet152(pretrained=True)
        #自己模型
        self.conv_model = resnet.resnet50(pretrained=pretrained)
        # ====== 最后的全连接层调整成我们所需要的维度 ======# ResNet网络中没有fc层
        # self.conv_model.fc = nn.Linear(self.conv_model.fc.in_features, latent_dim)#2048->latent_dim(512)

    def forward(self, x):
        # print("x1",x.size())
        x = self.conv_model(x)
        # print("x2", x.size())#[len，2048]
        return x

def get_10x_lr_params(model):
    b = [model.se]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

if __name__ == '__main__':
    pass
    # from torchvision import transforms
    # from PIL import Image
    # import numpy as np
    # data_transform = transforms.Compose([
    #     # transforms.Resize(256),
    #     # transforms.CenterCrop(224),  # 对图片中心进行裁剪
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # import os
    # def get_frames(file_dir):
    #     # load image
    #     frames = os.listdir(file_dir)
    #     frames_ = []
    #     for i in frames:
    #         if i[-3:] == "jpg":
    #             frames_.append(i)
    #
    #     frames_.sort(key=lambda x: int(x[:-4]))  # zw排序，去掉.jpg后缀再排序
    #     framess = [os.path.join(file_dir, img) for img in frames_]
    #     frame_count = len(framess)  ##取得某一个视频对应的所有图片
    #
    #     buffer = np.empty((frame_count, 3 ,224, 224), np.dtype('float32'))
    #     for i, frame_name in enumerate(framess):
    #         print(i,frame_name)
    #         img = Image.open(frame_name).convert('RGB')  # 黑白照片转4通道
    #         img = data_transform(img)
    #         buffer[i] = img
    #         if i == 239:
    #             print(img)
    #     return buffer  # 对应某一个视频所有图片[240,3,224,224]
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # buffer = get_frames('D:/Desktop/ZMT/root/avec_45/dev/0/365_2_Freeform_video')
    # input  = torch.tensor(buffer, dtype=torch.float32).to(device)
    # conv_model = Pretrained_conv(pretrained=True)
    # conv_model.to(device)
    # conv_model.eval()
    # with torch.no_grad():  # 不反向操作
    #     output = conv_model(input.to(device))
    #     print(output.cpu())
    #
    # img = Image.open("D:/Desktop/ZMT/root/avec_45/dev/0/365_2_Freeform_video/0000239.jpg").convert('RGB')  # 黑白照片转4通道
    # img = data_transform(img)
    # img = torch.unsqueeze(img, dim=0)
    # conv_model.eval()
    # with torch.no_grad():  # 不反向操作
    #     output = conv_model(img.to(device))
    #     print(output.cpu())
    #
    # import pickle
    # path_e = 'D:/Desktop/ZMT/root/avec_45/dev/0/365_2_Freeform_video/img_AffectNet_11_embedding.pkl'  # img_AffectNet_embedding.pkl
    # # print("path_e",path_e)
    # with open(path_e, 'rb') as f:
    #     A = pickle.load(f, encoding='bytes')  # read
    #     f.close()
    #
    # print(A[-1])

    # mask = torch.ones(8,2420, dtype=torch.int64)
    # print(input.size())
    # net = Net(inpt_dim = 14,sqe_len =240)
    # out =net(input)
    # print(out.size())


    # conv_model = resnet.resnet50(pretrained=True)
    # # ====== 固定所有卷积层 ======
    # for (name, param) in conv_model.named_parameters():
    #     # if name in to_freeze_dict:
    #     #     param.requires_grad = False
    #     # else:
    #     #     pass
    #     print(name)

    # net = R2D_TF(pretrained=True)
    # input = torch.rand(8,8,3,224,224, dtype=torch.float32)
    # out = net(input,0,0,0,0,0)
    # print(out.size())


    #
    # dims=8
    # k_size=8
    # v_size=8
    #
    # net = AttentionBlock(dims, k_size, v_size)
    # input = torch.rand(2, 4, 8, dtype=torch.float32)
    # v= torch.rand(16,512, dtype=torch.float32)
    # a= torch.rand(16,512, dtype=torch.float32)
    # p= torch.rand(16,512, dtype=torch.float32)
    # x = torch.stack((v, a, p), 1)
    # # c = torch.stack([a, b], 0)
    # print(x.size())
    # # x = nn.AdaptiveAvgPool2d((1,512))(x)
    # x = nn.AdaptiveAvgPool1d(1)(x)##自适应池化层,池化后的每个通道上的大小是一个1x1的 1表示输出的特征图的长度为1
    # print(x.size())
    # x = torch.flatten(x, 1)  # 将张量拉成一维的向量（B，2048）
    # print(x.size())
    # # mask = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]]
    # mask = torch.from_numpy(np.array([[0,1,1,1],[0,0,0,1]]))
    # # mask = mask.data.eq(0).unsqueeze(1)
    # # mask = mask.unsqueeze(1).expand(2, 4, 4)
    # # mask = ex(mask)
    # out = net(input,mask)
    # print(out.size())

    se = functions.SeLayer()
    x = torch.rand(16, 3, 512, dtype=torch.float32)
    x = se(x)
    print(x.size())

    # tensor1 = torch.from_numpy(np.array([[0.25,0.25,0.25,0.25],[1.0,0,0,0]]))
    # tensor2 = torch.from_numpy(np.array([[9.,9.,9.,9],[1,1,1,1]]))
    # # print("t",torch.matmul(tensor1, tensor2))
    # print("tt",tensor1*tensor2)

    # input = np.zeros((27, 3, 224, 224), np.dtype('float32'))
    # print(torch.tensor(input[-1]).size())
    #
    # # 补最后一帧图像
    # b = [input[-1]]
    #
    # for i in range(32-input.shape[0]):
    #     input = np.append(input, b, axis=0)
    #
    # print(torch.tensor(input).size())
