#coding:utf8
import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import torch
import random
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from dataloaders.dataset import VideoDataset
from network import ConvTF
from opts import parse_opts
import numpy as np

args = parse_opts()

#多卡
if torch.cuda.device_count() > 1:
    print("多卡")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # '0,1'指定GPU编号
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 创建GPU对象
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
if args.resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(args.save_dir_root, 'run', 'run_*')))
    b = []
    for r in runs:
        a = int(r.split('_')[-1])
        b.append(a)
    run_id = max(b)
else:
    runs = sorted(glob.glob(os.path.join(args.save_dir_root, 'run', 'run_*')))
    b = []
    for r in runs:
        a = int(r.split('_')[-1])
        b.append(a)
    run_id = max(b) + 1

save_dir = os.path.join(args.save_dir_root, 'run', 'run_' + str(run_id))#保存日志
saveName = args.modelName + '-' + args.dataset
# BDI-II评分在0 - 13之间为最低，在14 - 19之间为轻度，在20 - 28之间为中度，在29 - 63之间为严重。
def Judgment_class(y_pred, y_true):
    y_p = y_pred.tolist()
    y_t = y_true.tolist()
    m = []
    for i in range(len(y_p)):
        if 0 <= y_p[i] <= 13 and 0.0 <= y_t[i] <= 13:
            m.append(True)
        elif 14 <= y_p[i] <= 19 and 14.0 <= y_t[i] <= 19:
            m.append(True)
        elif 20 <= y_p[i] <= 28 and 20.0 <= y_t[i] <= 28:
            m.append(True)
        elif 29 <= y_p[i] <= 63 and 29.0 <= y_t[i] <= 63:
            m.append(True)
        else:
            m.append(False)
    return (torch.tensor(m).to(device))


class My_HuberLoss(nn.Module):
    def __init__(self, delta=5.0):#3.0这个值如何寻找到最优
        super(My_HuberLoss, self).__init__()
        self.delta = delta
        self.d = 2.0########1,2,3,4
    def forward(self, y_pred, y_true):
        residual = torch.abs(y_true - y_pred)
        # print("d",self.d)
        #HuberLoss
        # mask = residual < self.delta#?$delta是一个超参数，用于控制平滑程度,误差较小的情况下使用平方误差，而在误差较大的情况下使用绝对误差
        # loss = torch.where(mask, 0.5 * residual ** 2, self.delta * residual - 0.5 * self.delta ** 2).mean()
        # DRLoss
        mask1 = residual < self.d
        m = Judgment_class(y_pred, y_true)########DRLoss
        mask1 = mask1&m#满足同一类别且误差较小########DRLoss
        # print(mask1)
        mask2 = residual < self.delta
        '''
        1.当输入参数为三个时，即torch.where(condition, x, y)，返回满足 x if condition else y的tensor，注意x,y必须为tensor
        2.当输入参数为一个时，即torch.where(condition)，返回满足condition的tensor索引的元组(tuple)
        '''
        loss1 = torch.where(mask1, 0.1*0.5 * residual ** 2, 0.5 * residual ** 2)#乘以一个0.1权重
        loss1.masked_fill_(~mask2, 0)
        loss2 = torch.where(mask2, 0, self.delta * residual - 0.5 * self.delta ** 2)
        loss = (loss1 + loss2).mean()

        return loss


def train_model(dataset=args.dataset, save_dir=save_dir, lr=args.lr,num_epochs=args.nEpochs, save_epoch=args.snapshot, useTest=args.useTest, test_interval=args.nTestInterval):
    if args.modelName == 'ConvTF':
        model = ConvTF.R2D_TF(pretrained=args.pretrained)
        # train_params = [{'params': ConvTF.get_10x_lr_params(model), 'lr': lr}]
    elif args.modelName == 'LSTM':
        model = ConvTF.R2D_Lstm(hidden_size=512, lstm_layers=2, bidirectional=False,pretrained=args.pretrained)#latent_dimLSTM输入维度  hidden_size 单向512 双向512*2
        # train_params = model.parameters()
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError

    #添加多卡
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids=[0, 1],output_device=0)
        print("Lets use", torch.cuda.device_count(), "GPUs!")

    Loss = My_HuberLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    if args.resume_epoch == 0:
        print("Training {} from scratch...".format(args.modelName))
        # checkpoint = torch.load("./root/ConvTF-avec_epoch_589.pth.tar")#huberloss
        
        # model_dict = checkpoint['state_dict']
        # s_dict = model.state_dict()
        # for name in s_dict:
        #     if name not in model_dict or 'cnn_a' in name or 'embedding_p' in name:
        #         print(name, "预训练模型没有这个网络层")  # 预训练模型有sa ca层
        #         continue
        #     s_dict[name] = model_dict[name]
        # model.load_state_dict(s_dict)
        
        # model.load_state_dict(checkpoint['state_dict'])

    else:
        checkpoint = torch.load(os.path.join(save_dir, saveName + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(os.path.join(save_dir,args.modelName, saveName + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar')))
        if torch.cuda.device_count() > 1:#多卡训练的模型
            model_dict = checkpoint['state_dict']
            s_dict = model.state_dict()
            for name in s_dict:
                name_ = '.'.join(name.split(".")[1:]) #module.conv_model.conv_model.conv1.weight
                if name_ not in model_dict:  # ResNet网络中没有fc层
                    print(name_, "预训练模型没有这个网络层")  # 预训练模型有sa ca层
                    continue
                s_dict[name] = model_dict[name_]
            model.load_state_dict(s_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['opt_dict'])#优化器在CPU上
        # 增加以下几行代码，将optimizer里的tensor数据全部转到GPU上 zw
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    print('模型总的参数量: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('模型训练的参数量: %.2fM' % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))

    Loss.to(device)
    model.to(device)

    log_dir = os.path.join(save_dir,args.modelName, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))

    train_dataset = VideoDataset(dataset=dataset, split='train', clip_len=args.clip_len)
    val_dataset = VideoDataset(dataset=dataset, split='dev',  clip_len=args.clip_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(val_dataset, batch_size=13, shuffle=True, num_workers=4)#drop_last=True不足的batchsize丢掉
    # test_dataloader  = val_dataloader

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    test_size = len(val_dataset)

    H = 20#模型性能好于基准才保存
    for epoch in range(args.resume_epoch, num_epochs):
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()
            running_loss = 0.0
            rmse = 0.0
            mae = 0.0
            if phase == 'train':
                model.train()
            else:
                model.eval()
    ##################################################          TF
            for samples in tqdm(trainval_loaders[phase]):#################################if args.modelName == 'TF':

                if args.modelName == 'LSTM':
                    if torch.cuda.device_count() > 1:
                        model.module.Lstm.reset_hidden_state()
                    else:
                        model.Lstm.reset_hidden_state()  ###单卡

                labels = samples['label'].to(device)

                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(samples)
                else:
                    with torch.no_grad():
                        outputs = model(samples)

                #######################################
                #回归
                outputs = torch.squeeze(outputs, 1)
                loss = Loss(outputs, labels)

                # 计算RMSE
                rmse += torch.sqrt(torch.mean((outputs - labels) ** 2)).item() * len(outputs)
                # 计算MAE
                mae += torch.mean(torch.abs(outputs - labels)).item() * len(outputs)

                if phase == 'train':
                    loss.backward()#?????????????retain_graph=True 在.backward()调用过程中，所有中间结果在不再需要时都会被删除
                    optimizer.step()

                running_loss += loss.item()*len(outputs) #要么乘 那么用'sum'  得每个epoch的loss值之和

            epoch_loss = running_loss / trainval_sizes[phase]#trainval_sizes[phase]训练数据之和200

            print('RMSE:', rmse / trainval_sizes[phase])#每个epoch总差值
            print('MAE:', mae / trainval_sizes[phase])

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_RMSE',rmse / trainval_sizes[phase], epoch)
                writer.add_scalar('data/train_MAE', mae / trainval_sizes[phase], epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_RMSE',rmse / trainval_sizes[phase], epoch)
                writer.add_scalar('data/val_MAE', mae / trainval_sizes[phase], epoch)

            print("[{}] Epoch: {}/{} Loss: {}".format(phase, epoch + 1, args.nEpochs, epoch_loss))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        #保存
        if epoch > 200:
            if epoch % save_epoch == (save_epoch - 1) or (rmse / trainval_sizes[phase] + mae / trainval_sizes[phase]) < H:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar'))
                print("Save model at {}\n".format(os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if rmse / trainval_sizes[phase] + mae / trainval_sizes[phase] < H:
            H = rmse / trainval_sizes[phase] + mae / trainval_sizes[phase]

        if phase == 'train':
            scheduler.step()

def val_predict(dataset=args.dataset):#python train.py --clip_len 16  --dataset avec  --batch_size 16
    model = ConvTF.R2D_TF(pretrained=False)
    # checkpoint = torch.load("D:/Desktop/MDDR论文/ZMT/root/run/run_588/ConvTF-avec_epoch-65.pth.tar")#
    checkpoint = torch.load("/root/ZMT/ConvTF-avec_epoch-50.pth.tar")#huberloss
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    model.eval()

    val_dataset = VideoDataset(dataset=dataset, split='dev',  clip_len=args.clip_len)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4,drop_last=True)#drop_last=True不足的batchsize丢掉
    test_size = len(val_dataset)

    start_time = timeit.default_timer()

    rmse =0.0
    mae = 0.0
    # y =[]
    p =[]

    for samples in tqdm(val_dataloader):#inputs, labels, attn_mask_A, attn_mask_B , B
        labels = samples['label'].to(device)
        with torch.no_grad():
            outputs = model(samples)#inputs,attn_mask_A

        # 回归
        outputs = torch.squeeze(outputs, 1)

        # y_pred = outputs.detach().cpu().numpy()
        # labels_ = labels.cpu().numpy()
        ################################
        # print("preds", y_pred)
        # print("labels", labels_)
        # p.extend(y_pred)
        # y.extend(labels_)

        rmse += torch.sqrt(torch.mean((outputs - labels) ** 2)).item() *labels.size(0)
        mae += torch.mean(torch.abs(outputs - labels)).item() * labels.size(0)


    # f = open("p.txt", "w")
    # f.writelines(str(p))
    # f.close()

    # f = open("y.txt", "w")
    # f.writelines(str(y))
    # f.close()

    print('RMSE:', rmse / test_size )
    print('MAE:', mae / test_size )

    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

    #"batch_size=6   7.0,6.0"
    # "batch_size=8   7.32,5.93"
    #"batch_size=13   7.16,5.63"
    # "batch_size=15   6.96,5.34"
    # "batch_size=17   6.40,4.96"
    # "batch_size=18   7.2,5.48"
    # "batch_size=21   6.22,4.72"
    # "batch_size=22   7.0,5.34"
    # "batch_size=23   7.28,5.53"
    # "batch_size=26   5.84 4.50"
    # "batch_size=27   6.0 4.62" = 28  29(6.88,5.25)

if __name__ == "__main__":
    train_model()
    # val_predict()


