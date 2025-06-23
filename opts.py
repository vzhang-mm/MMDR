import argparse
from pathlib import Path

def parse_opts():
    parser = argparse.ArgumentParser(description='Action Recognition')
    parser.add_argument('--nEpochs', default=100, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default:32)')
    parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate (default:5e-4')
    parser.add_argument('--num_classes', default=101, type=int, help='The number of classes we would train on')
    parser.add_argument('--dataset', default='dfew', type=str, help='训练数据名称')
    parser.add_argument('--resume_epoch', default=0, type=int, help='从多个个Epoch后接着训练')
    parser.add_argument('--nTestInterval', default=1000, type=int, help='Run on test set every nTestInterval epochs')
    parser.add_argument('--snapshot', default=20, type=int, help='多少个Epoch后保存模型')
    parser.add_argument('--useTest', default=True, type=bool, help='See evolution of the test set when training')
    parser.add_argument('--clip_len', default=16, type=int, help='视频抽多少帧')
    parser.add_argument('--clip_video_len', default=32, type=int, help='长视频切多少片')

    parser.add_argument('--pretrained', default=False, type=bool, help='是否加载预训练模型')
    parser.add_argument('--checkpoint_path', default=r"./root/R3D-dfew.pth.tar", type=str, help='预训练模型所在路径')
    parser.add_argument('--AT', default=True, type=bool, help='在视频片段特征提取阶段增加注意力机制')
    parser.add_argument('--DB', default=True, type=bool, help='增加DropBlock')
    # parser.add_argument('--RD', default=False, type=bool, help='打乱图片帧时序')
    parser.add_argument('--GD', default=False, type=bool, help='固定图像层参数')
    parser.add_argument('--root_dir', default=r'./root/', type=str, help='视频所在文件夹')
    parser.add_argument('--out_dir', default=r'./root/AVEC2014_clips_results', type=str, help='抽帧后图片，也就是训练图片数据所在文件夹')
    parser.add_argument('--save_dir_root', default=r'./root', type=str, help='保存日志，训练结果')
    parser.add_argument('--modelName', default='R3D', type=str, help='使用的模型')

    parser.add_argument('--num_workers', default=1, type=int,
                        help='initial num_workers, the number of processes that generate batches in parallel (default:4)')
    parser.add_argument('--split_size', default=0.2, type=int, help='set the size of the split size between validation '
                                                                    'data and train data')
    parser.add_argument('--seed', default=42, type=int,
                        help='initializes the pseudorandom number generator on the same number (default:42)')
    parser.add_argument('--latent_dim', default=512, type=int, help='The dim of the Conv FC output (default:512)')
    parser.add_argument('--hidden_size', default=256, type=int,
                        help="The number of features in the LSTM hidden state (default:256)")
    parser.add_argument('--lstm_layers', default=2, type=int, help='Number of recurrent layers (default:2)')
    parser.add_argument('--bidirectional', default=True, type=bool,
                        help='set the LSTM to be bidirectional (default:True)')

    args = parser.parse_args()

    return args

# python train.py --modelName ConvTF --dataset avec2014