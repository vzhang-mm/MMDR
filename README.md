## Paper: MDDR: Multi-modal Dual-Attention aggregation for Depression Recognition

## 🔗 Resources

### 1. Pretrained Facial Feature Extractor
- `resNet50_AffectNet.pth`: [Download Link](https://example.com/download/resNet50_AffectNet.pth)  

### 2. Preprocessed AVEC Datasets
- **AVEC2013 & AVEC2014(processed)**: [Download](https://pan.baidu.com/s/1IGo1cC9IjR2iTyYAOtyS8A?pwd=nw2t), code: nw2t 

## 🛠️ Requirements

```
torch>=1.12.0
torchaudio
opencv-python
scikit-learn
tqdm
pandas
numpy
```

## 🚀 Training

To train the model on the AVEC2014 dataset using the `MDDR` architecture:

```bash
python train.py --modelName ConvTF --dataset avec2014
```

Additional arguments:

* `--lr`: learning rate (default: 1e-4)
* `--batch_size`: training batch size (default: 32)
* `--epochs`: number of training epochs (default: 50)

## 📁 Directory Structure

```
├── data/
│   ├── avec2013/
│   └── avec2014/
├── dataloaders/
│   ├── dataset.py
├── network/
│   ├── ConvTF.py
│   ├── functions.py
│   ├── resnet.py
│   ├── transformer.py
├── root/
│   ├── run/
├── mypath.py
├── opts.py
├── train.py
└── resNet50_AffectNet.pth
```

## 📞 Contact

For questions or access to datasets, please contact: \[zhangwei_self@qq.com]

