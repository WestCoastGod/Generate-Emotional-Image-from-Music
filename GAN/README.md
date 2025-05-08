emotion_gan/
├── data/ # 存放训练数据
│ ├── train/ # 图片 + VA 标签 csv
│ └── test/
├── models/
│ ├── eru.py # ERU 模块
│ ├── generator.py # 生成器
│ ├── discriminator.py # 判别器
├── utils/
| ├── data_clean.py # 数据預處理
│ ├── dataloader.py # 数据加载
│ └── losses.py # AFM-loss
├── train.py # 训练脚本
├── generate.py # 生成图像脚本
└── requirements.txt

Virtual Environment
conda create -n emotion_gan python=3.8
conda activate emotion_gan
pip install torch torchvision torchaudio matplotlib numpy pillow
