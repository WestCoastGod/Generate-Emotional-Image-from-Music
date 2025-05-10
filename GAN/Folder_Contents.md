emotion_gan/
├── data/ # Stores training data
│ ├── train/ # Images + VA label CSV
│ └── test/
├── models/
│ ├── eru.py # ERU module
│ ├── generator.py # Generator
│ ├── discriminator.py # Discriminator
├── utils/
│ ├── data_clean.py # Data preprocessing
│ ├── dataloader.py # Data loading
│ └── losses.py # AFM-loss
├── train.py # Training script
├── generate.py # Image generation script
└── requirements.txt
