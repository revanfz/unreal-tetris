# UNREAL TETRIS
Repositori ini merupakan implementasi dari Tugas Akhir dengan judul **IMPLEMENTASI ARSITEKTUR _UNSUPERVISED REINFORCEMENT WITH AUXILIARY LEARNING_ (UNREAL) UNTUK MENGEMBANGKAN AGEN CERDAS (STUDI KASUS: NES TETRIS)**

oleh

[Revan Fauzi Algifari](https://www.github.com/revanfz); 120140049

## Latar Belakang
- Keberhasilan penerapan DRL dalam menghasilkan kebijakan yang mendekati optimal pada _video game_ [1]
- _Real world scenario_ yang memiliki desain _reward_ yang bersifat _delayed_ & _sparse_ memperlambat pembelajaran menggunakan teknik DRL [2]
- Kompleksitas Tetris, memiliki _state space_ yang besar dan membutuhkan strategi _long-term planning_ [3][4]

## Desain _Neural Network_
### _Shared Network_ (CNN & LSTM)
Shared network merupakan lapisan bersama yang digunakan antar _network_ untuk mengolah input data

![Shared Network](https://github.com/revanfz/unreal-tetris/blob/main/img/Shared%20Network.png?raw=true)

### _Actor-Critic Network_
_Actor-Critic_ menghasilkan luaran berupa probabilitas setiap aksi dan nilai _value_

![_Actor Critic Network_](https://github.com/revanfz/unreal-tetris/blob/main/img/AC%20Network.png?raw=true)

### _Pixel Control Network_
Jaringan ini mengolah luaran LSTM dengn proses dekonvolusi untuk merekonstruksi gambar observasi dan memaksimalkan nilai pergantian intensitas piksel
![_Pixel Control Network_](https://github.com/revanfz/unreal-tetris/blob/main/img/PC%20Network.png?raw=true)

### _Reward Prediction Network_
Jaringan ini bertugas untuk memprediksi _reward_ pada frame berikutnya saat diberikan 3 urutan frame observasi
![_Reward Prediction Network_](https://github.com/revanfz/unreal-tetris/blob/main/img/RP%20Network.png?raw=true)

## Installation
Clone proyek

```bash
  git clone https://github.com/revanfz/unreal-tetris.git
```

Masuk ke direktori proyek
```bash
  cd unreal-tetris
```

Buat environment _python_ berdasarkan file ```environment.yml```
```bash
  conda env create -f environment.yml
```

## How to use
```bash
usage: main.py 
               Implementation model UNREAL:
               IMPLEMENTASI ARSITEKTUR UNSUPERVISED REINFORCEMENT WITH AUXILIARY LEARNING (UNREAL)
               UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)

               [-h] [--lr LR] [--gamma GAMMA] [--beta BETA] [--pc-weight PC_WEIGHT] [--grad-norm GRAD_NORM]
               [--unroll-steps UNROLL_STEPS] [--save-interval SAVE_INTERVAL] [--max-steps MAX_STEPS]
               [--hidden-size HIDDEN_SIZE] [--optimizer OPTIMIZER] [--num-agents NUM_AGENTS] [--log-path LOG_PATH]
               [--model-path MODEL_PATH] [--resume-training RESUME_TRAINING]

options:
  -h, --help                        show this help message and exit
  --lr LR                           Learning rate
  --gamma GAMMA                     discount factor for rewards
  --beta BETA                       entropy coefficient
  --pc-weight PC_WEIGHT             pixel control loss weight
  --grad-norm GRAD_NORM             Gradient norm clipping
  --unroll-steps UNROLL_STEPS       jumlah step sebelum mengupdate parameter global
  --save-interval SAVE_INTERVAL     jumlah episode sebelum menyimpan checkpoint model
  --max-steps MAX_STEPS             Maksimal step pelatihan
  --hidden-size HIDDEN_SIZE         Jumlah hidden size
  --optimizer OPTIMIZER             optimizer yang digunakan
  --num-agents NUM_AGENTS           Jumlah agen yang berjalan secara asinkron
  --log-path LOG_PATH               direktori plotting tensorboard
  --model-path MODEL_PATH           direktori penyimpanan model hasil training
  --resume-training RESUME_TRAINING Load weight from previous trained stage
```

## Tabel _Hyperparameter_
Parameter| Value
:-----------------------:|:-----------------------:|
lr (&alpha;)			            | 0.00012
entropy coefficient (&beta;)        | 0.00318
discount factor (&gamma;)           | 0.95
pixel control weight (&lambda;pc)   | 0.05478
t (rollout length)                  | 20
per agent replay buffer size        | 2000
num agent                           | 4
critic loss coefficient             | 0.5
max grad norm                       | 40
optimizer                           | RMSProp
max step                            | 1e7

## Reference
[1] [G. N. Yannakakis, “AI and Games: The Remarkable Case of Malta,” Xjenza Online, vol. 11, no. Special Issue, pp. 59–66, 2023, doi: 10.7423/XJENZA.2023.1.07](https://www.xjenza.org/ISSUES/8/07.pdf)

[2] [M. Jaderberg et al., “REINFORCEMENT LEARNING WITH UNSUPERVISED AUXILIARY TASKS,” 2017.](https://arxiv.org/abs/1611.05397)

[3] [S. Algorta and Ö. Simsek, “The Game of Tetris in Machine Learning”. 2019.](https://arxiv.org/abs/1905.01652)

[4] [Z. Chen, “Playing Tetris with Deep Reinforcement Learning,” 2021.](https://www.ideals.illinois.edu/items/118525)

## Acknowledgement
1. [UNREAL](https://github.com/miyosuda/unreal/tree/master) by [@miyosuda](https://github.com/miyosuda)
2. [UNREAL](https://github.com/voiler/unreal) by [@voiler](https://github.com/voiler)

## License

[MIT](https://github.com/revanfz/unreal-tetris/blob/main/LICENSE)