# 2021 AI Challenge   
DNN-based Speech Enhancement in the frequency domain  

# 팀 소개
주제 : Data Augmentation을 통한 DNN 기반 음성 향상


# 목차
1. 연구 소개
2. 실험 환경
3. 모델 별 성능 측정
4. 네트워크 모델 구조 변형
5. 다양한 데이터 증강 기법
6. 데이터 증강을 통한 성능 측정
7. 결론

# 1. 연구 소개
> Speech Enhancement란?  
* 음성향상기술은 잡음이 섞인 음성데이터에서 잡음 음성만을 분리하는 기술이다.  
최근 딥러닝을 이용한 음성 향상 기술은 높은 성과를 보이고 있다.  

* 응용 분야로는 음성인식, 화상회의, 보청기 등 많은 곳에 쓰인다.  
> 연구 목적
1. 제공된 음성 데이터에 대한 최적의 모델 탐색
2. 모델 구조 변형을 통한 성능 비교
3. 데이터 증강을 통한 성능 향상

# 2. 실험 환경
> **Setting**  
* CPU : i7-11700 2.50GHz  
* GPU : GeForce GTX 1080 Ti  
* RAM : 16GB   
* Framework : Pytorch  
* Optimizer : Adam 
* Network : DCCRN 
* Loss function : SI-SNR  
* Learning rate : 0.001  
* Batch size : 10  
* SNR(dB) : [5, 10, 15]  
* Learning method : T-F masking  
   

> **Dataset**  
* DNS Challenge Dataset  
Clean Data :  1680개    
Noise Data : 298개  

* Noisy Data :   
(Noisy Data : Noise를 입힌 Clean 데이터)  
Clean과 Noise 데이터의 Train/Validation 비율을 3:1로 고정  
즉, Train : Validation = 1120 : 560  
SNR 5,10,15dB 환경으로 Noisy 데이터 생성 : 3360개  
Augmentation (Data Shifting, Minus, Reverse)을 통한 Noisy 데이터 생성 : 각 기법 당 3360개   
**총 Noisy Data : 13440개**  
* Deep Noise Suppression (DNS) Challenge 4 - ICASSP 2022 [[link]](https://github.com/microsoft/DNS-Challenge)  

> **Evaluation metrics**
* PESQ (Perceptual Evaluation of Speech Quality)  


# 3. 모델 별 성능 측정 및 비교  
모든 실험은 Loss가 적절히 수렴했다고 판단되었음을 가정한다.  
  
> Network에 따른 성능 비교  
* DCCRN (Blue)
* FullSubNet (Orange)
![image](https://user-images.githubusercontent.com/87358781/146786508-cb499d8d-9ec0-4d60-a64a-93fad7f53d09.png)  
Noisy Data 3360개를 동일한 실험 조건으로 Network만 변경하여 실험한다.  
실험 결과, DCCRN에서는 PESQ가 2.140, FullSubNet에서는 2.112으로 DCCRN인 경우 더 나은 성능을 보인다.  
**Network 모델 베이스라인을 DCCRN으로 선정한다.**

> Loss에 따른 성능 비교  

|Loss|MSE|SI-SNR|SDR|SI-SDR|
|:---:|:---:|:---:|:---:|:---:|
|PESQ|1.925|1.984|1.942|1.743|

* Noisy Data 1120개, Network 모델은 DCCRN을 기준으로 손실함수만 변경하여 실험한다.  
실험 결과, 손실함수(Loss)가 SI-SNR인 경우 PESQ 1.984로 가장 좋은 성능을 보인다.  
**손실 함수 베이스라인을 SI-SNR로 선정한다.**  

> Perceptual Loss 추가에 따른 성능 비교
* 추가하지 않음 (Blue) 
* LMS 추가 (Gray)
* PMSQE 추가 (Red)  
Perceptual Loss는 지각적 성능 향상을 목적으로 한다.  
![image](https://user-images.githubusercontent.com/87358781/146793078-2490b53f-2791-417c-8057-0f9b337bb94f.png)    
Data 3360개, DCCRN, SI-SNR을 기준으로 Perceptual Loss를 추가하였을 때,  
PESQ는 PMSQE일 때 2.162, LMS는 2.153으로 증가함을 볼 수 있으며, PMSQE에서 가장 큰 상승을 보인다.  
더불어, PMSQ 수치 뿐 아니라 사람이 실제로 느끼는 음성의 질이 매우 좋아짐을 볼 수 있었다. 
**Perceptual Loss는 지각적 성능 뿐 아니라 PESQ 수치도 향상시킨다.**

# 4. 네트워크 모델 구조 변형  
CRN(Convolutional Recurrent Neural Network) 구조는 다음과 같다.
![CRN Structure](https://user-images.githubusercontent.com/87358781/146889875-397d6e1f-4fef-4751-856f-fa7113c7f2d8.png)  
Encoder, Decoder를 보면 6개 층의 Convolution Layer와 2개의 LSTM으로 구성된다.  
위와 같은 네트워크 모델 구조를 변형하여 성능을 향상시키고자 한다.  
1. LSTM 구조 변형 
> LSTM 개수 변화에 따른 성능 비교  
* 기존 (2개)
* 기존 -1 (1개)
* 기존 +1 (3개)
* 기존 +2 (4개)  
DCCRN의 구조를 보면 LSTM 2개가 존재한다. LSTM의 수를 변화시켰을 때 성능 비교는 다음과 같다.  
![image](https://user-images.githubusercontent.com/87358781/146795418-00951943-4ada-49a7-8c6a-5eab0aecb33a.png)  
베이스라인으로 Data 3360개의 데이터를 실험한 결과, 성능 차이는 미미했고 기존 모델인 LSTM이 2개일 때 가장 좋은 성능을 보인다.  
**LSTM 개수를 변화 시킬 경우 성능이 저하된다.**  

1. Convolution Layer 구조 변형  
> Convolution Layer 개수 변화에 따른 성능 비교  
* 기존 (6층)  
* 기존 -1 (5층)  
* 기존 +1 (7층)  
* 기존 +2 (8층)    
DCCRN의 구조를 보면 Convolution Layer가 6층으로 구성된다. Convolution Layer의 개수를 변화시켰을 때 성능 비교는 다음과 같다.  
![image](https://user-images.githubusercontent.com/87358781/146915540-9b9d3fca-7adb-4f23-8740-ed0eff790b3b.png)  
베이스라인으로 Data 3360개의 데이터를 실험한 결과, 성능 차이는 미미했고 기존 모델인 Convolution이 6개일 때 가장 좋은 성능을 보인다.  
**Convolution Layer 개수를 변화 시킬 경우 성능이 저하된다.**  

# 5. 데이터 증강 기법을 통한 성능 향상
> Data Augmentation이란?
* 원본 데이터에 인위적인 변화를 주어 기존과는 다른 데이터를 형성하는 법을 말한다.  
* 사람에게는 같은 데이터로 보일 수 있지만 인공지능에게는 매우 다른 데이터가 되어 학습에 도움을 줄 수 있다.  

1. Shifting the sound  
* 음성 데이터의 시퀀스를 shift하는 방법
* 순서가 바뀌어 들리게 된다.
기존 데이터 3360개에 Shifting한 데이터 3360개를 더한 7720개 데이터셋 생성  
![image](https://user-images.githubusercontent.com/87358781/146796910-d8d009aa-24af-4bc9-bd7c-db7f20b7038f.png)  
결과적으로 데이터 증강 전에는 PESQ가 2.238, 증강 후에는 2.490로 대폭 향상됨을 볼 수 있다.

2. Reverse the sound
* 음성 데이터를 거꾸로 변환하는 방법
* 음성이 거꾸로 들리게 된다.  
기존 데이터 3360개에 Reverse한 데이터 3360개를 더한 7720개 데이터셋 생성  
![image](https://user-images.githubusercontent.com/87358781/146798422-d40fba4f-a365-4692-a4db-6b2a742b0bb9.png)  
결과적으로 Shifting보다는 저조했지만, PESQ가 2.238에서 2.462로 대폭 향상됨을 볼 수 있다.  

3. Minus the sound 
* 음성 데이터의 위상을 뒤집는 방법
* 사람이 듣기에는 원본 데이터와 동일하게 들린다.  
기존 데이터 3360개에 Reverse한 데이터 3360개를 더한 7720개 데이터셋 생성
![image](https://user-images.githubusercontent.com/87358781/146798834-efc600ac-53bb-4627-bc71-82bc6d1ea9c4.png)  
결과적으로 Shifting보다는 저조했지만, PESQ가 2.238에서 2.451로 대폭 향상됨을 볼 수 있다.  
* **세 가지 데이터 증강 기법을 비교하면 Shifting, Minus, Reverse 순서로 성능이 향상되며, 세 가지 모두 기존 성능보다 대폭 향상된다.**  

# 7. 결론
> 전체 Augmentation 데이터를 통한 성능 향상 비교 
* 원본 데이터 3360개, Shifting/Reverse/Minus를 통한 증강 데이터 10080개 데이터를 결합  
 (부족한 메모리로 인해 결합에 어려움을 겪었지만 Swap Memory를 증가시켜 해결함)
* 총 13440개 데이터셋으로 실험  
![image](https://user-images.githubusercontent.com/87358781/146916556-1484d0ec-1482-4da1-be91-c81544ccef62.png)  
Augmentation 데이터를 포함한 13440개 데이터셋의 실험 결과 기존 PESQ 2.238에서 2.549로 대폭 향상됨을 볼 수 있다.  

> 연구 성과
1. 제공된 데이터셋의 경우 네트워크로는 DCCRN, 손실함수로는 SI-SNR의 조합이 가장 큰 성능을 보임.  
2. Perceptual Loss를 추가하는 것은 사람이 실제로 느끼는 음질의 향상 뿐 아니라 음성 평가 지표인 PESQ 수치에서도 향상을 보인다.  
3. DCCRN 네트워크 구조의 LSTM, Convolution Layer의 수를 변형시키는 것은 성능 향상에 도움을 주지 않는다.  
4. 데이터 Augmentation은 한정된 데이터를 증가시킬 수 있으며, 성능 향상에 많은 도움을 준다. 
 (데이터가 증가할수록 메모리의 부족으로 오류가 발생한다. SWAP Memory를 통해 해결할 수 있지만, 속도가 매우 저하된다.)  

>  추가 연구 계획
1. 추가적인 Augmentation 기법으로 실험을 하여 데이터 셋을 증가시킨다.  
2. FullSubNet 네트워크 구조를 변형하여 실험하고 기존 모델과 비교한다.  
3. 데이터셋 증가에 따른 메모리 부족을 해결할 수 있는 방안을 모색한다.

# Manual
## Step 1
> 1. 데이터셋 디렉토리 생성  
> (1) train/clean, train/noise 디렉토리 생성  
> (2) validation/clean, validation/noise 디렉토리 생성  
> 디렉토리 생성 후 wav 파일을 각각 넣으면 된다.  
> 단, train, validation의 clean 데이터는 사전에 지정한 비율로 나누어 할당해야한다.  
> 예시)  
./Dataset/train/clean  
./Dataset/train/noise  
./Dataset/validation  
./Dataset/validation/clean  
여기서 validation에 noise가 없는 이유는 validation 수행 시 noise를 train 때 썼던 noise를 사용하기 때문이다.  
  
> 2. Dataset 훈련, 검증 데이터 분할하기  
clean 데이터는 1680개로 구성된다.  
train : validation = 9:1 = 1512 : 168  
train : validation = 8:2 = 1344 : 336   
train : validation = 7:3 = 1176 : 505  
train : validation = 3:1 = 1120 : 560
본 실험에서는 각 비율에 대한 성능을 비교하였으며 성능이 가장 좋은 train : validation = 3:1로 채택한다.    
여기서 유의해야할 점은 train에 넣은 clean wav파일은 validation에 넣으면 안된다. 
noise의 경우엔 train에 넣은 noise 그대로 validation에 사용하면 된다.

## Step 2
1. generate_noisy_data.py 설정  
설정한 mode의 clean 데이터와 noise 데이터를 결합시켜 noisy 데이터로 만들어주는 코드.  
(clean 데이터 + noise 데이터 => noisy 데이터)  
> main에서 speech_dir = Path("./Dataset")로 설정. 참고로 ./는 현재디렉토리를 의미.  
> 현재 디렉토리인 프로젝트 디렉토리의 하위 디렉토리인 Dataset을 기본 경로로 설정하겠다는 것임.  

2. generate_noisy_data.py 실행  
이 파이썬 코드는 3가지 인자를 입력해야함. mode, snr, fs
> (1) 터미널에서 파이참에서 사용 중인 conda 가상환경을 활성화한다  
(2) 터미널에서 프로젝트 디렉토리로 이동한다  
(3) python generate_noisy_data.py [mode] [snr] [fs] 입력    
    예시) python generate_noisy_data.py 'train' '0' '16000'  
    train noisy data를 0db로 clean data 수 만큼 생성  
(4) 인자를 train으로 넣었으면 train의 clean wav 파일과 noise wav 파일이 합쳐져 noisy라는 디렉토리를 생성하고 그 안에 noisy wav 파일을 만들 것임.

## Step 3  
* wav_to_numpy.py  
생성한 noisy wav에 해당하는 라벨 clean wav 파일을 찾아 매핑하는 코드  
(noisy data의 파일 명은 clean data 파일 명 + 합성에 사용한 noise data 이름이기 때문에 noise data 이름을 제외한 앞 부분을 따서 clean 폴더에서 찾고, 리스트에 [noisy, clean]를 append 해주는 방식임)  
> (1) wav_to_numpy.py 설정 및 실행 (train noisy 라벨링)  
> (2) wav_to_numpy_validation.py 설정 및 실행 (validation noisy 라벨링)  

## Step 4
* dataloader.py 설정
> dataloader.py의 Wave_Dataset 클래스에서 train / valid의 path를 Step 3에서 생성한 npy 파일로 지정  
예시) self.input_path = "./Dataset/train_dataset_norm.npy"  

## Step 5
1. config.py 설정  
모델 구조, 훈련 방식 등을 교체하기 위한 configuration  
setting을 변경해가며 데이터셋에 대한 최적의 실험 세팅을 찾아내고자 함.  
> (1) LOSS : MSE, SDR, SI-SNR, SI-SDR  
> (2) MODEL : DCCRN, CRN, FullSubNet  
> (3) Skip Conncection : True/False  
> (4) EPOCH  
> (5) BATCH  
> (6) Learning Rate  
> (7) Sampling frequency   
(expr_num에 지정된 이름으로 실험결과가 ./models에 저장됨.)  
2. train_interface.py 실행  

## Strp 6
> 실험 결과 분석
터미널에서 tensorboard --logdir ./logs 명렁어 실행하면 locahost 주소가 뜸.  
이 주소에서 성능 지표를 확인할 수 있음.  

# Reference  
DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement  
Yanxin Hu, Yun Liu, Shubo Lv, Mengtao Xing, Shimin Zhang, Yihui Fu, Jian Wu, Bihong Zhang, Lei Xie  [[arXiv]](https://arxiv.org/abs/2008.00264) [[code]](https://github.com/huyanxin/DeepComplexCRN)  
  
FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement  
Xiang Hao, Xiangdong Su, Radu Horaud, Xiaofei Li  [[arXiv]](https://arxiv.org/abs/2010.15508) [[code]](https://github.com/haoxiangsnr/FullSubNet)
  
Deep Noise Suppression (DNS) Challenge 4 - ICASSP 2022 [[link]](https://github.com/microsoft/DNS-Challenge)  

>'Supervised and helped by C-J Lee, S-R Hwang, and H-U Yoon'
