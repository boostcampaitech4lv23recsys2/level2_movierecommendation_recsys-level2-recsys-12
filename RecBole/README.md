# RecBole

RecVAE 모델을 사용하기 위하여 RecBole을 구현하였으며, 추후 다른 모델로 사용 범위를 넓힐 수 있습니다.  

# Directory Structure

config - 모델 별 config 파일을 저장합니다. 파일의 확장자는 .yaml 입니다.  
data - RecBole에서 사용하는 data 폴더입니다. 하위에 movie 폴더가 있으며, 이곳에 데이터가 저장됩니다.  
model - 학습된 모델이 저장됩니다.  

# Files

data_creator.py - submission에 필요한 unique_user.csv 파일과 RecBole에 사용되는 Atomic Files (ex. .inter, .item, .user ...)를 생성합니다.  
hyper.test - hyper parameter tuning에 사용되는 파일입니다. Tuning 하고자 하는 parameter를 전달합니다.  
inference.py - Movie Recommendation 대회에 맞도록 모델을 inference하며, submission.csv 파일을 생성합니다.  
requirements.txt - ```pip install -r requirements.txt``` 명령어를 통해 설치할 수 있습니다.  
run_hyper.py - hyper parameter tuning을 하는 파일입니다.  
train.py - 모델 학습을 하는 파일입니다.  
utils.py - submission 제작에 필요한 여러 함수들이 있습니다.  

# How to run

0. data/train 디렉토리에 new_years.tsv 파일이 있어야합니다. 없을 경우, EDA/YYS_EDA_Year.ipynb를 실행하여 생성할 수 있습니다.
1. 학습하고자 하는 모델에 대한 .yaml 파일이 RecBole/config에 있어야 합니다. 현재 RecVAE.yaml만 존재합니다.
2. 일부 모델의 경우, Trainer를 맞춤으로 지정해줘야 합니다. 자세한 내용은 [RecBole Docs](https://recbole.io/docs/recbole/recbole.trainer.trainer.html#module-recbole.trainer.trainer) 에서 확인 바랍니다.
3. requirements를 설치하여서 환경을 맞춰줍니다.
4. ```python data_creator.py``` 명령어를 입력하여 RecBole에 필요한 데이터를 생성합니다.
5. ```python train.py --model=[모델명] --config_files=[모델명.yaml]``` 명령어를 입력하여 학습을 진행합니다.
6. ```python inference.py --config_files=[모델명.yaml] --type=[모델의 type]``` 명령어를 입력하여 추론을 진행합니다.
7. 제출 파일은 code/submission에 저장됩니다.

# More information

모델의 계열에 따라 필요한 input 파일의 종류가 다릅니다. 자세한 내용은 [링크](https://recbole.io/docs/user_guide/data/atomic_files.html) 에서 확인할 수 있습니다.  
이번 대회에 주로 사용될 General 계열이나 Sequential 계열의 경우 .inter 파일만을 필수적으로 요구합니다.