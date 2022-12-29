# Ensemble

각각의 모델은 다른 시야를 가지고 있습니다. 특히 모델의 계열이 다를 경우 그러한 경향성은 뚜렷하게 두드러집니다.  
다양한 모델을 앙상블하여 견고한 결과를 만들고자 하며, 다양한 방법의 ensemble과 voting을 시도해볼 수 있습니다.  

# Directory Structure

result - ensemble(voting)이 완료된 제출용 결과 파일이 저장됩니다.  
source - ensemble(voting) 하고자 하는 csv 파일들을 위치합니다.  

# Files

voting.py - top11-15의 정보를 추가적으로 이용하는 방식으로 voting을 구현하였습니다. 자세한 설명은 하단에 기록되어있습니다.  
requirements.txt - ```pip install -r requirements.txt``` 명령어를 통해 설치할 수 있습니다.  

# voting.py

## Intro
A모델과 B모델을 앙상블할 때, Top10 정보만 사용할 경우 사실상 가중치가 높은 모델의 Top10만이 재선별되는 문제점이 있습니다.  
이러한 문제점을 해결하기 위해 Top11-15 정보를 추가적으로 사용하여 Top10 중에서도 유력한 후보를 효율적으로 선별하고자 했습니다.  

## Inputs
source 폴더에 voting하고자 하는 csv 파일을 2개 이상 넣어주어야 합니다. 단, voting.py를 이용하여 voting을 진행하기 위해서는,  
반드시 Top15 csv 파일을 넣어주어야 합니다.  

<p align="center">
  <img src="https://user-images.githubusercontent.com/67851701/209912570-8de91bce-1488-4e61-a2ad-603d25c1b952.png" width=300 height=400>
  <img src="https://user-images.githubusercontent.com/67851701/209912913-886d904b-1058-49c6-a569-e432caf49075.png" width=300 height=400>
</p>

## How to vote
사용 방법은 간단합니다.  

0. ```python voting.py``` 명령어를 입력합니다.  
1. voting에 사용하고자 하는 파일 번호를 입력합니다. ex) 0 1  
2. voting에 사용하고자 하는 Top1-10과 Top11-15의 가중치를 입력합니다. ex) 0.75 0.25  
3. voting에 사용하고자 하는 모델들의 가중치를 입력합니다. ex) 모델이 2개일 경우, 0.4 0.6 / 3개일 경우, 0.3 0.3 0.4  
4. voting 결과는 ensemble/result에 저장됩니다.  

## How it calculates

<p align="center">
  <img src="https://user-images.githubusercontent.com/67851701/209916832-abf4a449-2c2e-45f1-8423-7a619878bbb4.png">
</p>  

Top1-10과 Top11-15의 가중치가 0.75:0.25이며, 모델 A와 모델 B의 가중치가 0.6:0.4인 상황을 예시로 합니다.  

1. item이 두 모델 모두에서 Top10에 선정된 경우, 0.45+0.3=0.75의 가중치를 가지게 됩니다.
2. item이 A 모델에서 Top10, B 모델에서 Top15에 선정된 경우, 0.45+0.1=0.55의 가중치를 가지게 됩니다.
3. item이 A 모델에서 Top15, B 모델에서 Top10에 선정된 경우, 0.15+0.3=0.45의 가중치를 가지게 됩니다.
4. item이 두 모델 모두에서 Top15에 선정된 경우, 0.15+0.1=0.25의 가중치를 가지게 됩니다.
5. item이 A 모델에서만 Top10에 선정된 경우, 0.45의 가중치를 가지게 됩니다.
6. item이 B 모델에서만 Top10에 선정된 경우, 0.3의 가중치를 가지게 됩니다.
7. item이 A 모델에서만 Top15에 선정된 경우, 0.15의 가중치를 가지게 됩니다.
8. item이 B 모델에서만 Top15에 선정된 경우, 0.1의 가중치를 가지게 됩니다.  

이 경우에서는, Top15에 위치한 item은 Top10에 선정된 아이템 중에서 변별력을 부여하는 용도로만 사용이 됩니다.  
Top15에만 위치한 아이템으로는 새로운 Top10에 선정되는 shake up이 발생할 수 없습니다.  
모델의 갯수, Top10과 Top15의 가중치, 모델 별 가중치에 따라 Top15에만 존재하는 아이탬으로 새로운 Top10에 선별되는 shake up이 발생할 수 있습니다.  

결과적으로, 계산된 가중치가 높은 순서부터 새로운 10개의 item을 선별하여 submission 파일을 만들게 됩니다.  