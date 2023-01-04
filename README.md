# :cinema: Movie Recommendation

<p align="center">
  <img src="https://user-images.githubusercontent.com/67851701/210546490-7c1e61ba-61f8-45a2-8233-b52ddd7febe5.png">  
</p>


<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white"> <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=white"> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=white"> <img src="https://img.shields.io/badge/W&B-FFBE00?style=for-the-badge&logo=WeightsandBiases&logoColor=white"> <img src="https://img.shields.io/badge/Scikit_learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">


## :loudspeaker: Competition Introduction

여러분들께서는 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측하게 됩니다.  

이 문제는 timestamp를 고려한 사용자의 순차적인 이력을 고려하고 implicit feedback을 고려한다는 점에서,  
시중의 여러 강의에서 사용되는 1-5점 평점 (explicit feedback) 기반의 행렬을 사용한 협업 필터링 문제와 차별화됩니다.  

대회에 대한 더 자세한 내용은 [대회 개요](https://github.com/boostcampaitech4lv23recsys2/level2_movierecommendation_recsys-level2-recsys-12/wiki/Overview) 에서 확인할 수 있습니다!


## :droplet: Team Members

Naver Boostcamp AI Tech 4기 Recsys 12조, **Recommendation is All You Need**

| [<img src="https://github.com/thkyang324.png" width="100px">](https://github.com/thkyang324) | [<img src="https://github.com/tree-jhk.png" width="100px">](https://github.com/tree-jhk) | [<img src="https://github.com/daeni-dang.png" width="100px">](https://github.com/daeni-dang) | [<img src="https://github.com/JisooRyu99.png" width="100px">](https://github.com/JisooRyu99) | [<img src="https://github.com/7dudtj.png" width="100px">](https://github.com/7dudtj) |  
| :---: | :---: | :---: | :---: | :---: |  
| [강태훈](https://github.com/thkyang324) | [권준혁](https://github.com/tree-jhk) | [김다은](https://github.com/daeni-dang) | [류지수](https://github.com/JisooRyu99) | [유영서](https://github.com/7dudtj) |

- **강태훈** : Template 구조 선정, ADMM-SLIM 모델 구현, AutoEncoder 계열 모델 성능 비교  
- **권준혁** : S3rec, SASrec 모델 개선, 튜닝 시 sweep 사용  
- **김다은** : MultiVAE 베이스라인 구현, MultiVAE RecBole  
- **류지수** : EASE 베이스라인 구현, EASE RecBole 적용  
- **유영서** : RecBole 환경 구축, RecVAE 모델 구현, Voting 구현  


## :santa: How we work?
팀원 간의 원활한 소통을 위해, 다양한 협업 도구를 사용하였습니다!

- ### Working Tools

Slack | Notion | W&B | Github
:---: | :---: | :---: | :---:
<img src="https://user-images.githubusercontent.com/67851701/207574270-2e4aaead-f915-41a4-b3b1-ed2252e60cc3.png"  width="100" height="100"/> | <img src="https://user-images.githubusercontent.com/67851701/207574394-07b37c3d-e32d-44e9-a359-0e8935ef7bf2.png"  width="100" height="100"/> | <img src="https://user-images.githubusercontent.com/67851701/207574592-db1e7b71-fb3d-4db3-889d-157a8e70fd38.png"  width="100" height="100"/> | <img src="https://user-images.githubusercontent.com/67851701/207576204-88378715-df1f-41af-8394-c5115a2b8999.png"  width="100" height="100"/>

저희 팀의 협업 방식에 대하여 더 궁금하시다면, [Team RAYN의 협업 방식](https://github.com/boostcampaitech4lv23recsys2/level2_movierecommendation_recsys-level2-recsys-12/wiki/Cooperation) 에서 확인할 수 있습니다!


## :books: Project Outline

<p>
  <img src="https://user-images.githubusercontent.com/67851701/210553870-693aed13-784b-4bc2-82d3-2778399f0a15.png" width="800">  
</p>

다양한 모델을 실험한 후, Top10을 보다 효과적으로 앙상블 하기 위해 고안안 voting 알고리즘에 Top15를 넣어준 후, 최종 Top10을 선발합니다.  


## :shopping_cart: Model Result

모델 실험 결과는 다음과 같습니다.  

| Model | ADMMSLIM | EASE | MultiVAE | RecVAE | S3Rec |
| :---: | :---: | :---: | :---: | :---: | :---: |
| public Recall@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| private Recall@10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


## :flower_playing_cards: Voting Strategy

<p>
  <img src="https://user-images.githubusercontent.com/67851701/210558404-188a4f45-30be-4277-b389-9301817152fe.png" width="800">  
</p>

여러 모델을 실험한 결과, EASE 모델이 다른 모델과의 성능 차이가 많이 나는 점을 확인하였습니다.  
이로 인하여 앙상블 과정에서 문제점이 도출되었습니다.  

두 모델을 Hard Voting 하려는 경우, 두 모델의 성능 차이로 인하여 앙상블 결과의 성능이 저하되는 문제가 있었으며,  
두 모델에 가중치를 다르게 부여할 경우, 가중치가 높은 모델의 Top10만 그대로 선발되는 문제가 있었습니다.  

앙상블에서의 문제점을 해결하기 위해, Top11~15의 데이터를 추가적으로 사용하여 Top10을 선발하는 voting 알고리즘을 구현하였습니다.  

새로운 voting 알고리즘은 크게 두 종류로 나뉘어집니다.  
첫번째는, Top11\~15의 정보를 이용하여 Top10 간의 변별력을 부여하여 새로운 Top10을 선발하는 방식이며,  
두번째는, Top11\~15에만 존재하는 item이 새로운 Top10에 선발될 수 있는 가능성을 제공하는 방식입니다.  

Voting Strategy에 대한 더 자세한 내용은 [앙상블 전략](https://github.com/boostcampaitech4lv23recsys2/level2_movierecommendation_recsys-level2-recsys-12/wiki/Voting) 에서 확인할 수 있습니다!


## :seedling: To see more
저희 Project에 대한 더 자세한 정보를 알고싶으시다면, [Project Wiki](https://github.com/boostcampaitech4lv23recsys2/level2_movierecommendation_recsys-level2-recsys-12/wiki) 를 확인하세요!


## :100: Competition Result

<p>
  <img src="https://user-images.githubusercontent.com/67851701/210548268-2abe03f6-b6c3-45d9-b22f-41093d75d5c4.JPG" width="400">  
</p>


| 리더보드 | Recall@10 | 순위 |
| :---: | :---: | :---: |
| public | 0.0000 | 00등 |
| private | 0.0000 | 00등 |

**최종 순위: 전체 13팀 중 00등**
<p> <img src="https://user-images.githubusercontent.com/67851701/206601614-09bd63a0-472d-4884-8ff2-f992f9787dba.JPG"> </p>
