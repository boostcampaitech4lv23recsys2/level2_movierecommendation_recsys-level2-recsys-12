# Movie Recommendation Baseline Code

영화 추천 대회를 위한 베이스라인 코드입니다. 다음 코드를 대회에 맞게 재구성 했습니다.

### Reference
- https://github.com/Darel13712/ease_rec
- https://arxiv.org/pdf/1905.03375.pdf
- https://github.com/SeongBeomLEE/RecsysTutorial/tree/main/EASER
- https://dl.acm.org/doi/pdf/10.1145/3460231.3474273

## How to run EASE
1. python train/EASE_train
- 저장 파일 : EASE_날짜_시간_{TopK개수}.csv로 저장됩니다.

## How to run EASER
1. python train/EASE_train --model EASER
- 저장 파일 : EASER_날짜_시간_{TopK개수}.csv로 저장됩니다.


----
EASE와 EASER의 argparse는 arg_parse/EASE_args.py에서 확인할 수 있습니다. 