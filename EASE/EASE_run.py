from EASE_models import EASE
import pandas as pd
import os
from datetime import datetime
from pytz import timezone
from EASE_args import parse_args

import warnings
warnings.filterwarnings(action="ignore")


def main(args):
    
    train_df = pd.read_csv(args.data_path + 'train_ratings.csv')
    


    ease = EASE()
    
    users = train_df.user.unique()
    items = train_df.item.unique()
    
    # if args.wandb:
    #     #wandb.login() # 최초 로그인
    #     wandb.init(
    #         project='EASE', entity="movie-recsys-12"
    #     )
    #     #wandb.run.name = f"bs:{args.batch_size}_lr:{args.lr}"
    #     wandb.config = vars(args)

    
    ease.fit(train_df, args.lambda_)
    
    result_df = ease.predict(train_df, users, items, args.K)
    
    print(result_df)


    file_name = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
    result_df[["user", "item"]].to_csv(args.output_dir + f"{file_name}_{args.K}.csv", index=False)

   
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)