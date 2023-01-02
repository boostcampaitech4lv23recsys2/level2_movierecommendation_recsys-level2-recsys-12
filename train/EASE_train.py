import sys
sys.path.append("../")

from models.EASE import EASE
import pandas as pd
import os
from datetime import datetime
from pytz import timezone
from arg_parse.EASE_args import parse_args
from data_loader.EASE_dataloader import dataloader
from datetime import datetime
#from trainer.EASE_trainer import predict

import warnings
warnings.filterwarnings(action="ignore")


def main(args):
    
    train_df, users, items = dataloader(args)
    
    ease = EASE()
    
    ease.fit(train_df, args.lambda_)
    result_df = ease.predict(train_df, users, items, args.K)

    file_name = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
    result_df[["user", "item"]].to_csv(args.output_dir + f"EASE_{file_name}_{args.K}.csv", index=False)

   
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
    
