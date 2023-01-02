import pandas as pd
import os
from pytz import timezone
from arg_parse.EASE_args import parse_args

import warnings
warnings.filterwarnings(action="ignore")


def dataloader(args):
    
    train_df = pd.read_csv(args.data_path + 'train_ratings.csv')
    
    users = train_df.user.unique()
    items = train_df.item.unique()
    
    return train_df, users, items
    
