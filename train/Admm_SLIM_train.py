import sys
sys.path.append("../")

import pandas as pd

from arg_parse.Admm_SLIM_args import get_args
from data_loader.Admm_SLIM_dataloader import *
from models.Admm_SLIM import AdmmSlim
from utils.Admm_SLIM_utils import *


def main(args):
    make_matrix_data_set = MakeMatrixDataSet(args=args)
    user_train, user_valid = make_matrix_data_set.get_train_valid_data()
    X = make_matrix_data_set.make_sparse_matrix()
    user_decoder =  make_matrix_data_set.user_decoder
    item_decoder = make_matrix_data_set.item_decoder
    model = AdmmSlim(args)
    model.fit(X = X)
    ndcg, hit = evaluate(model = model, X = X.todense(), user_train = user_train, user_valid = user_valid)
    print(f'NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')
    pd.DataFrame(
        make_submission(model = model, X = X.todense(), user_decoder=user_decoder, item_decoder=item_decoder, args=args), 
        columns=["user", "item"]).to_csv(args.save_path + "admm_slim_submission.csv", index=False
        )

if __name__ == "__main__":
    args = get_args()
    main(args)
    