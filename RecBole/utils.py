from logging import getLogger

import numpy as np
import pandas as pd
import torch
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import init_logger, init_seed
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from tqdm import tqdm


def add_last_item(old_interaction, last_item_id, max_len=50):
    new_seq_items = old_interaction["item_id_list"][-1]
    if old_interaction["item_length"][-1].item() < max_len:
        new_seq_items[old_interaction["item_length"][-1].item()] = last_item_id
    else:
        new_seq_items = torch.roll(new_seq_items, -1)
        new_seq_items[-1] = last_item_id
    return new_seq_items.view(1, len(new_seq_items))


def predict_for_all_item(external_user_id, dataset, test_data, model, config):
    """
    Predict using all items
    """
    model.eval()
    with torch.no_grad():
        uid_series = dataset.token2id(dataset.uid_field, [external_user_id])
        index = np.isin(dataset[dataset.uid_field].numpy(), uid_series)
        input_interaction = dataset[index]
        test = {
            "item_id_list": add_last_item(
                input_interaction,
                input_interaction["item_id"][-1].item(),
                model.max_seq_length,
            ),
            "item_length": torch.tensor(
                [
                    input_interaction["item_length"][-1].item() + 1
                    if input_interaction["item_length"][-1].item()
                    < model.max_seq_length
                    else model.max_seq_length
                ]
            ),
        }
        new_inter = Interaction(test)
        new_inter = new_inter.to(config["device"])
        new_scores = model.full_sort_predict(new_inter)
        new_scores = new_scores.view(-1, test_data.dataset.item_num)
        new_scores[:, 0] = -np.inf
    return torch.topk(new_scores, 15)


def generate_predict(
    dataset, test_data, model, config, user_data_file="./data/movie/unique_user.csv"
):
    user_data = pd.read_csv(user_data_file, dtype=str)
    users = user_data["user"].unique()

    predict = []

    for user in tqdm(users):
        uid_series = dataset.token2id(dataset.uid_field, [user])
        _, topk_iid_list = full_sort_topk(
            uid_series, model, test_data, k=15, device=config["device"]
        )
        external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())[0]
        predict.append(list(external_item_list))
    return predict


def generate_predict_seq(
    dataset, test_data, model, config, user_data_file="./data/movie/unique_user.csv"
):
    """
    Generate prediction from user profile data
    """
    user_data = pd.read_csv(user_data_file, dtype=str)
    users = user_data["user"].unique()

    predict = []

    for user in tqdm(users):
        temp = predict_for_all_item(user, dataset, test_data, model, config)
        external_item_list = dataset.id2token(dataset.iid_field, temp.indices.cpu())[0]
        predict.append(list(external_item_list))
    return predict


def gererate_submission_from_prediction(
    prediction,
    user_data_file="./data/movie/unique_user.csv",
    output_dir="../submission/RecBole_submission.csv",
):
    """
    Generate submission file from prediction list
    """
    user_data = pd.read_csv(user_data_file, dtype=str)
    users = user_data["user"].unique()

    result = []
    for index, user in enumerate(users):
        for item in prediction[index]:
            result.append([user, item])
    pd.DataFrame(result, columns=["user", "item"]).to_csv(output_dir, index=False)
