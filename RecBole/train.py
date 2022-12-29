import argparse
import os
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.trainer.trainer import RecVAETrainer
from recbole.utils import get_model, init_logger, init_seed

if __name__ == "__main__":

    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, default="RecVAE", help="name of models"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="movie", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    args = parser.parse_args()

    # set config
    args.config_files = os.path.join("./config", args.config_files)
    config = Config(
        model=args.model, dataset=args.dataset, config_file_list=[args.config_files]
    )

    # init random seed
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    imported_model = get_model(args.model)
    model = imported_model(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    if args.model == "RecVAE":
        trainer = RecVAETrainer(config, model)
    else:
        trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)
