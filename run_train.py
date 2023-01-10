import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from sasrec_models import S3RecModel
from trainers import FinetuneTrainer
from sasrec_utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)
import warnings
warnings.filterwarnings(action="ignore")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep", default="True", type=bool)
    parser.add_argument("--wandb", default=True, type=bool, help="option for running wandb")

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.01)
    parser.add_argument("--max_seq_length", default=200, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
    parser.add_argument("--aap_weight", type=float, default=0.2, help="aap loss weight")
    parser.add_argument("--mip_weight", type=float, default=1.5, help="mip loss weight")
    parser.add_argument("--map_weight", type=float, default=1.0, help="map loss weight")
    parser.add_argument("--sp_weight", type=float, default=0.5, help="sp loss weight")

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.95, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # parser.add_argument("--using_pretrain", action="store_true")
    parser.add_argument("--using_pretrain", default=True)
    parser.add_argument("--wandb", default="NO_USE", type=str, help="option for running wandb")
    parser.add_argument("--tqdm", default=1, type=int, help="option for running tqdm")
    
    # LR Scheduler
    parser.add_argument("--scheduler", type=str, default="None", help="Choice LR-Scheduler")

    parser.add_argument("--lr_gamma", type=float, default=0.5, help="scheduler lr gamma")

    parser.add_argument("--lr_step_size", type=int, default=20, help="scheduler lr step_size")
    parser.add_argument("--lr_step_size_up", type=int, default=10, help="scheduler lr step_size_up")
    parser.add_argument("--lr_step_size_down", type=int, default=20, help="scheduler lr step_size_down")

    parser.add_argument("--lr_milestones", nargs='+', type=int, default=[30,60,90], help="scheduler lr milestones")
    parser.add_argument("--lr_base_lr", type=float, default=0.001, help="scheduler lr base_lr")
    parser.add_argument("--lr_max_lr", type=float, default=0.1, help="scheduler lr max_lr")
    parser.add_argument("--lr_mode", type=str, default="triangular", help="scheduler lr mode")

    parser.add_argument("--lr_T_0", type=int, default=30, help="scheduler lr T_0")
    parser.add_argument("--lr_T_mult", type=int, default=2, help="scheduler lr T_mult")
    parser.add_argument("--lr_T_max", type=float, default=0.001, help="scheduler lr T_max")
    parser.add_argument("--lr_T_up", type=int, default=5, help="scheduler lr T_up")

    parser.add_argument("--lr_eta_min", type=float, default=0.001, help="scheduler lr eta_min")
    parser.add_argument("--lr_eta_max", type=float, default=0.01, help="scheduler lr eta_max")

    parser.add_argument("--model_name", default="Finetune_full", type=str)

    args = parser.parse_args()

    # save model args
    args_str = f"{args.model_name}_{args.data_name}_max_seq_len_{args.max_seq_length}_hidden_{args.hidden_size}_beta2_{args.adam_beta2}_attn_drop_{args.attention_probs_dropout_prob}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
        args.data_file
    )

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    test_dataset = SASRecDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.batch_size
    )

    if args.wandb:
        import wandb
        wandb.login()
    
        wandb.init(project="max_seq_len_200_sweep", entity="movie-recsys-12", config=vars(args))
        # wandb.run.name = f"{args_str}"
    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(
        model, train_dataloader, eval_dataloader, test_dataloader, None, args
    )

    print(args.using_pretrain)
    if args.using_pretrain:
        # pretrained_path = os.path.join(args.output_dir, f"pretrain_max_seq_len_{args.max_seq_length}_hidden_{args.hidden_size}_aap_{args.aap_weight}_mip_{args.mip_weight}_map_{args.map_weight}.pt")
        pretrained_path = os.path.join(args.output_dir, f"pretrain_max_seq_len_{args.max_seq_length}.pt")
        try:
            trainer.load(pretrained_path)
            print(f"Load Checkpoint From {pretrained_path}!")

        except FileNotFoundError:
            print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    else:
        print("Not using pretrained model. The Model is same as SASRec")

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True, sweep=args.sweep)
    for epoch in range(args.epochs):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)
        if args.wandb:
            wandb.log(
                {
                    "RECALL@5": scores[0],
                    "NDCG@5": scores[1],
                    "RECALL@10": scores[2],
                    "NDCG@10": scores[3],
                    "Lr_rate": trainer.scheduler.optimizer.param_groups[0]['lr'],
                }
            )
        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
