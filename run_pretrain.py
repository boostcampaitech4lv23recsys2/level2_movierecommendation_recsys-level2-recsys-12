import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import PretrainDataset
from sasrec_models import S3RecModel
from trainers import PretrainTrainer
from sasrec_utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs_long,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sweep", default="True", type=bool)
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args

    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of transformer model"
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
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=300, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # pre train args
    parser.add_argument(
        "--pre_epochs", type=int, default=300, help="number of pre_train epochs"
    )
    parser.add_argument("--pre_batch_size", type=int, default=32)

    parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
    parser.add_argument("--aap_weight", type=float, default=0.2, help="aap loss weight")
    parser.add_argument("--mip_weight", type=float, default=2.0, help="mip loss weight")
    parser.add_argument("--map_weight", type=float, default=1.5, help="map loss weight")
    parser.add_argument("--sp_weight", type=float, default=0.5, help="sp loss weight")

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.95, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.99, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    
    parser.add_argument("--wandb", default=True, type=bool, help="option for running wandb")
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
    
    parser.add_argument("--model_name", default="Pretrain", type=str)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    check_path(args.output_dir)

    args.checkpoint_path = os.path.join(args.output_dir, "Pretrain.pt")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # args.data_file = args.data_dir + args.data_name + '.txt'
    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"
    # concat all user_seq get a long sequence, from which sample neg segment for SP
    user_seq, max_item, long_sequence = get_user_seqs_long(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    args.item2attribute = item2attribute

    model = S3RecModel(args=args)
    trainer = PretrainTrainer(model, None, None, None, None, args)

    checkpoint_path = os.path.join(args.output_dir, f"pretrain_max_seq_len_{args.max_seq_length}.pt")
    early_stopping = EarlyStopping(checkpoint_path, patience=3, verbose=True, sweep=args.sweep)

    pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
    
    if args.wandb:
        import wandb
        wandb.login()
        wandb.init(project="max_len_200", entity="movie-recsys-12", config=vars(args))
        wandb.run.name = f"max_len: {args.max_seq_length} hs: {args.hidden_size} atten_do: {args.attention_probs_dropout_prob} hidden_do: {args.hidden_dropout_prob} bs:{args.batch_size} lr:{args.lr}"
    
        for epoch in range(args.pre_epochs):
            pretrain_sampler = RandomSampler(pretrain_dataset)
            pretrain_dataloader = DataLoader(
                pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size
            )

            losses = trainer.pretrain(epoch, pretrain_dataloader)

            # ## comparing `sp_loss_avg``
            loss_avg = (losses["aap_loss_avg"] + losses["mip_loss_avg"] + losses["map_loss_avg"] * 0.1) / 3
            # early_stopping(np.array([-losses["aap_loss_avg"]]), trainer.model)
            early_stopping(np.array([loss_avg]), trainer.model)
            
            if args.wandb:
                wandb.log(
                    {
                        "aap_loss_avg": losses["aap_loss_avg"],
                        "mip_loss_avg": losses["mip_loss_avg"],
                        "map_loss_avg": losses["map_loss_avg"],
                        "loss_avg": loss_avg,
                        "Lr_rate": trainer.scheduler.optimizer.param_groups[0]['lr'],
                    }
                )
                
            if early_stopping.early_stop:
                print("Early stopping")
                break


if __name__ == "__main__":
    main()
