import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="MultiDAE", type=str, help="select model [MultiDAE, MultiVAE]")
    parser.add_argument("--save_path", default="/opt/ml/input/code/submission", type=str)
    parser.add_argument("--data_path", default="/opt/ml/input/data/train", type=str)
    parser.add_argument("--p_dims", nargs="+", type=int, default=[200, 600], help="scheduler lr milestones")
    parser.add_argument("--valid_samples", type=int, default=10, help="hidden size of transformer model")
    parser.add_argument("--seed", type=int, default=42, help="hidden size of transformer model")
    parser.add_argument("--total_anneal_steps", type=int, default=200000, help="hidden size of transformer model")
    parser.add_argument("--batch_size", type=int, default=500, help="hidden size of transformer model")
    parser.add_argument("--num_epochs", type=int, default=200, help="hidden size of transformer model")
    parser.add_argument("--num_workers", type=int, default=4, help="hidden size of transformer model")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="hidden dropout p")
    parser.add_argument("--anneal_cap", type=float, default=0.2, help="hidden dropout p")
    parser.add_argument("--lr", type=float, default=0.0001, help="lr")
    parser.add_argument("--loss_type",type=str,default="Multinomial",help="select loss type [Multinomial, Gaussian, Logistic]")
    parser.add_argument("--wandb", type=str, default="NO_USE", help="option for running wandb")
    args = parser.parse_args()

    return args
