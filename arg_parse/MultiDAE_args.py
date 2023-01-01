import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="/opt/ml/input/data/train", type=str)
    parser.add_argument("--p_dims", nargs='+', type=int, default=[200,400,600], help="scheduler lr milestones")
    parser.add_argument("--valid_samples", type=int, default=10, help="hidden size of transformer model")
    parser.add_argument("--seed", type=int, default=42, help="hidden size of transformer model")
    parser.add_argument("--total_anneal_steps", type=int, default=200000, help="hidden size of transformer model")
    parser.add_argument("--batch_size", type=int, default=2**15, help="hidden size of transformer model")
    parser.add_argument("--num_epochs", type=int, default=50, help="hidden size of transformer model")
    parser.add_argument("--num_workers", type=int, default=4, help="hidden size of transformer model")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="hidden dropout p")
    parser.add_argument("--anneal_cap", type=float, default=0.2, help="hidden dropout p")
    parser.add_argument("--lr", type=float, default=0.005, help="lr")

    args = parser.parse_args()

    return args