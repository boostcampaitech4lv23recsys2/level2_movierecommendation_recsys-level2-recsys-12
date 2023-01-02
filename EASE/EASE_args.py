import argparse

def parse_args():
    parser = argparse.ArgumentParser("arguments for EASE")

    parser.add_argument('--data_path', default="/opt/ml/input/data/train/", type=str)
    parser.add_argument('--lambda_', default=300, help="lambda_: l2-regularization term", type=float)
    parser.add_argument('--K', default=10, help="decision top-K", type=int)
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--wandb', default=True, help='Wandb run')
    
    args = parser.parse_args()

    return args