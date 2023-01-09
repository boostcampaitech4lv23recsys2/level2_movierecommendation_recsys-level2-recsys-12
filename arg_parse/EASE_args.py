import argparse

def parse_args():
    parser = argparse.ArgumentParser("arguments for EASE")
    parser.add_argument('--K', default=10, help="decision top-K", type=int)
    parser.add_argument('--output_dir', default='../submission/')
    parser.add_argument('--data_path', default="/opt/ml/input/data/train/", type=str)
    parser.add_argument('--model', default="EASE", type=str, help="Choose between EASE and EASER.")
    
    ###################### EASE ##########################
    parser.add_argument('--lambda_', default=300, help="lambda_: l2-regularization term", type=float)
    
    ###################### EASER ##########################
    parser.add_argument('--threshold', default=3500, type=float)
    parser.add_argument('--lambdaBB', default=500, type=float)
    parser.add_argument('--lambdaCC', default=10000, type=float)
    parser.add_argument('--rho', default=50000, type=float)
    parser.add_argument('--epochs', default=100, type=float)

    
    args = parser.parse_args()

    return args