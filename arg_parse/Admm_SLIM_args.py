import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--save_path", default="/opt/ml/input/code/submissio", type=str)
    parser.add_argument("--valid_samples", type=int, default=10, help="length of valid samples")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    parser.add_argument("--n_iter", type=int, default=50, help="set iteration")
    parser.add_argument("--lambda_1", type=int, default=12, help="set lambda_1")
    parser.add_argument("--lambda_2", type=int, default=19, help="set lambda_2")
    parser.add_argument("--rho", type=int, default=20, help="set rho")
    parser.add_argument("--eps_rel", type=float, default=0.004642, help="set eps_rel")
    parser.add_argument("--eps_abs", type=float, default=0.0003856, help="set eps_abs")

    parser.add_argument("--positive", type=bool, default=True, help="set positive")
    parser.add_argument("--verbose", type=bool, default=False, help="set verbose")

    parser.add_argument("--predict_size", type=int, default=10, help="예측할 영화의 개수를 입력하세요.")
    parser.add_argument("--wandb", type=bool, default=False, help="wandb 쓸지 말지")
    parser.add_argument("--sweep", type=bool, default=False, help="sweep 쓸지 말지")

    args = parser.parse_args()

    return args