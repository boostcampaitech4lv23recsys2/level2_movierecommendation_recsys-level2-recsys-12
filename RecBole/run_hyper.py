import argparse
import os

from recbole.quick_start import objective_function
from recbole.trainer import HyperTuning


def main():

    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_files", type=str, default=None, help="fixed config files"
    )
    parser.add_argument("--params_file", type=str, default=None, help="parameters file")
    parser.add_argument(
        "--output_file", type=str, default="hyper_example.result", help="output file"
    )
    args, _ = parser.parse_known_args()

    # get input
    args.config_files = os.path.join("./config", args.config_files)
    args.params_file = os.path.join("./hyper", args.params_file)

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    hp = HyperTuning(
        objective_function,
        algo="bayes",  # random, exhaustive
        # early_stop=10,
        # max_evals=100,
        params_file=args.params_file,
        fixed_config_file_list=[args.config_files],
    )
    hp.run()
    hp.export_result(output_file=args.output_file)
    print("best params: ", hp.best_params)
    print("best result: ")
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == "__main__":
    main()
