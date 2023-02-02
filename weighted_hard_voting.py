import os
import argparse
import pandas as pd
from tqdm import tqdm

class WeightedHardVoting():
    def __init__(self, args):
        if not args.file_list:
            raise "적어도 하나의 파일은 입력해야 합니다."
        if not args.weight:
            args.weight = [1]*len(args.file_list)
        if len(args.file_list) != len(args.weight):
            raise "파일과 가중치의 길이는 동일해야합니다. 모든 파일의 가중치를 동일하게 설정하려면 --weight를 제거하세요."
        self.args = args
        self.get_popularity()
        self.make_weighted_files()
        self.cat_files()
        self.make_result()
        self.make_submission_file()

    def get_popularity(self):
        self.rating = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
        self.itme2papular = self.rating.item.value_counts().to_dict()

    def make_weighted_files(self):
        self.files = []
        for x, w in zip(self.args.file_list, self.args.weight):
            file = pd.read_csv(os.path.join(self.args.data_path, x))
            file["vote"] = -w
            self.files.append(file)
    
    def cat_files(self):
        self.data = pd.concat(self.files)
    
    def make_result(self):
        self.result = self.data.groupby(["user","item"]).sum().reset_index()
        self.result["papular"] = -self.result.item.map(self.itme2papular)
        self.result = self.result.drop_duplicates().sort_values(["vote","papular"])

        self.top10 = []

        for user_id in tqdm(self.rating.user.unique(),"select top10 data..."):
            user_data = self.result[self.result.user == user_id]
            for item_id in user_data.item.values[:10]:
                self.top10.append([user_id, item_id])
    
    def make_submission_file(self):
        pd.DataFrame(self.top10, columns = ["user","item"]).to_csv(self.args.save_name+".csv", index=False)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="/opt/ml/input/code/submission")
    parser.add_argument("--file_list", nargs='+', type=list, default=[])
    parser.add_argument("--weight", nargs='+', type=list, default=[])

    parser.add_argument("--use_all_files", type=bool, default=False)
    parser.add_argument("--save_name", type=str, default="submission")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.use_all_files:
        args.file_list = os.listdir(args.data_path)
    print(args)
    ensemble = WeightedHardVoting(args)
