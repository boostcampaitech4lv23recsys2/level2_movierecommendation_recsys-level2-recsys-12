# import libraries
import math
import os
import sys

import pandas as pd
from tqdm import tqdm

# get file list
file_list = os.listdir("./source")
file_list = [file for file in file_list if file.endswith(".csv")]
if len(file_list) <= 1:
    sys.exit("[Error] More than 2 csv files are needed. Exit voting.")


# select files
while True:

    # show csv list
    print(
        'Select csv "Number"s in ascending order. Seperate inputs with space bar!\nex) 0 1'
    )
    for i in range(len(file_list)):
        print(f"[{i}] : {file_list[i]}")

    # get inputs
    inputs = input().strip()
    number_list = [int(item) for item in inputs.split()]

    # exception handling
    asc = True
    prev = -1
    for i in number_list:
        if prev >= i:
            print("You have to write number in ascending order. Try again.\n")
            asc = False
            continue
        else:
            prev = i
    if not asc:
        continue
    if len(number_list) < 2:
        print("You have to choose 2 csv files or more.\n")
        continue
    if len(number_list) > len(file_list):
        print("There exists wrong number. Try again.\n")
        continue
    for item in number_list:
        if item >= len(file_list):
            print("There exists wrong number. Try again.\n")
            continue

    break


# make selected file list
new_file_list = []
for idx in number_list:
    new_file_list.append(file_list[idx])
file_list = new_file_list


# show selected files
print("\n----------Files you selected----------")
for idx in range(len(file_list)):
    print(f"[{idx}] : {file_list[idx]}")
print("--------------------------------------\n")


# get ratio for top1-10 and top11-15
while True:

    # get inputs
    print(
        "Write ratio for Top1-10 and Top11-15. Sum of ratios must be 1.\nex) 0.75 0.25"
    )
    inputs = input().strip()
    ratio1_list = [float(item) for item in inputs.split() if inputs]

    # exception handling
    if len(ratio1_list) != 2:
        print("Wrong ratios. Try again.\n")
        continue
    if math.fabs(sum(ratio1_list) - 1.0) > sys.float_info.epsilon:
        print("Sum of ratios is not 1. Try again.\n")
        continue
    break


# get ratio for input csv files
while True:

    # get inputs
    print("\nWrite ratio for input csv files. Sum of ratios must be 1.\nex) 0.5 0.5")
    inputs = input().strip()
    ratio2_list = [float(item) for item in inputs.split() if inputs]

    # exception handling
    if len(ratio2_list) < 2:
        print("Wrong ratios. Try again.\n")
        continue
    if math.fabs(sum(ratio2_list) - 1.0) > sys.float_info.epsilon:
        print("Sum of ratios is not 1. Try again.\n")
        continue
    break


# summary inputs
print("\n----------input information----------")
print(f"Ratio of Top1-10 & Top11-15:\n{ratio1_list[0]} : {ratio1_list[1]}")
print("Ratio of input csv files:")
for idx in range(len(file_list)):
    print(f"[{idx}] : {file_list[idx]} - {ratio2_list[idx]}")
print("-------------------------------------\n")


# load csv files
csv_list = []
for idx in range(len(file_list)):
    csv_file = pd.read_csv(os.path.join("./source", file_list[idx]))
    csv_list.append(csv_file)


# get user information
user_list = csv_list[0]["user"].unique()


# voting
movie_list = []
idx = 0
for user in tqdm(user_list, desc="Voting"):
    tmp_dict = dict()
    for i, csv in enumerate(csv_list):
        for add in range(10):
            if csv["item"][idx + add] not in tmp_dict:
                tmp_dict[csv["item"][idx + add]] = ratio1_list[0] * ratio2_list[i]
            else:
                tmp_dict[csv["item"][idx + add]] += ratio1_list[0] * ratio2_list[i]
        for add in range(10, 15):
            if csv["item"][idx + add] not in tmp_dict:
                tmp_dict[csv["item"][idx + add]] = ratio1_list[1] * ratio2_list[i]
            else:
                tmp_dict[csv["item"][idx + add]] += ratio1_list[1] * ratio2_list[i]
    sorted_dict = sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True)
    for i in range(10):
        movie_list.append(sorted_dict[i][0])
    idx += 15


# make submission
new_user_list = []
for user in user_list:
    for i in range(10):
        new_user_list.append(user)
df = pd.DataFrame({"user": new_user_list, "item": movie_list})
result_name = ""
for i in range(len(file_list)):
    result_name += str(file_list[i][:-4]) + str(ratio2_list[i])
    result_name += "_"
result_name += f"top10_{ratio1_list[0]}_top15_{ratio1_list[1]}.csv"
if not os.path.exists("./result"):
    os.makedirs("./result")
df.to_csv(os.path.join("./result", result_name), index=False)
