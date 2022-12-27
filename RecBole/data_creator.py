import os

import pandas as pd

# SETTINGS
TARGET_DIR = os.path.join(os.getcwd(), "data/movie")
os.makedirs(TARGET_DIR, exist_ok=True)
print("Data Creation Start!")


# make unique_user.csv
FILE = "/opt/ml/input/data/train/train_ratings.csv"
TARGET_NAME = "unique_user.csv"

df = pd.read_csv(FILE)
unique = df["user"].unique()
df = pd.DataFrame(unique, columns=["user"])
df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False)
print("Create unique_user.csv!")


# make movie.inter
FILE = "/opt/ml/input/data/train/train_ratings.csv"
TARGET_NAME = "movie.inter"

df = pd.read_csv(FILE)
df = df.rename(
    columns={
        "user": "user_id:token",
        "item": "item_id:token",
        "time": "timestamp:float",
    }
)
df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False, sep="\t")
print("Create movie.inter!")


# make movie.csv
def make_feature_sequence(x):
    x = list(set(x))
    y = ""
    for item in x:
        y += str(item + " ")
    return y.rstrip()


data_path = "/opt/ml/input/data/train/"
train = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
directors = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep="\t")
genres = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")
titles = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep="\t")
writers = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep="\t")
years = pd.read_csv(os.path.join(data_path, "new_years.tsv"), sep="\t")

genre_df = pd.merge(train, genres, on=["item"])
genre_df = genre_df.drop(["user", "time"], axis=1)
new_genres = genre_df.groupby("item")["genre"].apply(make_feature_sequence)

writer_df = pd.merge(train, writers, on=["item"])
writer_df = writer_df.drop(["user", "time"], axis=1)
new_writers = writer_df.groupby("item")["writer"].apply(make_feature_sequence)

director_df = pd.merge(train, directors, on=["item"])
director_df = director_df.drop(["user", "time"], axis=1)
new_directors = director_df.groupby("item")["director"].apply(make_feature_sequence)

df = pd.merge(years, new_genres, on=["item"])
df = pd.merge(df, new_writers, on=["item"])
df = pd.merge(df, new_directors, on=["item"])

df.to_csv("data/movie/movie.csv", index=False)
print("Create movie.csv!")


# make movie.item
FILE = "data/movie/movie.csv"
TARGET_NAME = "movie.item"

df = pd.read_csv(FILE)
df = df.rename(
    columns={
        "item": "item_id:token",
        "year": "year:token",
        "genre": "genre:token_seq",
        "writer": "writer:token_seq",
        "director": "director:token_seq",
    }
)
df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False, sep="\t")
print("Create movie.item!")
