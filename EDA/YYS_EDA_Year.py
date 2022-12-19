#!/usr/bin/env python
# coding: utf-8

# In[137]:


# import libraries
import numpy as np
import pandas as pd

get_ipython().run_line_magic("matplotlib", "inline")
import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings(action="ignore")


# In[138]:


# set data
data_path = "/opt/ml/input/data/train/"
train = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
directors = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep="\t")
genres = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")
titles = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep="\t")
writers = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep="\t")
years = pd.read_csv(os.path.join(data_path, "years.tsv"), sep="\t")
train


# # Year.tsv 확인하기

# In[139]:


years.isnull().sum()


# In[140]:


years.info()


# In[141]:


years["year"].describe()


# * 1922년부터 2014년까지의 정보가 있다

# In[142]:


min_year = years["year"].min()
max_year = years["year"].max()
print(f"min year: {min_year}, max year: {max_year}")


# In[143]:


sns.countplot(data=years, x="year", order=range(min_year, max_year, 5))
plt.xticks(rotation=45)
plt.show()


# * year 값의 분포이다

# In[144]:


print(f"Year data shape: {years.shape}")
print(f"number of items: {years['item'].nunique()}")
print(f"number of years: {years['year'].nunique()}")


# # 결측치 처리하기

# In[145]:


train.shape


# In[146]:


train.info()


# In[147]:


train.describe()


# In[148]:


train.nunique()


# In[149]:


len(set(train["item"]))


# In[150]:


len(set(years["item"]))


# In[151]:


len(set(train["item"]) - set(years["item"]))


# * train에서 8개의 item에 대해서 year 값이 years.tsv 데이터에 존재하지 않는다

# In[152]:


train = pd.merge(train, years, how="outer")
train.isnull().sum()


# * year 정보가 없는 8개의 item 결측값이 전체 train에서 1832개의 결측값을 만들고 있다

# In[153]:


item_list = list(train[train["year"].isnull()]["item"].unique())
item_list


# * year 정보가 없는 8개의 item이다. 이들의 year 정보 결측치를 채워서 years.tsv를 다시 만들자
# * item의 title에서 year의 정보를 찾아서 결측치를 채우자

# In[154]:


train = pd.merge(train, titles, how="outer")
train.isnull().sum()


# In[155]:


item_title = train[train["item"] == item_list[0]].iloc[0, 4].split()[-1][1:-1]
item_title


# In[156]:


years


# In[157]:


for i in range(0, 8):
    item_title = train[train["item"] == item_list[i]].iloc[0, 4].split()[-1][1:-1]
    years.loc[len(years)] = {"item": item_list[i], "year": item_title}


# In[158]:


years["year"] = years["year"].astype("int64")
years = years.sort_values(["year"])


# In[159]:


years["year"].dtype


# In[160]:


years.head(20)


# In[161]:


years.info()


# In[162]:


years.describe()


# * 결측치를 채운 결과, min year는 1902로 바뀌었고, max year는 2015로 바뀌었다

# # 검증

# In[163]:


new_years = years.reset_index().drop(["index"], axis=1)
years = pd.read_csv(os.path.join(data_path, "years.tsv"), sep="\t")


# In[164]:


new_years.head(10)


# In[165]:


years.head(10)


# In[166]:


new_years.tail(10)


# In[167]:


years.tail(10)


# In[168]:


print(f"Shape of years: {years.shape}")
print(f"Shape of new years: {new_years.shape}")
print(f"Number of unique year in years:\n{years.nunique()}")
print(f"Number of unique year in new years:\n{new_years.nunique()}")


# * 새로 추가된 year는 기존에 존재하지 않던 새로운 year 값이다

# In[169]:


new_years.to_csv(os.path.join(data_path, "new_years.tsv"), sep="\t", index=False)


# * 결측치를 채워준 새로운 new_years.tsv를 생성한다

# In[170]:


train = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
directors = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep="\t")
genres = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")
titles = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep="\t")
writers = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep="\t")
years = pd.read_csv(os.path.join(data_path, "new_years.tsv"), sep="\t")


# In[171]:


train = pd.merge(train, years, how="outer")
train.isnull().sum()


# * 결측치가 처리된 것을 볼 수 있다

# # 다른 정보와 결합해서 EDA 진행하기

# In[172]:


train = pd.merge(train, directors, how="outer")
train = pd.merge(train, genres, how="outer")
train = pd.merge(train, titles, how="outer")
train = pd.merge(train, writers, how="outer")
train


# In[173]:


train.isnull().sum()


# ## user와 year의 관계 알아보기

# In[174]:


train["user"].nunique()


# In[175]:


def min_max_year(s):
    return s.min(), s.max()


# In[176]:


user_year = train.groupby(["user"]).agg({"year": min_max_year})
user_year.head(20)


# * user와 year을 비교하는 것에서 유의미한 결론은 없는 것 같다
# * year는 영화의 개봉연도이기에, 2015년에도 1902년 영화를 볼 수 있다
# * 유저 별 시청 년도를 비교하는 것은 user와 timestamp를 비교해야 할 것으로 보인다

# ## director와 year의 관계 알아보기

# In[177]:


train["director"].nunique()


# In[178]:


user_year = train.groupby(["director"]).agg({"year": min_max_year})
user_year.head(20)


# * 감독 별 제작한 영화의 출시 년도 정보이다
# * 어떠한 의미를 유도할 수 있을까..?

# ## genre와 year의 관계 알아보기

# In[179]:


train["genre"].nunique()


# In[180]:


genre_list = list(train["genre"].unique())
min_year = years["year"].min()
max_year = years["year"].max()


# In[185]:


tmp_genres = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")
tmp_years = pd.read_csv(os.path.join(data_path, "new_years.tsv"), sep="\t")
genre_data = pd.merge(tmp_genres, tmp_years, how="inner")
genre_data


# In[187]:


for i in range(18):
    specific_genre_data = genre_data[genre_data["genre"] == genre_list[i]]
    sns.countplot(
        data=specific_genre_data, x="year", order=range(min_year, max_year, 5)
    ).set_title(genre_list[i])
    plt.xticks(rotation=45)
    plt.show()


# * 해당 장르가 유행하던 시기를 대략적으로 파악해볼 수는 있음
# * 표본 수가 적은 장르의 경우, 주의해서 봐야함

# ## writer와 year의 관계 알아보기

# In[182]:


train["writer"].nunique()


# In[183]:


user_year = train.groupby(["writer"]).agg({"year": min_max_year})
user_year.head(20)


# * 작가 별 대본을 작성한 영화의 출시 년도 정보이다
# * 어떠한 의미를 유도할 수 있을까..?

# In[ ]:
