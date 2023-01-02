from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class EASE:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
        #self.genre_enc = LabelEncoder()
                
    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user'])
        items = self.item_enc.fit_transform(df.loc[:, 'item'])
        #genres = self.genre_enc.fit_transform(df.loc[:, 'genre'])
        
        return users, items

    def fit(self, df, lambda_: float = 0.5, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(df)
        values = (
            np.ones(df.shape[0])
            if implicit
            else df['rating'].to_numpy() / df['rating'].max()
        )

        X = csr_matrix((values, (users, items)))
        self.X = X
        
        G = X.T @ X.toarray()   # G = XX'
        diagIndices = np.diag_indices(G.shape[0])   
        G[diagIndices] += lambda_   # X'X + λI
        P = np.linalg.inv(G)    
        B = P / (-np.diag(P))   # I − Pˆ · diagMat(γ˜)
        B[diagIndices] = 0  # diag(B) = 0

        self.B = B
        self.pred = X.dot(B)

    def predict(self, train, users, items, k):
        items = self.item_enc.transform(items)
        train_df = train.loc[train.user.isin(users)]
        train_df['ci'] = self.item_enc.transform(train_df.item)
        train_df['cu'] = self.user_enc.transform(train_df.user)
        groupby_user = train_df.groupby('user')
        
        user_preds = pd.DataFrame()
        for user, group in tqdm(groupby_user):
            watched = set(group['ci'])  # The movie users watched
            candidates = [item for item in items if item not in watched]
            
            predict_user = group.cu.iloc[0]
            pred = np.take(self.pred[predict_user,:], candidates)
            res = np.argpartition(pred, -k)[-k:] # top-K
            r = pd.DataFrame(
                {
                    "user": [user] * len(res),
                    "item": np.take(candidates, res),
                    "score": np.take(pred, res),
                }
            ).sort_values('score', ascending=False)
            user_preds = pd.concat([user_preds,r])

        user_preds['item'] = self.item_enc.inverse_transform(user_preds['item'])
        

        return user_preds 
        