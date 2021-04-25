import pandas as pd
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Model
from deepmatch.models import *
import dataread
embedding_dim=32
SEQ_LEN=50
def gen_user_columns():
    samples_data=dataread.readdata()
    user_feature_columns=[
        SparseFeat('user_id',max(samples_data["user_id"])+1,embedding_dim),
        SparseFeat("gender", max(samples_data["gender"])+1, embedding_dim),
                            SparseFeat("age", max(samples_data["age"])+1, embedding_dim),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', max(samples_data["movie_id"])+1, embedding_dim,\
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
    ]
    return user_feature_columns

def gen_item_columns():
    samples_data=dataread.readdata()
    item_feature_columns = [SparseFeat('movie_id', max(samples_data["movie_id"])+1, embedding_dim), \
                       SparseFeat('movie_type_id', max(samples_data["movie_type_id"])+1, embedding_dim)]
    return item_feature_columns

user_feature=gen_user_columns()
print(user_feature)