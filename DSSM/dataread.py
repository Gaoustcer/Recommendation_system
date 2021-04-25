from random import sample
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import plot_model
def readdata():
    #print("Hello,the filepath is %s",filepath)
    samples_data=pd.read_csv("PyTorch_Practice\\DSSM\\samples.txt",sep="\t",header=None)
    samples_data.columns=["user_id","gender","age","hist_movie_id","hist_len","movie_id","movie_type_id","label"]
    samples_data.head()
    samples_data=shuffle(samples_data)
    #train_model_input={"user_i"}
    return samples_data

def getlabelandinput(testdata):
    testdata=readdata("PyTorch_Practice\\DSSM\\samples.txt")
    X=testdata[["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id"]]
    Y=testdata["label"]
    train_model_input={"user_id":np.array(X["user_id"]),"gender":np.array(X["gender"]),"age":np.array(X["age"]),\
        "hist_movie_id":np.array([[int(i) for i in l.split(',')] for l in X["hist_movie_id"]]),\
        "hist_len": np.array(X["hist_len"]), \
        "movie_id": np.array(X["movie_id"]), \
        "movie_type_id": np.array(X["movie_type_id"])
        }
    label=np.array(Y)
    return train_model_input,label