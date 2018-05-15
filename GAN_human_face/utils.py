import pickle
import numpy as np

def save_pickle(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def read_pickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


