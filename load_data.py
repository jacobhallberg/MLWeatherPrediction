import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

def load_data(data_location):
    return pd.read_csv(data_location)

def min_max_normalization(df):
    return (df - df.min()) / (df.max() - df.min())

def load_normalized_data(data_location, y_variable):
    df = load_data(data_location)
    y_df = df[y_variable]
    X_df = df.drop(y_variable, axis=1)
    
    # Matrix Math is easier with floats instead of strings. Convert String Values to encoded labels.
    le = LabelEncoder()
    le.fit(y_df)
    y_df = le.transform(y_df)
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y_df),y_df)
    
#     y_df = pd.DataFrame(to_categorical(y_df, len(np.unique(y_df, return_counts=True)[0])))
    
    
    normalized_X_df = min_max_normalization(X_df)

    x_train, x_test, y_train, y_test = train_test_split(normalized_X_df.values, y_df, test_size=0.2)
    
    return x_train, x_test, y_train, y_test, np.unique(y_df).shape[0], class_weights
