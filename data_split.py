import numpy as np
from sklearn.utils import shuffle
"""This is to split the data into train and test, future work need to add stratified split, cross validation"""

def train_test_split(df):
    """Split the given dataframe into train test or validation"""
    shuffled_features = shuffle(df)
    shuffled_features = shuffled_features.fillna(0)
    dataframe_to_split = np.random.rand(len(shuffled_features)) < 0.8
    train = shuffled_features[dataframe_to_split]
    test = shuffled_features[~dataframe_to_split]
    # train_validation_split = np.random.rand(len(train_validation))<0.8
    # train = train_validation[train_validation_split]
    # validation = train_validation[~train_validation_split]
    print(f"Number of Samples in Training set: {len(train)}"
          # f"Number of Samples in Validation set: {len(validation)}" 
          f"Number of Samples in Test set: {len(test)}")

    return train, test
