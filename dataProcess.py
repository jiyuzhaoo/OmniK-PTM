import re
from collections import Counter
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from Resample.Random_under import random_undersample_with_indices
from Resample.adaptive_sampler import EfficientAdaptiveClusterSampler

def extract_file_number(filename):
    match = re.search(r'\((\d+)\)', filename)
    if match:
        return int(match.group(1))
    return None

def load_data(path):
    seq = []
    Y = []
    files = os.listdir(path)
    for file in files:
        y = extract_file_number(file)
        with open(os.path.join(path, file), 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    continue
                seq.append(line)
                Y.append(y)
    return seq, Y

def feature_extraction(seq, test_PSTAAP=False):
    from Feature_extraction_algorithms.PSTAAP import PSTAAP_feature
    from Feature_extraction_algorithms.Physicochemical import PC_feature
    data2 = PC_feature(seq)
    N = len(seq)
    empty_list_array = [[] for _ in range(N)]
    data = np.array(empty_list_array, dtype=object)
    feature = PSTAAP_feature(seq, test_PSTAAP)
    data = np.hstack((data, feature))
    return data.astype(np.float32), data2.astype(np.float32)

def data_resample(X, Y, sample_strategy):
    if sample_strategy == 1:
        under = ClusterCentroids(
            sampling_strategy={1: 927}, random_state=1, voting='hard',
            estimator=MiniBatchKMeans(n_init=1, random_state=1)
        )
        x_resampled, y_resampled = under.fit_resample(X, Y)
        def find_indices(X, X_resampled):
            indices = []
            for resampled_point in X_resampled:
                index = np.where((X == resampled_point).all(axis=1))[0][0]
                indices.append(index)
            return np.array(indices)
        index_resampled = find_indices(X, x_resampled)
    elif sample_strategy == 2:
        sampling_strategy = {1: 927}
        x_resampled, y_resampled, index_resampled = random_undersample_with_indices(sampling_strategy, X, Y, 4)
    elif sample_strategy == 3:
        sampling_strategy = {1: 927}
        x_resampled, y_resampled, index_resampled = EfficientAdaptiveClusterSampler(
            sampling_strategy, X, Y,
            max_k=8,
            subset_size=1500,
            smooth_factor=10.0
        )
    else:
        original_resampled = np.arange(X.shape[0])
        return X, Y, original_resampled
    return x_resampled, y_resampled, index_resampled

def make_ylabel(y):
    Y = np.zeros((y.shape[0], 4))
    for i in range(y.shape[0]):
        if y[i] == 1:
            Y[i] = np.array([1, 0, 0, 0])
        elif y[i] == 2:
            Y[i] = np.array([0, 1, 0, 0])
        elif y[i] == 3:
            Y[i] = np.array([0, 0, 1, 0])
        elif y[i] == 4:
            Y[i] = np.array([0, 0, 0, 1])
        elif y[i] == 5:
            Y[i] = np.array([1, 1, 0, 0])
        elif y[i] == 6:
            Y[i] = np.array([1, 0, 1, 0])
        elif y[i] == 7:
            Y[i] = np.array([1, 0, 0, 1])
        elif y[i] == 8:
            Y[i] = np.array([0, 1, 1, 0])
        elif y[i] == 9:
            Y[i] = np.array([1, 1, 1, 0])
        elif y[i] == 10:
            Y[i] = np.array([1, 1, 0, 1])
        elif y[i] == 11:
            Y[i] = np.array([1, 1, 1, 1])
    return Y

def process_data(train_path=None, test_path=None, sample_strategy=0):
    X_train, X_test, Y_train, Y_test = None, None, None, None
    label_train, label_test = None, None
    X_train2, original_train_X, original_train_Y, original_train_label, original_train_X2, X_test2 = [None] * 6
    if train_path:
        seq_train, Y_train = load_data(train_path)
        X_train, X_train2 = feature_extraction(seq_train)
        Y_train = np.array(Y_train)
        original_train_X = X_train.astype(np.float32)
        original_train_X2 = X_train2.astype(np.float32)
        original_train_Y = Y_train
        X_train, Y_train, index = data_resample(X_train, Y_train, sample_strategy)
        X_train2 = X_train2[index]
        label_train = make_ylabel(Y_train)
        original_train_label = make_ylabel(original_train_Y)
    if test_path:
        seq_test, Y_test = load_data(test_path)
        X_test, X_test2 = feature_extraction(seq_test, test_PSTAAP=True)
        Y_test = np.array(Y_test)
        label_test = make_ylabel(Y_test)
    if train_path:
        return X_train, Y_train, label_train, X_test, Y_test, label_test, original_train_X, original_train_Y, original_train_label, X_train2, original_train_X2, X_test2
    return X_test, Y_test, label_test
