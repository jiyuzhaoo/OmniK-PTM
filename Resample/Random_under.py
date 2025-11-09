from imblearn.under_sampling import RandomUnderSampler
import numpy as np


def random_undersample_with_indices(sampling_strategy, X, Y, random_state=42):

    under = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)

    X_resampled, y_resampled = under.fit_resample(X, Y)


    indices = []
    for resampled_point in X_resampled:
        index = np.where((X == resampled_point).all(axis=1))[0][0]
        indices.append(index)


    return X_resampled, y_resampled, indices