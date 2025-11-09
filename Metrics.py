from sklearn.metrics import precision_score, recall_score, hamming_loss
import numpy as np
import warnings

class Metrics:
    def __init__(self):

        self.Aiming = 0
        self.Coverage = 0
        self.Acc = 0
        self.A_T = 0
        self.A_F = 0

        self.class_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}
        self.class_correct_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}
        self.class_absolute_true_rates = {}
        self.label_mapping = {
            np.array([1, 0, 0, 0], dtype=np.float64).tobytes(): 1,
            np.array([0, 1, 0, 0], dtype=np.float64).tobytes(): 2,
            np.array([0, 0, 1, 0], dtype=np.float64).tobytes(): 3,
            np.array([0, 0, 0, 1], dtype=np.float64).tobytes(): 4,
            np.array([1, 1, 0, 0], dtype=np.float64).tobytes(): 5,
            np.array([1, 0, 1, 0], dtype=np.float64).tobytes(): 6,
            np.array([1, 0, 0, 1], dtype=np.float64).tobytes(): 7,
            np.array([0, 1, 1, 0], dtype=np.float64).tobytes(): 8,
            np.array([1, 1, 1, 0], dtype=np.float64).tobytes(): 9,
            np.array([1, 1, 0, 1], dtype=np.float64).tobytes(): 10,
            np.array([1, 1, 1, 1], dtype=np.float64).tobytes(): 11
        }


    @staticmethod
    def accuracy(y_true, y_pred):  # Hamming Score
        temp = 0
        for i in range(y_true.shape[0]):
            temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
        return temp / y_true.shape[0]

    @staticmethod
    def absolute_true(y_true, y_pred):
        count = 0
        for i in range(0, y_pred.shape[0]):
            if (y_pred[i] == y_true[i]).all():
                count += 1
        return count / y_true.shape[0]

    def calculate_metrics(self, y_true, y_pred):
        self.Aiming += precision_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=1)
        self.Coverage += recall_score(y_true=y_true, y_pred=y_pred, average='samples')
        self.Acc += self.accuracy(y_true, y_pred)
        self.A_T += self.absolute_true(y_true, y_pred)
        self.A_F += hamming_loss(y_true, y_pred)

    def transform_format(self, is_kfold=False):
        if is_kfold:
            self.Aiming = "{:.2f}%".format(self.Aiming /5 * 100)
            self.Coverage = "{:.2f}%".format(self.Coverage /5 * 100)
            self.Acc = "{:.2f}%".format(self.Acc /5 * 100)
            self.A_T = "{:.2f}%".format(self.A_T /5 * 100)
            self.A_F = "{:.2f}%".format(self.A_F /5 * 100)
        else:
            self.Aiming = "{:.2f}%".format(self.Aiming * 100)
            self.Coverage = "{:.2f}%".format(self.Coverage * 100)
            self.Acc = "{:.2f}%".format(self.Acc * 100)
            self.A_T = "{:.2f}%".format(self.A_T * 100)
            self.A_F = "{:.2f}%".format(self.A_F * 100)
        # print(
        #     f"[INFO]\tAiming:{self.Aiming},Coverage:{self.Coverage},Accuracy:{self.Acc},Absolute_True:{self.A_T},Absolute_False:{self.A_F}")
        print(
            f"[INFO]\t{self.Aiming},{self.Coverage},{self.Acc},{self.A_T},{self.A_F}")

    def accumulate_counts(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
        for i in range(y_true.shape[0]):

            label_idx = self.label_mapping.get(y_true[i].tobytes())

            if (y_true[i] == y_pred[i]).all():
                self.class_correct_counts[label_idx] += 1

            self.class_counts[label_idx] += 1

    def calculate_each_class_absolute_true_rate(self):
        for label_idx in self.class_counts:
            if self.class_counts[label_idx] != 0:
                self.class_absolute_true_rates[label_idx] = "{:.2f}%".format(
                    (self.class_correct_counts[label_idx] / self.class_counts[label_idx]) * 100)
            else:
                self.class_absolute_true_rates[label_idx] = "{:.2f}%".format(0 * 100)
        return list(self.class_absolute_true_rates.values())


def calculate_custom_metrics(true_labels, predicted_labels, max_tags=4):
    true_labels = true_labels
    predicted_labels = predicted_labels


    metrics = []


    true_counts = np.sum(true_labels == 1, axis=1)

    for j in range(max_tags, 0, -1):
        P_j_sum = 0
        C_j_sum = 0

        for k in range(j, max_tags + 1):

            valid_base_samples = true_counts >= k

            match_counts = np.sum((true_labels == 1) & (predicted_labels == 1), axis=1)

            P_k = np.sum((match_counts >= j) & valid_base_samples)

            C_k = np.sum(valid_base_samples)

            P_j_sum += P_k
            C_j_sum += C_k

        MR_j = P_j_sum / C_j_sum if C_j_sum > 0 else 0
        metrics.append(MR_j)

    return metrics