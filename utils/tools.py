import numpy as np

def compute_accuracy_f1(tp: int, fp: int, tn: int, fn: int):
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    if tp == 0:
        f1 = 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
    return acc, f1

def mean_and_var(values):
    # if not values:
    #     return 0.0, 0.0
    # m = sum(values) / len(values)
    # if len(values) == 1:
    #     return m, 0.0
    # var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    m = np.mean(values)
    var = np.var(values, ddof=1) if len(values) > 1 else 0.0
    return m, var