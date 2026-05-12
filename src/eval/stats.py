import numpy as np
from scipy import stats


def paired_t_test(a, b):
    return stats.ttest_rel(a, b)


def bootstrap_ci(data, n=1000):
    samples = []
    data = np.array(data)
    for _ in range(n):
        s = np.random.choice(data, size=len(data), replace=True)
        samples.append(np.mean(s))
    return np.percentile(samples, [2.5, 97.5])


def wilcoxon(a, b):
    return stats.wilcoxon(a, b)