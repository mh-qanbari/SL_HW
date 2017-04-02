import numpy as np

g_TRAIN_FILENAME = "Dataset-Question1/trainingData.txt"
g_TEST_FILENAME = "Dataset-Question1/testingData.txt"
g_TRAIN_SIZE = 200
g_TEST_SIZE = 100
g_FEATURE_SIZE = 6


def count(ds, feature_index, value, filter):
    """ Count value of feature_index with a filter
    :param ds: 
    :param feature_index: 
    :param value: 
    :param filter: contains a feature index and value of that.
    :type filter: tuple
    :return: number of matched instance and number of instance that the feature_index feature is value
    :rtype: tuple
    """
    matched_count = 0
    total = 0
    for instance in ds:
        if instance[filter[0]] == filter[1]:
            total += 1
            if instance[feature_index] == value:
                matched_count += 1
    return matched_count, total

if __name__ == "__main__":
    """ [y, x1, x2, ...]
        y: instance label
        x1: first feature value
        x2: second feature value
        ...
    """
    ds_training = np.zeros((g_TRAIN_SIZE, g_FEATURE_SIZE), np.int8)
    ds_testing = np.zeros((g_TEST_SIZE, g_FEATURE_SIZE), np.int8)

    """ [(q(y=0), q(y=1)) , (q1(y,0), q1(y,1)), (q2(y,0), q2(y,1)), ...]
        q(y=0): n(y=0) / N
        q(y=1): n(y=1) / N
        q1(y,0): n(x1=0 | y) / n(x1=0)
        q1(y,1): n(x1=1 | y) / n(x1=1)
        q2(y,0): n(x2=0 | y) / n(x2=0)
        q2(y,1): n(x2=1 | y) / n(x2=1)
        ...
    """
    naive_bayes_table_list = [] # np.zeros(g_FEATURE_SIZE)

    # Load training data
    with open(g_TRAIN_FILENAME, mode='r') as training_file:
        i = 0
        for row in training_file:
            cells = np.array(row.split(), np.int8)
            # print cells
            ds_training[i, :] += cells
            i += 1
        # print ds_training

    # Load testing data
    with open(g_TEST_FILENAME, mode='r') as testing_file:
        i = 0
        for row in testing_file:
            cells = np.array(row.split(), np.int8)
            ds_testing[i, :] += cells
            i += 1

    # Naive Bayes Model
    y0 = 0
    y1 = 0
    for instance in ds_training:
        if instance[0] == 0:
            y0 += 1
        else:
            y1 += 1
    N = y0 + y1
    naive_bayes_table_list.append((1.0 * y0 / N, 1.0 * y1 / N))
    for feature_index in range(1, g_FEATURE_SIZE):
        c0, c0_t = count(ds_training, feature_index, 0, (0, 0))
        c1, c1_t = count(ds_training, feature_index, 0, (0, 1))
        naive_bayes_table_list.append((1. * c0 / c0_t, 1. * (c0_t - c0) / c0_t, 1. * c1 / c1_t, 1. * (c1_t - c1) / c1_t))
    # print naive_bayes_table_list

    # Prediction
    # p(y|X) = p(y) * p(x0|y) * p(x1|y)...
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for instance in ds_testing:
        p0 = naive_bayes_table_list[0][0]
        for feature_index in range(1, g_FEATURE_SIZE):
            cell_index = instance[feature_index]
            p0 *= naive_bayes_table_list[feature_index][cell_index]
        p1 = naive_bayes_table_list[0][1]
        for feature_index in range(1, g_FEATURE_SIZE):
            cell_index = instance[feature_index] + 2
            p1 *= naive_bayes_table_list[feature_index][cell_index]
        if p1 > p0:     # positive
            result = (instance[0] == 1)
            if result:  # true
                tp += 1
            else:       # false
                fp += 1
            # print 1, instance[0] == 1
        else:           # negative
            result = (instance[0] == 0)
            if result:  # true
                tn += 1
            else:       # false
                fn += 1
            # print 0, instance[0] == 0
    print 100. * (tp + tn) / (tp + fp + tn + fn)
