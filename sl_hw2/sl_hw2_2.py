import numpy as np

g_DATA_FEATURES_FILENAME = "Dataset - Question2/orders_train.txt"
g_DATA_CLASS_FILENAME = "Dataset - Question2/orders_class.txt"

true = True
false = False

if __name__ == "__main__":
    with open(g_DATA_FEATURES_FILENAME, 'r') as features_file:
        __ignore_feature = true
        for line in features_file:
            if __ignore_feature:
                __ignore_feature = false
                continue
            feature_list = line.split(';')

            for feature in feature_list:

            break
