import numpy as np

true = True
false = False

train_data = np.loadtxt("vowel.train.txt", delimiter=',', dtype=np.float, usecols=range(1, 12))
test_data = np.loadtxt("vowel.test.txt", delimiter=',', dtype=np.float, usecols=range(1, 12))

means = []
covs = []
for i in range(1, 12):
    mean = train_data[train_data[:, 0] == i].mean(axis=0)
    means.append(mean)
    cov = np.cov(train_data[train_data[:, 0] == i].transpose())
    covs.append(cov)
cov_matrix = np.cov(train_data[:, 1:].transpose())

for test_instance in test_data:
    x = test_instance[1:]
    sum = 0.0
    for i in range(1, 12):
        mean = means[i-1][1:]
        mean_T = mean.transpose()
        cov_rev = np.linalg.inv(cov_matrix)
        sum += (-0.5 * np.dot(np.dot(mean_T, cov_rev), mean) + np.dot(np.dot(mean_T, cov_rev), x) + np.log(1/11.))
    print sum

'''
# -- QDA --
for test_instance in test_data:
    x = test_instance[1:]
    sum = 0.0
    for i in range(1, 12):
        mean = means[i - 1][1:]
        mean_T = mean.transpose()
        cov = covs[i-1]
        cov_rev = np.linalg.inv(cov)
        print x.shape, mean_T.shape, cov_rev.shape, mean.shape
        sum += (-0.5 * np.log(np.linalg.det(cov)) -0.5 * np.dot(np.dot((x.transpose() - mean_T), cov_rev), (x - mean)) +
                np.log(1/11.))
    print sum
'''
