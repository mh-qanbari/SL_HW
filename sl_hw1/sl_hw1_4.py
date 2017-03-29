import scipy.io
import numpy
import matplotlib.pyplot as plot

g_MNIST_ALL_FILENAME = "mnist_all.mat"
g_TRAIN1 = "train1"
g_TEST1 = "test1"
g_TRAIN2 = "train2"
g_TEST2 = "test2"
g_TRAIN7 = "train7"
g_TEST7 = "test7"
g_IMG_COUNT = 200
g_IMG_WIDTH = 28
g_IMG_HEIGHT = 28


def load_mnist_data_127(class_size, image_width, image_height):
    mat = scipy.io.loadmat(g_MNIST_ALL_FILENAME)
    # train_data = scipy.zeros((class_size, g_IMG_HEIGHT, g_IMG_WIDTH), numpy.float)
    # train_label = scipy.zeros(class_size, numpy.int)
    _data_shape = (class_size, image_height * image_width)
    _train1 = numpy.zeros(_data_shape, numpy.float)
    _train2 = numpy.zeros(_data_shape, numpy.float)
    _train7 = numpy.zeros(_data_shape, numpy.float)
    _i = 0
    for item, value in mat.iteritems():
        if type(value) is not scipy.ndarray:
            continue
        # print "<dataset name>:", item, " | <shape>:", value.shape

        if item == g_TRAIN1:
            _train1 = numpy.array(value[:class_size, :]).reshape(_data_shape)
            _train1 = numpy.concatenate((_train1, numpy.array([[1]] * class_size)), axis=1)

        elif item == g_TRAIN2:
            _train2 = numpy.array(value[:class_size, :]).reshape(_data_shape)
            _train2 = numpy.concatenate((_train2, numpy.array([[2]] * class_size)), axis=1)

        elif item == g_TRAIN7:
            _train7 = numpy.array(value[:class_size, :]).reshape(_data_shape)
            _train7 = numpy.concatenate((_train7, numpy.array([[7]] * class_size)), axis=1)

        else:
            continue
        _i += 1
    return _train1, _train2, _train7


def K_NN(k, data, instance):
    _datum_count = len(data)
    _distance_list = numpy.zeros(_datum_count, numpy.float)
    for _i in range(_datum_count):
        datum = data[_i]
        euclidean_distance = numpy.sum(numpy.power(datum[:-1] - instance, 2))
        _distance_list[_i] = euclidean_distance
    _sorted_indices = numpy.argsort(_distance_list)
    _sorted_labels = data[_sorted_indices[:k], -1]
    _counter = dict()
    for _label in _sorted_labels:
        if _label in _counter:
            _counter[_label] += 1
        else:
            _counter[_label] = 1
    _most_repeated_label = max(_counter)
    return _most_repeated_label

if __name__ == "__main__":
    train_portion = 0.8
    train1, train2, train7 = load_mnist_data_127(g_IMG_COUNT, g_IMG_WIDTH, g_IMG_HEIGHT)
    numpy.random.shuffle(train1)
    numpy.random.shuffle(train2)
    numpy.random.shuffle(train7)
    end_train_index = int(numpy.ceil(g_IMG_COUNT * train_portion))
    data = numpy.concatenate((train1, train2, train7))
    train = numpy.concatenate((train1[:end_train_index], train2[:end_train_index], train7[:end_train_index]))
    test = numpy.concatenate((train1[end_train_index:], train2[end_train_index:], train7[end_train_index:]))

    correct_1 = 0
    correct_3 = 0
    incorrect_1 = 0
    incorrect_3 = 0
    correct_4Indices_1 = [-1] * 4
    incorrect_4Indices_1 = [-1] * 4
    last_correct_4Indices_1 = 0
    last_incorrect_4Indices_1 = 0

    i = -1
    for instance in test:
        i += 1
        # K = 1
        instance_label = K_NN(1, train, instance[:-1])
        if instance_label == instance[-1]:
            correct_1 += 1
            if last_correct_4Indices_1 < 4:
                correct_4Indices_1[last_correct_4Indices_1] = i
                last_correct_4Indices_1 += 1
        else:
            incorrect_1 += 1
            if last_incorrect_4Indices_1 < 4:
                incorrect_4Indices_1[last_incorrect_4Indices_1] = i
                last_incorrect_4Indices_1 += 1
        # K = 3
        instance_label = K_NN(3, train, instance[:-1])
        if instance_label == instance[-1]:
            correct_3 += 1
        else:
            incorrect_3 += 1

    print "a)"
    print "********************************"
    print "WHEN k is 1:"
    print "\tcorrect\t\t", correct_1
    print "\tincorrect\t", incorrect_1
    print "\tACCURACY\t", 1. * correct_1 / (correct_1 + incorrect_1)
    print "********************************"
    print "WHEN k is 3:"
    print "\tcorrect\t\t", correct_3
    print "\tincorrect\t", incorrect_3
    print "\tACCURACY\t", 1. * correct_3 / (correct_3 + incorrect_3)
    print

    print "b)"
    print "4 Correct Predicted Test Instance"
    for index in correct_4Indices_1:
        img = test[index, :-1].reshape((g_IMG_HEIGHT, g_IMG_WIDTH))
        implt = plot.imshow(img, cmap="gray")
        plot.show(implt)
    print "4 Incorrect Predicted Test Instance"
    for index in incorrect_4Indices_1:
        img = test[index, :-1].reshape((g_IMG_HEIGHT, g_IMG_WIDTH))
        implt = plot.imshow(img, cmap="gray")
        plot.show(implt)
    print

    print "c)"
    numpy.random.shuffle(data)
    start_index = 0
    end_index = len(data)
    k_fold = 5
    fold_size = end_index / k_fold
    test_start_index = 0
    test_end_index = fold_size
    plot_avg_x_list = [1, 3, 5, 7]
    average_accuracies = numpy.array([0.] * 4)
    for fold in range(k_fold):
        print "Fold Index :", fold + 1

        plot_x_list = [0] * 4
        plot_y_list = [0] * 4

        _train = numpy.concatenate((data[0: test_start_index], data[test_end_index:]))
        _test = data[test_start_index: test_end_index]
        test_start_index += fold_size
        test_end_index += fold_size

        for k in [1, 3, 5, 7]:
            print "\t%d-NN :" % k
            correct = 0
            incorrect = 0
            for instance in _test:
                instance_label = K_NN(k, train, instance[:-1])
                if instance_label == instance[-1]:
                    correct += 1
                else:
                    incorrect += 1
            accuracy = 1. * correct / (correct + incorrect)
            plot_x_list[k/2] = k
            plot_y_list[k/2] = accuracy
            average_accuracies[k/2] += accuracy
            print "\t\tcorrect\t\t", correct
            print "\t\tincorrect\t", incorrect
            print "\t\taccuracy\t", accuracy
            print

        bar_width = 0.9  # the width of the bars
        fig, ax = plot.subplots()
        rects = ax.bar(plot_x_list, plot_y_list, bar_width, color='g')
        ax.set_ylabel('Accuracy')
        ax.set_title('K')
        ax.set_xticks(plot_x_list)
        plot.show()

    print "Average Accuracy :"
    print "\tk = %d\t\tAccuracy :%.2f" % (1, average_accuracies[0])
    print "\tk = %d\t\tAccuracy :%.2f" % (3, average_accuracies[1])
    print "\tk = %d\t\tAccuracy :%.2f" % (5, average_accuracies[2])
    print "\tk = %d\t\tAccuracy :%.2f" % (7, average_accuracies[3])
    print


    average_accuracies /= k_fold
    bar_width = 0.9  # the width of the bars
    fig, ax = plot.subplots()
    rects = ax.bar(plot_avg_x_list, average_accuracies, bar_width, color='g')
    ax.set_ylabel('Average Accuracy')
    ax.set_title('K')
    ax.set_xticks(plot_avg_x_list)
    plot.show()
