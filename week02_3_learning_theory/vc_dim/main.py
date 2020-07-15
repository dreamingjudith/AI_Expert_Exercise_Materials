import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from knn import knn_train
from logistic_regression import logistic_regression_train
from decision_tree import decision_tree_train


def vc_dimension_upper_bound(train_loss, vc_dim, sample_count, failure_rate):
    upper_bound = train_loss \
        + np.sqrt((2*vc_dim*np.log(np.exp(1)*sample_count)/vc_dim)/sample_count) \
        + np.sqrt((np.log(1/failure_rate))/(2*sample_count))

    return upper_bound


def main():
    failure_rate = 0.05

    dimension_list = list()
    train_count_list = list()

    logistic_regression_loss_train = list()
    knn_loss_train = list()
    decision_tree_loss_train = list()

    logistic_regression_upper = list()
    knn_upper = list()
    decision_tree_upper = list()

    logistic_regression_loss_test = list()
    knn_loss_test = list()
    decision_tree_loss_test = list()

    for i in range(1, 10):
        for j in range(10, 101, 20):
            dimensions = i
            train_count = j
            test_count = j

            train_result_logistic, test_result_logistic = logistic_regression_train(
                dimensions, train_count, test_count)
            train_result_knn, test_result_knn = knn_train(
                dimensions, train_count, test_count)
            train_result_dt, test_result_dt, average_terminal_count = decision_tree_train(
                dimensions, train_count, test_count)

            vc_dimension_logistic = i+1
            vc_dimension_knn = j
            vc_dimension_dt = average_terminal_count

            upper_bound_logistic = vc_dimension_upper_bound(
                train_result_logistic[0], vc_dimension_logistic, train_count, failure_rate)
            upper_bound_knn = vc_dimension_upper_bound(
                train_result_knn[0], vc_dimension_knn, train_count, failure_rate)
            upper_bound_dt = vc_dimension_upper_bound(
                train_result_dt[0], vc_dimension_dt, train_count, failure_rate)

            dimension_list.append(dimensions)
            train_count_list.append(train_count)

            logistic_regression_loss_train.append(train_result_logistic[0])
            logistic_regression_upper.append(upper_bound_logistic)
            logistic_regression_loss_test.append(test_result_logistic[0])
            
            knn_loss_train.append(train_result_knn[0])
            knn_upper.append(upper_bound_knn)
            knn_loss_test.append(test_result_knn[0])

            decision_tree_loss_train.append(train_result_dt[0])
            decision_tree_upper.append(upper_bound_dt)
            decision_tree_loss_test.append(test_result_dt[0])

    result_data = [dimension_list, train_count_list,
                   logistic_regression_loss_train, logistic_regression_upper, logistic_regression_loss_test,
                   knn_loss_train, knn_upper, knn_loss_test,
                   decision_tree_loss_train, decision_tree_upper, decision_tree_loss_test]

    result_data = np.array(result_data)
    result_data = result_data.T

    result_column_name = ["dimensions", "train_count",
                          "logistic_train_loss", "logistic_upper_bound", "logistic_test_loss",
                          "knn_train_loss", "knn_upper_bound", "knn_test_loss",
                          "tree_train_loss", "tree_upper_bound", "tree_test_loss"]

    result_DF = pd.DataFrame(result_data, columns=result_column_name)

    print(result_DF)


if __name__ == "__main__":
    main()
