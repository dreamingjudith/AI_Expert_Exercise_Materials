import numpy as np
import matplotlib.pyplot as plt


from App.Pre_processing.data_generation import create_toy_data
from App.evaluation import *
from App.plot import plot


# Model
from App.logistic_regressior import LogisticRegression


def logistic_regression_train(dimensions, train_count, test_count, iterations=10):
    
    # For evaluation
    BCE_list_train = []
    BCE_list_test = []

    for i in range(iterations):
        # File import
        train_x_data, train_y_data = create_toy_data(dimensions, train_count, test_count, add_outliers=True, training=True)
        test_x_data, test_y_data = create_toy_data(dimensions, train_count, test_count, add_outliers=True, training=False)
        
        # Feature engineering
        X_train = train_x_data
        X_test = test_x_data
        
        # Model
        logistic_regression = LogisticRegression()

        # Training (Learning)
        logistic_regression.fit(X_train, train_y_data)
        
        # Predicting a training set
        y_hat_train = logistic_regression.classify(X_train)

        # Predicting a test set
        y_hat_test = logistic_regression.classify(X_test)
        
        # Evaluating
        BCE_value_train = binary_cross_entropy(train_y_data, y_hat_train)
        BCE_value_test = binary_cross_entropy(test_y_data, y_hat_test)
        
        # Appending
        BCE_list_train.append(BCE_value_train)
        BCE_list_test.append(BCE_value_test)
        
        # Plotting
        if dimensions == 2 and i == 0:
            x1_test, x2_test = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))
            X_test_plot = np.array([x1_test, x2_test]).reshape(2, -1).T

            # logistic regression
            y_hat_plot = logistic_regression.classify(X_test_plot)
            plot(train_x_data, train_y_data, test_x_data, test_y_data, x1_test, x2_test, y_hat_plot, "./Results/logits_result")
            

    # average MSE 
    (average_BCE_train, BCE_std_train) = average_metric(BCE_list_train)
    (average_BCE_test, BCE_std_test) = average_metric(BCE_list_test)
    

    return (average_BCE_train, BCE_std_train), (average_BCE_test, BCE_std_test)

    
if __name__ == "__main__":
    # For evaluation
    dimensions = 2
    train_count = 100
    test_count = 20

    # Training 
    model_train_result, model_test_result = logistic_regression_train(dimensions, train_count, test_count)

    # Print
    print('*'*10, "dimensions : {}".format(dimensions), "train_count : {}".format(train_count), "*"*10)
    print('*'*25, "train", "*"*25)
    print(
        '[Average BCE] \n',
        '{:7.4f} \n'.format(model_train_result[0]),
        '[BCE_std] \n',
        '{:7.4f}'.format(model_train_result[1]))

    print('*'*25, "test", "*"*25)
    print(
        '[Average BCE] \n',
        '{:7.4f} \n'.format(model_test_result[0]),
        '[BCE_std] \n',
        '{:7.4f}'.format(model_test_result[1]))