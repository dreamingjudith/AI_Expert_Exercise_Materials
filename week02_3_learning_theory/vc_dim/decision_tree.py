import numpy as np
import matplotlib.pyplot as plt


from App.Pre_processing.data_generation import create_toy_data
from App.evaluation import *
from App.plot import plot


# Model
from App.decision_tree_maker import *
from App.metric import gini_index



def decision_tree_train(dimensions, train_count, test_count, iterations=10):
    # Set Hyperparameters
    max_depth = 5
    min_size = 10
    
    # For evaluation
    BCE_list_train = []
    BCE_list_test = []
    Terminal_count_list = []

    for i in range(iterations):
        # File import
        train_x_data, train_y_data = create_toy_data(dimensions, train_count, test_count, add_outliers=True, training=True)
        test_x_data, test_y_data = create_toy_data(dimensions, train_count, test_count, add_outliers=True, training=False)
        
        # Feature engineering
        X_train = train_x_data
        X_test = test_x_data
        
        # Model
        decision_tree = Decision_tree()

        # combine the features and targets
        train_y_data_tree = train_y_data[:, None]
        Tree_input = np.hstack((X_train, train_y_data_tree))

        # Training (Learning)
        tree, terminal_count = decision_tree.build_tree(Tree_input, max_depth, min_size, np.shape(Tree_input)[1]-1)
        
        # Predicting a training set    
        y_hat_train = decision_tree.predicts(tree, X_train)

        # Predicting a test set
        y_hat_test = decision_tree.predicts(tree, X_test)
        
        # Evaluating
        BCE_value_train = binary_cross_entropy(train_y_data, y_hat_train)
        BCE_value_test = binary_cross_entropy(test_y_data, y_hat_test)
        
        # Appending
        BCE_list_train.append(BCE_value_train)
        BCE_list_test.append(BCE_value_test)
        Terminal_count_list.append(terminal_count)
        
        # Plotting
        if dimensions == 2 and i == 0:
            x1_test, x2_test = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))
            X_test_plot = np.array([x1_test, x2_test]).reshape(2, -1).T

            # logistic regression
            y_hat_plot = decision_tree.predicts(tree, X_test_plot)
            plot(train_x_data, train_y_data, test_x_data, test_y_data, x1_test, x2_test, y_hat_plot, "./Results/tree_result")
            

    # average MSE 
    (average_BCE_train, BCE_std_train) = average_metric(BCE_list_train)
    (average_BCE_test, BCE_std_test) = average_metric(BCE_list_test)
    average_terminal_count = np.average(Terminal_count_list)
    

    return (average_BCE_train, BCE_std_train), (average_BCE_test, BCE_std_test), average_terminal_count
  
if __name__ == "__main__":
    # For evaluation
    dimensions = 2
    train_count = 100
    test_count = 20

    model_train_result, model_test_result, average_terminal_count = decision_tree_train(dimensions, train_count, test_count)

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

    print('*'*25, "test", "*"*25)
    print(
        '[Average terminal count] \n',
        '{}'.format(average_terminal_count))