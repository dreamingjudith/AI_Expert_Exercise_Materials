import numpy as np

def create_toy_data(dimensions=2, train_count = 25, test_count = 10, add_outliers=False, training=True):
    # x0 \in N(-1, 1) + \xi
    # X1 \in N(1, 1) + \xi
    outlier_count = np.int(train_count * 0.4) if training else np.int(test_count * 0.4) 

    if training :
        x0 = np.random.normal(size=(train_count, dimensions)) - .5
        x1 = np.random.normal(size=(train_count, dimensions)) + .5

        if add_outliers:
            x_1 = np.random.normal(size=(outlier_count, dimensions)) + np.array([10] * dimensions) #braodcasting
            return np.concatenate([x0, x1, x_1]), np.concatenate([np.zeros(np.int(train_count)), np.ones(np.int(train_count) + np.int(outlier_count))]).astype(np.int)   
    
        return np.concatenate([x0, x1]), np.concatenate([np.zeros(np.int(train_count)), np.ones(np.int(train_count))]).astype(np.int)
    
    else:
        x0 = np.random.normal(size=(test_count, dimensions)) - .5
        x1 = np.random.normal(size=(test_count, dimensions)) + .5
        
        if add_outliers:
            x_1 = np.random.normal(size=(outlier_count, dimensions)) + np.array([10] * dimensions) #braodcasting
            return np.concatenate([x0, x1, x_1]), np.concatenate([np.zeros(np.int(test_count)), np.ones(np.int(test_count) + np.int(outlier_count))]).astype(np.int)   
    
        return np.concatenate([x0, x1]), np.concatenate([np.zeros(np.int(test_count)), np.ones(np.int(test_count))]).astype(np.int)



if __name__ == '__main__':
    train_x, train_y = create_toy_data(2, True, True)
    print(np.shape(train_x))
    print(np.shape(train_y))
    train_x_2, train_y_2 = create_toy_data(3, True, True)
    print(np.shape(train_x_2))
    print(np.shape(train_y_2))

    train_x_2, train_y_2 = create_toy_data(3, False, True)
    print(np.shape(train_x_2))
    print(np.shape(train_y_2))

    train_x_2, train_y_2 = create_toy_data(4, False, False)
    print(np.shape(train_x_2))
    print(np.shape(train_y_2))