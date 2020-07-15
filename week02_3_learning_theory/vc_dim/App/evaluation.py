import numpy as np

# Calculate accuracy percentage
def accruacy(actual, predicted):
	'''
		Args : 
			actual : np.array [datapoint,]
			predicted : np.array [datapoint, ]

		Returns : 
			accuracy : scalar
	'''
	assert np.shape(actual) == np.shape(predicted)
	# The number of point
	data_count = np.shape(actual)[0]

	# Comaprison
	correct = np.sum(np.equal(actual, predicted))

	return float(correct) / float(data_count) * 100.0

def binary_cross_entropy(y, y_hat):
    BCE = -1 * np.mean(y * np.log(y_hat + 1e-7) + (1-y) * np.log(1 - y_hat + 1e-7))
    return BCE

def average_metric(metrics):
    average_metric = np.mean(metrics)
    std_metric = np.sqrt(np.mean(np.square(metrics - average_metric)))

    return (average_metric, std_metric)

