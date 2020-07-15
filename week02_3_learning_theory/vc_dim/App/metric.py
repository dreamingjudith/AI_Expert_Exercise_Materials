import numpy as np

# Calculate the Gini index for a split dataset
def gini_index(groups):
	'''
		Gini index

		Args:
			groups : (left, right) dataset [n_datapoints, n_features + 1], [n_datapoints, n_features + 1]

		Returns:
			gini : gini index (scalar)
	'''

	# count all samples at split point
	n_instances = np.sum([np.shape(group)[0] for group in groups])
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(np.shape(group)[0])
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		(values, counts) = np.unique(group[:,-1], return_counts=True)
		for ind in range(np.size(values)):
			p = counts[ind] / np.sum(counts)
			score += p * p

		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)

		assert gini >= 0
		
	return gini