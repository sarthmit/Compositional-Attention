import sys
import random
import numpy as np

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True
# Logging utility
from util import log

"""
Combinatorics:
For n objects, there are n^2 possible combinations, n of which will be 'same' trials.
To balance 'same' and 'different' trials, create copies of 'same' trials.
Number of 'different' trials = n * (n - 1).
Number of 'same' trials = n.
To balance trial types, create ((n * (n - 1)) / n) copies of 'same' trials.
Number of total possible trials = (n * (n - 1)) * 2.
For training set, n will actually be (n - m).
For test set, n will actually be m.
"""

# Dimensionality of multiple-choice output
y_dim = 2
# Sequence length
seq_len = 2

# Task generator
def create_task(args, train_shapes, test_shapes):

	log.info('n_shapes = ' + str(args.n_shapes) + '...')
	log.info('m_holdout = ' + str(args.m_holdout) + '...')
	# If m = 0, training and test sets are drawn from same set of shapes
	if args.m_holdout == 0:
		# Total number of possible trials
		shapes_avail = args.n_shapes
		total_trials = (shapes_avail * (shapes_avail - 1)) * 2
		log.info('Total possible trials = ' + str(total_trials) + '...')
		# Proportion of training set size vs. test set size
		train_proportion = args.train_proportion
		test_proportion = 1 - train_proportion
		# Create training/test set sizes
		args.train_set_size = np.round(train_proportion * total_trials).astype(np.int)
		args.test_set_size = np.round(test_proportion * total_trials).astype(np.int)
		log.info('Training set size = ' + str(args.train_set_size) + '...')
		log.info('Test set size = ' + str(args.test_set_size) + '...')
	# Otherwise, training and test sets are completely disjoint (in terms of the shapes that are used)
	else:
		# Ensure that there are enough potential trials for desired training set size (or change training set size)
		shapes_avail = args.n_shapes - args.m_holdout
		total_trials = (shapes_avail * (shapes_avail - 1)) * 2
		log.info('Total possible training trials = ' + str(total_trials) + '...')
		if args.train_set_size > total_trials:
			log.info('Desired training set size (' + str(args.train_set_size) + ') is larger than total number of possible training trials for this task (' + str(total_trials) + ')...')
			log.info('Changing training set size to ' + str(total_trials) + '...')
			args.train_set_size = total_trials
		else:
			log.info('Training set size = ' + str(args.train_set_size) + '...')
		# Ensure that there are enough potential trials for desired test set size (or change test set size)
		shapes_avail = args.n_shapes - (args.n_shapes - args.m_holdout)
		total_trials = (shapes_avail * (shapes_avail - 1)) * 2
		log.info('Total possible test trials = ' + str(total_trials) + '...')
		if args.test_set_size > total_trials:
			log.info('Desired test set size (' + str(args.test_set_size) + ') is larger than total number of possible test trials for this task (' + str(total_trials) + ')...')
			log.info('Changing test set size to ' + str(total_trials) + '...')
			args.test_set_size = total_trials
		else:
			log.info('Test set size = ' + str(args.test_set_size) + '...')

	# If m = 0, training and test sets are drawn from same set of shapes
	if args.m_holdout == 0:
		# Create all possible trials
		same_trials = []
		diff_trials = []
		for shape1 in train_shapes:
			for shape2 in train_shapes:
				if shape1 == shape2:
					same_trials.append([shape1, shape2])
				else:
					diff_trials.append([shape1, shape2])
		# Shuffle
		random.shuffle(same_trials)
		random.shuffle(diff_trials)
		# Split trials for train and test sets
		same_trials_train = same_trials[:np.round(train_proportion * len(same_trials)).astype(np.int)]
		same_trials_test = same_trials[np.round(train_proportion * len(same_trials)).astype(np.int):]
		diff_trials_train = diff_trials[:np.round(train_proportion * len(diff_trials)).astype(np.int)]
		diff_trials_test = diff_trials[np.round(train_proportion * len(diff_trials)).astype(np.int):]
	# Otherwise, training and test sets are completely disjoint (in terms of the shapes that are used), and can be generated separately
	else:
		# Create all possible training trials
		same_trials_train = []
		diff_trials_train = []
		for shape1 in train_shapes:
			for shape2 in train_shapes:
				if shape1 == shape2:
					same_trials_train.append([shape1, shape2])
				else:
					diff_trials_train.append([shape1, shape2])
		# Shuffle
		random.shuffle(same_trials_train)
		random.shuffle(diff_trials_train)
		# Create all possible test trials
		same_trials_test = []
		diff_trials_test = []
		for shape1 in test_shapes:
			for shape2 in test_shapes:
				if shape1 == shape2:
					same_trials_test.append([shape1, shape2])
				else:
					diff_trials_test.append([shape1, shape2])
		# Shuffle
		random.shuffle(same_trials_test)
		random.shuffle(diff_trials_test)
	# Duplicate 'same' trials to match number of 'different' trials
	same_trials_train_balanced = []
	for t in range(len(diff_trials_train)):
		same_trials_train_balanced.append(same_trials_train[np.floor(np.random.rand()*len(same_trials_train)).astype(np.int)])
	same_trials_test_balanced = []
	for t in range(len(diff_trials_test)):
		same_trials_test_balanced.append(same_trials_test[np.floor(np.random.rand()*len(same_trials_test)).astype(np.int)])
	# Combine all same and different trials for training set
	all_train_seq = []
	all_train_targ = []
	for t in range(len(same_trials_train_balanced)):
		all_train_seq.append(same_trials_train_balanced[t])
		all_train_targ.append(0)
	for t in range(len(diff_trials_train)):
		all_train_seq.append(diff_trials_train[t])
		all_train_targ.append(1)
	# Combine all same and different trials for test set
	all_test_seq = []
	all_test_targ = []
	for t in range(len(same_trials_test_balanced)):
		all_test_seq.append(same_trials_test_balanced[t])
		all_test_targ.append(0)
	for t in range(len(diff_trials_test)):
		all_test_seq.append(diff_trials_test[t])
		all_test_targ.append(1)
	# Shuffle trials in training set
	train_ind = np.arange(len(all_train_seq))
	np.random.shuffle(train_ind)
	all_train_seq = np.array(all_train_seq)[train_ind]
	all_train_targ = np.array(all_train_targ)[train_ind]
	# Shuffle trials in test set
	test_ind = np.arange(len(all_test_seq))
	np.random.shuffle(test_ind)
	all_test_seq = np.array(all_test_seq)[test_ind]
	all_test_targ = np.array(all_test_targ)[test_ind]
	# Select subset if desired dataset size is smaller than number of all possible trials
	if (args.train_set_size + args.test_set_size) < total_trials:
		all_train_seq = all_train_seq[:args.train_set_size, :]
		all_train_targ = all_train_targ[:args.train_set_size]
		all_test_seq = all_test_seq[:args.test_set_size, :]
		all_test_targ = all_test_targ[:args.test_set_size]

	# Create training and test sets
	train_set = {'seq_ind': all_train_seq, 'y': all_train_targ}
	test_set = {'seq_ind': all_test_seq, 'y': all_test_targ}

	return args, train_set, test_set