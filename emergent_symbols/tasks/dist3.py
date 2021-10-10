import sys
import random
import math
import numpy as np
import builtins
from itertools import combinations, permutations

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True
# Logging utility
from util import log

"""
Combinatorics:
For n objects, there are c = nC3 possible combinations of 3 objects.
For each of c combinations, there are p = 3*2*1 permutations.
There are mp = (3*2*1)**2 meta-permutations (allowing for the same permuation to appear in both rows).
Number of total possible trials is mp * c = ((3*2*1)**2) * nC3).
For training set, n will actually be (n - m).
For test set, n will actually be (n - (n-m)).
"""

# Dimensionality of multiple-choice output
y_dim = 4
# Sequence length
seq_len = 9

# Method for calculating number of combinations
def n_comb(n, r):
	return int(math.factorial(n) / (math.factorial(r) * math.factorial(n-r)))

# Task generator
def create_task(args, train_shapes, test_shapes):

	log.info('n_shapes = ' + str(args.n_shapes) + '...')
	log.info('m_holdout = ' + str(args.m_holdout) + '...')
	# If m = 0, training and test sets are drawn from same set of shapes
	if args.m_holdout == 0:
		# Total number of possible trials
		shapes_avail = args.n_shapes
		n_row_comb = n_comb(shapes_avail, 3)
		n_metaperm = (3*2*1)**2
		total_trials = n_metaperm * n_row_comb
		log.info('Total possible trials = ' + str(total_trials) + '...')
		if args.train_set_size + args.test_set_size > total_trials:
			# Proportion of training set size vs. test set size
			train_proportion = args.train_proportion
			test_proportion = 1 - train_proportion
			# Create training/test set sizes
			log.info('Desired training set size (' + str(args.train_set_size) + ') and test set size (' + str(args.test_set_size) + ') combined are larger than total number of possible trials for this task (' + str(total_trials) + ')...')
			args.train_set_size = np.round(train_proportion * total_trials).astype(np.int)
			log.info('Changing training set size to ' + str(args.train_set_size) + '...')
			args.test_set_size = np.round(test_proportion * total_trials).astype(np.int)
			log.info('Changing test set size to ' + str(args.test_set_size) + '...')
		else:
			log.info('Training set size = ' + str(args.train_set_size) + '...')
			log.info('Test set size = ' + str(args.test_set_size) + '...')
	else:
		# Total number of possible training trials
		shapes_avail = args.n_shapes - args.m_holdout
		n_row_comb = n_comb(shapes_avail, 3)
		n_metaperm = (3*2*1)**2
		total_trials = n_metaperm * n_row_comb
		log.info('Total possible training trials = ' + str(total_trials) + '...')
		if args.train_set_size > total_trials:
			log.info('Desired training set size (' + str(args.train_set_size) + ') is larger than total number of possible training trials for this task (' + str(total_trials) + ')...')
			log.info('Changing training set size to ' + str(total_trials) + '...')
			args.train_set_size = total_trials
		else:
			log.info('Training set size = ' + str(args.train_set_size) + '...')
		# Total number of possible training trials
		shapes_avail = args.n_shapes - (args.n_shapes - args.m_holdout)
		n_row_comb = n_comb(shapes_avail, 3)
		n_metaperm = (3*2*1)**2
		total_trials = n_metaperm * n_row_comb
		log.info('Total possible test trials = ' + str(total_trials) + '...')
		if args.test_set_size > total_trials:
			log.info('Desired test set size (' + str(args.test_set_size) + ') is larger than total number of possible test trials for this task (' + str(total_trials) + ')...')
			log.info('Changing test set size to ' + str(total_trials) + '...')
			args.test_set_size = total_trials
		else:
			log.info('Test set size = ' + str(args.test_set_size) + '...')

	# Generate complete matrix problems
	# If m = 0, training and test sets are drawn from same set of shapes
	if args.m_holdout == 0:
		# Create all possible combinations
		all_row_comb = builtins.list(combinations(train_shapes, 3))
		# Create all possible permutations for each combination
		all_comb_perm = builtins.list(permutations(range(3)))
		# Create all trials
		all_trials = []
		for comb in all_row_comb:
			for perm1 in all_comb_perm:
				for perm2 in all_comb_perm:
					all_trials.append(np.array([np.array(comb)[np.array(perm1)], np.array(comb)[np.array(perm2)]]))
		random.shuffle(all_trials)
		all_trials = np.array(all_trials)
		# Split trials for train and test sets
		trials_train = all_trials[:args.train_set_size,:,:]
		trials_test = all_trials[args.train_set_size:(args.train_set_size+args.test_set_size),:,:]
	# Otherwise, training and test sets are completely disjoint (in terms of the shapes that are used), and can be generated separately
	else:
		# Training trials
		# Create all possible combinations
		all_row_comb = builtins.list(combinations(train_shapes, 3))
		# Create all possible permutations for each combination
		all_comb_perm = builtins.list(permutations(range(3)))
		# Create all trials
		all_trials = []
		for comb in all_row_comb:
			for perm1 in all_comb_perm:
				for perm2 in all_comb_perm:
					all_trials.append(np.array([np.array(comb)[np.array(perm1)], np.array(comb)[np.array(perm2)]]))
		random.shuffle(all_trials)
		all_trials = np.array(all_trials)
		# Split trials for train and test sets
		trials_train = all_trials[:args.train_set_size,:,:]
		# Test trials
		# Create all possible combinations
		all_row_comb = builtins.list(combinations(test_shapes, 3))
		# Create all possible permutations for each combination
		all_comb_perm = builtins.list(permutations(range(3)))
		# Create all trials
		all_trials = []
		for comb in all_row_comb:
			for perm1 in all_comb_perm:
				for perm2 in all_comb_perm:
					all_trials.append(np.array([np.array(comb)[np.array(perm1)], np.array(comb)[np.array(perm2)]]))
		random.shuffle(all_trials)
		all_trials = np.array(all_trials)
		# Split trials for train and test sets
		trials_test = all_trials[:args.test_set_size,:,:]

	# Generate multiple-choice options
	# Training set
	train_answer_choices = []
	for t in range(trials_train.shape[0]):
		problem_shapes = trials_train[t,0,:]
		other_shapes = train_shapes[np.all(np.not_equal(np.expand_dims(problem_shapes,1), np.expand_dims(train_shapes,0)),0)]
		if other_shapes.shape[0] == 0:
			log.info('Training set does not have enough objects for 4 multiple-choice options, limiting to 3 options...')
			all_choices = problem_shapes
		else:
			np.random.shuffle(other_shapes)
			other_choice = other_shapes[0]
			all_choices = np.append(problem_shapes, other_choice)
		np.random.shuffle(all_choices)
		train_answer_choices.append(all_choices)
	# Test set
	test_answer_choices = []
	for t in range(trials_test.shape[0]):
		problem_shapes = trials_test[t,0,:]
		other_shapes = test_shapes[np.all(np.not_equal(np.expand_dims(problem_shapes,1), np.expand_dims(test_shapes,0)),0)]
		if other_shapes.shape[0] == 0:
			log.info('Test set does not have enough objects for 4 multiple-choice options, limiting to 3 options...')
			all_choices = problem_shapes
		else:
			np.random.shuffle(other_shapes)
			other_choice = other_shapes[0]
			all_choices = np.append(problem_shapes, other_choice)
		np.random.shuffle(all_choices)
		test_answer_choices.append(all_choices)

	# Create different versions of sequence and targets for multiple-choice and predictive versions of task
	# Training set
	train_MC_seq = []
	train_MC_targ = []
	for t in range(trials_train.shape[0]):
		pre_MC = trials_train[t,:,:].flatten()[:-1]
		MC_seq = np.concatenate([pre_MC, train_answer_choices[t]])
		img_targ_id = trials_train[t,-1,-1]
		MC_targ = np.where(train_answer_choices[t] == img_targ_id)[0][0]
		train_MC_seq.append(MC_seq)
		train_MC_targ.append(MC_targ)
	# Test set
	test_MC_seq = []
	test_MC_targ = []
	for t in range(trials_test.shape[0]):
		pre_MC = trials_test[t,:,:].flatten()[:-1]
		MC_seq = np.concatenate([pre_MC, test_answer_choices[t]])
		img_targ_id = trials_test[t,-1,-1]
		MC_targ = np.where(test_answer_choices[t] == img_targ_id)[0][0]
		test_MC_seq.append(MC_seq)
		test_MC_targ.append(MC_targ)

	# Create training and test sets
	train_set = {'seq_ind': np.array(train_MC_seq), 'y': np.array(train_MC_targ)}
	test_set = {'seq_ind': np.array(test_MC_seq), 'y': np.array(test_MC_targ)}

	return args, train_set, test_set