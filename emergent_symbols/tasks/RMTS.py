import sys
import random
import math
import numpy as np
import time
from itertools import combinations, permutations
import builtins
from copy import deepcopy

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True
# Logging utility
from util import log

"""
Combinatorics:
n is total number of objects in training or test set
for training set, n will actually be (n - m)
for test set, n will actually be m
n_same_trials = nC4 * 4C1 * 3C1 * 2C2 * 2 * 2
n_diff_trials = nC5 * 5C1 * 4C2 * 2 * 2C2 * 2 * 2
"""

# Dimensionality of multiple-choice output
y_dim = 2
# Sequence length
seq_len = 6
# Task segmentation (for context normalization)
task_seg = [[0,1], [2,3], [4,5]]

# Method for calculating number of combinations
def n_comb(n, r):
	return int(math.factorial(n) / (math.factorial(r) * math.factorial(n-r)))

# Create subsampled dataset
def subsampled_dset(shapes, n_trials):
	seq = np.array([]).astype(np.int)
	targ = np.array([]).astype(np.int)
	while seq.shape[0] < n_trials:
		# Sample same trial
		np.random.shuffle(shapes)
		comb = shapes[:4]
		same_targ = np.round(np.random.rand()).astype(np.int)
		if same_targ == 0:
			same_seq = [comb[0], comb[0], comb[1], comb[1], comb[2], comb[3]]
		elif same_targ == 1:
			same_seq = [comb[0], comb[0], comb[1], comb[2], comb[3], comb[3]]
		if seq.shape[0] == 0:
			seq = np.expand_dims(np.array(same_seq), 0)
			targ = np.append(targ, same_targ)
		else:
			if not np.any(np.all(seq == np.tile(same_seq, [seq.shape[0], 1]), 1)):
				seq = np.append(seq, np.expand_dims(np.array(same_seq), 0), 0)
				targ = np.append(targ, same_targ)
			else:
				sample_again = True
				while sample_again:
					# Sample another same trial
					np.random.shuffle(shapes)
					comb = shapes[:4]
					same_targ = np.round(np.random.rand()).astype(np.int)
					if same_targ == 0:
						same_seq = [comb[0], comb[0], comb[1], comb[1], comb[2], comb[3]]
					elif same_targ == 1:
						same_seq = [comb[0], comb[0], comb[1], comb[2], comb[3], comb[3]]
					if not np.any(np.all(seq == np.tile(same_seq, [seq.shape[0], 1]), 1)):
						sample_again = False
						seq = np.append(seq, np.expand_dims(np.array(same_seq), 0), 0)
						targ = np.append(targ, same_targ)
		# Sample different trial
		np.random.shuffle(shapes)
		comb = shapes[:5]
		diff_targ = np.round(np.random.rand()).astype(np.int)
		if diff_targ == 0:
			diff_seq = [comb[0], comb[1], comb[2], comb[3], comb[4], comb[4]]
		elif diff_targ == 1:
			diff_seq = [comb[0], comb[1], comb[2], comb[2], comb[3], comb[4]]
		if not np.any(np.all(seq == np.tile(diff_seq, [seq.shape[0], 1]), 1)):
			seq = np.append(seq, np.expand_dims(np.array(diff_seq), 0), 0)
			targ = np.append(targ, diff_targ)
		else:
			sample_again = True
			while sample_again:
				# Sample another same trial
				np.random.shuffle(shapes)
				comb = shapes[:5]
				diff_targ = np.round(np.random.rand()).astype(np.int)
				if diff_targ == 0:
					diff_seq = [comb[0], comb[1], comb[2], comb[3], comb[4], comb[4]]
				elif diff_targ == 1:
					diff_seq = [comb[0], comb[1], comb[2], comb[2], comb[3], comb[4]]
				if not np.any(np.all(seq == np.tile(diff_seq, [seq.shape[0], 1]), 1)):
					sample_again = False
					seq = np.append(seq, np.expand_dims(np.array(diff_seq), 0), 0)
					targ = np.append(targ, diff_targ)
	# Shuffle
	trial_order = np.arange(len(seq))
	np.random.shuffle(trial_order)
	seq = seq[trial_order,:]
	targ = targ[trial_order]
	# Select subset
	seq = seq[:n_trials,:]
	targ = targ[:n_trials]
	return seq, targ

# Create full dataset
def full_dset(shapes, n_trials):
	# All same trials
	all_same_seq = []
	all_same_targ = []
	all_same_trial_comb = builtins.list(combinations(shapes, 4))
	for comb in all_same_trial_comb:
		comb = builtins.list(comb)
		for s1 in comb:
			comb_minus_s1 = deepcopy(comb)
			comb_minus_s1.remove(s1)
			source_same_pair = [s1, s1]
			for s2 in comb_minus_s1:
				diff_pair = deepcopy(comb_minus_s1)
				diff_pair.remove(s2)
				diff_pair_perm = builtins.list(permutations(diff_pair, 2))
				for diff_pair in diff_pair_perm:
					same_pair = [s2, s2]
					diff_pair = builtins.list(diff_pair)
					choices = [same_pair, diff_pair]
					choices_perm = builtins.list(permutations(choices, 2))
					for targ, choices in enumerate(choices_perm):
						choices = builtins.list(choices)
						same_seq = source_same_pair + choices[0] + choices[1]
						all_same_seq.append(same_seq)
						all_same_targ.append(targ)
	# All different trials
	all_diff_seq = []	
	all_diff_targ = []
	all_diff_trial_comb = builtins.list(combinations(shapes, 5))	
	for comb in all_diff_trial_comb:
		comb = builtins.list(comb)
		all_diff1_comb = builtins.list(combinations(comb, 2))
		for diff1_comb in all_diff1_comb:
			diff1_comb = builtins.list(diff1_comb)
			comb_minus_diff1 = deepcopy(comb)
			comb_minus_diff1.remove(diff1_comb[0])
			comb_minus_diff1.remove(diff1_comb[1])
			all_diff1_perm = builtins.list(permutations(diff1_comb, 2))
			for diff1_perm in all_diff1_perm:
				source_diff_pair = builtins.list(diff1_perm)
				all_diff2_comb = builtins.list(combinations(comb_minus_diff1, 2))
				for diff2_comb in all_diff2_comb:
					diff2_comb = builtins.list(diff2_comb)
					s = deepcopy(comb_minus_diff1)
					s.remove(diff2_comb[0])
					s.remove(diff2_comb[1])
					same_pair = [s[0], s[0]]
					all_diff2_perm = builtins.list(permutations(diff2_comb, 2))
					for diff2_perm in all_diff2_perm:
						diff_pair = builtins.list(diff2_perm)
						choices = [diff_pair, same_pair]
						choices_perm = builtins.list(permutations(choices, 2))
						for targ, choices in enumerate(choices_perm):
							choices = builtins.list(choices)
							diff_seq = source_diff_pair + choices[0] + choices[1]
							all_diff_seq.append(diff_seq)
							all_diff_targ.append(targ)
	# Duplicate trials if necessary (so that trial types are balanced)
	# Same trials
	if len(all_same_seq) < n_trials/2:
		all_same_seq_augmented = deepcopy(all_same_seq)
		all_same_targ_augmented = deepcopy(all_same_targ)
		for a in range(int(n_trials/2) - len(all_same_seq)):
			trial_ind = np.floor(np.random.rand() * len(all_same_seq)).astype(np.int)
			all_same_seq_augmented.append(all_same_seq[trial_ind])
			all_same_targ_augmented.append(all_same_targ[trial_ind])
		all_same_seq = all_same_seq_augmented
		all_same_targ = all_same_targ_augmented
	# Different trials
	if len(all_diff_seq) < n_trials/2:
		all_diff_seq_augmented = deepcopy(all_diff_seq)
		all_diff_targ_augmented = deepcopy(all_diff_targ)
		for a in range(int(n_trials/2) - len(all_diff_seq)):
			trial_ind = np.floor(np.random.rand() * len(all_diff_seq)).astype(np.int)
			all_diff_seq_augmented.append(all_diff_seq[trial_ind])
			all_diff_targ_augmented.append(all_diff_targ[trial_ind])
		all_diff_seq = all_diff_seq_augmented
		all_diff_targ = all_diff_targ_augmented
	# Combine same and different trials
	seq = all_same_seq + all_diff_seq
	targ = all_same_targ + all_diff_targ
	# Shuffle
	trial_order = np.arange(len(seq))
	np.random.shuffle(trial_order)
	seq = np.array(seq)[trial_order,:]
	targ = np.array(targ)[trial_order]
	# Select subset
	seq = seq[:n_trials,:]
	targ = targ[:n_trials]
	return seq, targ

# Task generator
def create_task(args, train_shapes, test_shapes):

	log.info('n_shapes = ' + str(args.n_shapes) + '...')
	log.info('m_holdout = ' + str(args.m_holdout) + '...')
	# If m = 0, training and test sets are drawn from same set of shapes
	if args.m_holdout == 0:
		# Total number of possible trials
		shapes_avail = args.n_shapes
		n_same_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * n_comb(2,2) * 2 * 2
		n_diff_trials = n_comb(shapes_avail,5) * n_comb(5,2) * 2 * n_comb(3,2) * 2 * n_comb(1,1) * 2
		total_unique_trials = n_same_trials + n_diff_trials
		log.info('Total possible trials = ' + str(total_unique_trials) + '...')
		if n_diff_trials > n_same_trials:
			total_trials = n_diff_trials * 2
		else:
			total_trials = n_same_trials * 2
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
	# Otherwise, training and test sets are completely disjoint (in terms of the shapes that are used)
	else:
		# Ensure that there are enough potential trials for desired training set size (or change training set size)
		shapes_avail = args.n_shapes - args.m_holdout
		n_same_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * n_comb(2,2) * 2 * 2
		n_diff_trials = n_comb(shapes_avail,5) * n_comb(5,2) * 2 * n_comb(3,2) * 2 * n_comb(1,1) * 2
		total_unique_trials = n_same_trials + n_diff_trials
		log.info('Total possible training trials = ' + str(total_unique_trials) + '...')
		if n_diff_trials > n_same_trials:
			total_trials = n_diff_trials * 2
		else:
			total_trials = n_same_trials * 2
		if args.train_set_size > total_trials:
			log.info('Desired training set size (' + str(args.train_set_size) + ') is larger than total number of possible training trials for this task (' + str(total_trials) + ')...')
			log.info('Changing training set size to ' + str(total_trials) + '...')
			args.train_set_size = total_trials
		else:
			log.info('Training set size = ' + str(args.train_set_size) + '...')
		# Ensure that there are enough potential trials for desired test set size (or change test set size)
		shapes_avail = args.m_holdout
		n_same_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * n_comb(2,2) * 2 * 2
		n_diff_trials = n_comb(shapes_avail,5) * n_comb(5,2) * 2 * n_comb(3,2) * 2 * n_comb(1,1) * 2
		total_unique_trials = n_same_trials + n_diff_trials
		log.info('Total possible test trials = ' + str(total_unique_trials) + '...')
		if n_diff_trials > n_same_trials:
			total_trials = n_diff_trials * 2
		else:
			total_trials = n_same_trials * 2
		if args.test_set_size > total_trials:
			log.info('Desired test set size (' + str(args.test_set_size) + ') is larger than total number of possible test trials for this task (' + str(total_trials) + ')...')
			log.info('Changing test set size to ' + str(total_trials) + '...')
			args.test_set_size = total_trials
		else:
			log.info('Test set size = ' + str(args.test_set_size) + '...')

	# Create all possible trials
	if args.m_holdout == 0:
		if args.train_gen_method == 'subsample':
			all_seq, all_targ = subsampled_dset(train_shapes, args.train_set_size + args.test_set_size)
		elif args.train_gen_method == 'full_space':
			all_seq, all_targ = full_dset(train_shapes, args.train_set_size + args.test_set_size)
		# Split train and test sets
		train_seq = all_seq[:args.train_set_size,:]
		train_targ = all_targ[:args.train_set_size]
		test_seq = all_seq[args.train_set_size:,:]
		test_targ = all_targ[args.train_set_size:]
	# Otherwise, training and test sets are completely disjoint (in terms of the shapes that are used), and can be generated separately
	else:
		if args.train_gen_method == 'subsample':
			train_seq, train_targ = subsampled_dset(train_shapes, args.train_set_size)
		elif args.train_gen_method == 'full_space':
			train_seq, train_targ = full_dset(train_shapes, args.train_set_size)
		if args.test_gen_method == 'subsample':
			test_seq, test_targ = subsampled_dset(test_shapes, args.test_set_size)
		elif args.test_gen_method == 'full_space':
			test_seq, test_targ = full_dset(test_shapes, args.test_set_size)

	# Create training and test sets
	train_set = {'seq_ind': train_seq, 'y': train_targ}
	test_set = {'seq_ind': test_seq, 'y': test_targ}

	return args, train_set, test_set