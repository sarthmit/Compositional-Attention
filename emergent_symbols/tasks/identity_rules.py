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
n_AAA_trials = nC4 * 4C1 * 3C1 * 4P4
n_ABA_trials = nC4 * 4C1 * 3C1 * 2C1 * 1C1 * 4P$
n_ABB_trials = nC4 * 4C1 * 3C1 * 2C1 * 1C1 * 4P$
n_trials_total = n_AAA_trials + n_ABA_trials + n_ABB_trials
"""

# Dimensionality of multiple-choice output
y_dim = 4
# Sequence length
seq_len = 9

# Method for calculating number of combinations
def n_comb(n, r):
	return int(math.factorial(n) / (math.factorial(r) * math.factorial(n-r)))

# Create subsampled dataset
def subsampled_dset(shapes, n_trials):
	seq = np.array([]).astype(np.int)
	targ = np.array([]).astype(np.int)
	while seq.shape[0] < n_trials:
		# Sample AAA trial
		np.random.shuffle(shapes)
		comb = shapes[:4]
		A1 = comb[0]
		A2 = comb[1]
		np.random.shuffle(comb)
		AAA_targ = np.where((comb == A2).astype(np.int))[0][0]
		AAA_seq = [A1, A1, A1, A2, A2] + builtins.list(comb)
		if seq.shape[0] == 0:
			seq = np.expand_dims(np.array(AAA_seq), 0)
			targ = np.append(targ, AAA_targ)
		else:
			if not np.any(np.all(seq == np.tile(AAA_seq, [seq.shape[0], 1]), 1)):
				seq = np.append(seq, np.expand_dims(np.array(AAA_seq), 0), 0)
				targ = np.append(targ, AAA_targ)
			else:
				sample_again = True
				while sample_again:
					# Sample another AAA trial
					np.random.shuffle(shapes)
					comb = shapes[:4]
					A1 = comb[0]
					A2 = comb[1]
					np.random.shuffle(comb)
					AAA_targ = np.where((comb == A2).astype(np.int))[0][0]
					AAA_seq = [A1, A1, A1, A2, A2] + builtins.list(comb)
					if not np.any(np.all(seq == np.tile(AAA_seq, [seq.shape[0], 1]), 1)):
						sample_again = False
						seq = np.append(seq, np.expand_dims(np.array(AAA_seq), 0), 0)
						targ = np.append(targ, AAA_targ)
		# Sample ABA trial
		np.random.shuffle(shapes)
		comb = shapes[:4]
		A1 = comb[0]
		B1 = comb[1]
		A2 = comb[2]
		B2 = comb[3]
		np.random.shuffle(comb)
		ABA_targ = np.where((comb == A2).astype(np.int))[0][0]
		ABA_seq = [A1, B1, A1, A2, B2] + builtins.list(comb)
		if not np.any(np.all(seq == np.tile(ABA_seq, [seq.shape[0], 1]), 1)):
			seq = np.append(seq, np.expand_dims(np.array(ABA_seq), 0), 0)
			targ = np.append(targ, ABA_targ)
		else:
			sample_again = True
			while sample_again:
				np.random.shuffle(shapes)
				comb = shapes[:4]
				A1 = comb[0]
				B1 = comb[1]
				A2 = comb[2]
				B2 = comb[3]
				np.random.shuffle(comb)
				ABA_targ = np.where((comb == A2).astype(np.int))[0][0]
				ABA_seq = [A1, B1, A1, A2, B2] + builtins.list(comb)
				if not np.any(np.all(seq == np.tile(ABA_seq, [seq.shape[0], 1]), 1)):
					sample_again = False
					seq = np.append(seq, np.expand_dims(np.array(ABA_seq), 0), 0)
					targ = np.append(targ, ABA_targ)
		# Sample ABB trial
		np.random.shuffle(shapes)
		comb = shapes[:4]
		A1 = comb[0]
		B1 = comb[1]
		A2 = comb[2]
		B2 = comb[3]
		np.random.shuffle(comb)
		ABB_targ = np.where((comb == B2).astype(np.int))[0][0]
		ABB_seq = [A1, B1, B1, A2, B2] + builtins.list(comb)
		if not np.any(np.all(seq == np.tile(ABB_seq, [seq.shape[0], 1]), 1)):
			seq = np.append(seq, np.expand_dims(np.array(ABB_seq), 0), 0)
			targ = np.append(targ, ABB_targ)
		else:
			sample_again = True
			while sample_again:
				np.random.shuffle(shapes)
				comb = shapes[:4]
				A1 = comb[0]
				B1 = comb[1]
				A2 = comb[2]
				B2 = comb[3]
				np.random.shuffle(comb)
				ABB_targ = np.where((comb == B2).astype(np.int))[0][0]
				ABB_seq = [A1, B1, B1, A2, B2] + builtins.list(comb)
				if not np.any(np.all(seq == np.tile(ABB_seq, [seq.shape[0], 1]), 1)):
					sample_again = False
					seq = np.append(seq, np.expand_dims(np.array(ABB_seq), 0), 0)
					targ = np.append(targ, ABB_targ)
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
	all_4shape_comb = builtins.list(combinations(shapes, 4))
	# All AAA trials
	all_AAA_seq = []
	all_AAA_targ = []
	for comb in all_4shape_comb:
		comb = builtins.list(comb)
		all_comb_perm = builtins.list(permutations(comb, 4))
		for A1 in comb:
			comb_minus_A1 = deepcopy(comb)
			comb_minus_A1.remove(A1)
			for A2 in comb_minus_A1:
				for choice_perm in all_comb_perm:
					choice_perm = builtins.list(choice_perm)
					AAA_targ = np.where((np.array(choice_perm) == A2).astype(np.int))[0][0]
					AAA_seq = [A1, A1, A1] + [A2, A2] + choice_perm
					all_AAA_seq.append(AAA_seq)
					all_AAA_targ.append(AAA_targ)
	# All ABA trials
	all_ABA_seq = []
	all_ABA_targ = []
	for comb in all_4shape_comb:
		comb = builtins.list(comb)
		all_comb_perm = builtins.list(permutations(comb, 4))
		for A1 in comb:
			comb_minus_A1 = deepcopy(comb)
			comb_minus_A1.remove(A1)
			for B1 in comb_minus_A1:
				comb_minus_B1 = deepcopy(comb_minus_A1)
				comb_minus_B1.remove(B1)
				for A2 in comb_minus_B1:
					comb_minus_A2 = deepcopy(comb_minus_B1)
					comb_minus_A2.remove(A2)
					B2 = comb_minus_A2[0]
					for choice_perm in all_comb_perm:
						choice_perm = builtins.list(choice_perm)
						ABA_targ = np.where((np.array(choice_perm) == A2).astype(np.int))[0][0]
						ABA_seq = [A1, B1, A1] + [A2, B2] + choice_perm
						all_ABA_seq.append(ABA_seq)
						all_ABA_targ.append(ABA_targ)
	# All ABB trials
	all_ABB_seq = []
	all_ABB_targ = []
	for comb in all_4shape_comb:
		comb = builtins.list(comb)
		all_comb_perm = builtins.list(permutations(comb, 4))
		for A1 in comb:
			comb_minus_A1 = deepcopy(comb)
			comb_minus_A1.remove(A1)
			for B1 in comb_minus_A1:
				comb_minus_B1 = deepcopy(comb_minus_A1)
				comb_minus_B1.remove(B1)
				for A2 in comb_minus_B1:
					comb_minus_A2 = deepcopy(comb_minus_B1)
					comb_minus_A2.remove(A2)
					B2 = comb_minus_A2[0]
					for choice_perm in all_comb_perm:
						choice_perm = builtins.list(choice_perm)
						ABB_targ = np.where((np.array(choice_perm) == B2).astype(np.int))[0][0]
						ABB_seq = [A1, B1, B1] + [A2, B2] + choice_perm
						all_ABB_seq.append(ABB_seq)
						all_ABB_targ.append(ABB_targ)
	# Duplicate AAA trials (so that trial types are balanced)
	max_trials_per_trial_type = np.max([len(all_AAA_seq), len(all_ABA_seq), len(all_ABB_seq)])
	all_AAA_seq_augmented = deepcopy(all_AAA_seq)
	all_AAA_targ_augmented = deepcopy(all_AAA_targ)
	for a in range(max_trials_per_trial_type - len(all_AAA_seq)):
		trial_ind = np.floor(np.random.rand() * len(all_AAA_seq)).astype(np.int)
		all_AAA_seq_augmented.append(all_AAA_seq[trial_ind])
		all_AAA_targ_augmented.append(all_AAA_targ[trial_ind])
	all_AAA_seq = all_AAA_seq_augmented
	all_AAA_targ = all_AAA_targ_augmented
	# Combine same and different trials
	seq = all_AAA_seq + all_ABA_seq + all_ABB_seq
	targ = all_AAA_targ + all_ABA_targ + all_ABB_targ
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
		n_AAA_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * math.factorial(4)
		n_ABA_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * n_comb(2,1) * n_comb(1,1) * math.factorial(4)
		n_ABB_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * n_comb(2,1) * n_comb(1,1) * math.factorial(4)
		total_unique_trials = n_AAA_trials + n_ABA_trials + n_ABB_trials
		total_trials = np.max([n_AAA_trials, n_ABA_trials, n_ABB_trials]) * 3
		log.info('Total possible trials = ' + str(total_unique_trials) + '...')
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
		n_AAA_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * math.factorial(4)
		n_ABA_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * n_comb(2,1) * n_comb(1,1) * math.factorial(4)
		n_ABB_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * n_comb(2,1) * n_comb(1,1) * math.factorial(4)
		total_unique_trials = n_AAA_trials + n_ABA_trials + n_ABB_trials
		total_trials = np.max([n_AAA_trials, n_ABA_trials, n_ABB_trials]) * 3
		log.info('Total possible training trials = ' + str(total_unique_trials) + '...')
		if args.train_set_size > total_trials:
			log.info('Desired training set size (' + str(args.train_set_size) + ') is larger than total number of possible training trials for this task (' + str(total_trials) + ')...')
			log.info('Changing training set size to ' + str(total_trials) + '...')
			args.train_set_size = total_trials
		else:
			log.info('Training set size = ' + str(args.train_set_size) + '...')
		# Ensure that there are enough potential trials for desired test set size (or change test set size)
		shapes_avail = args.m_holdout
		n_AAA_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * math.factorial(4)
		n_ABA_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * n_comb(2,1) * n_comb(1,1) * math.factorial(4)
		n_ABB_trials = n_comb(shapes_avail,4) * n_comb(4,1) * n_comb(3,1) * n_comb(2,1) * n_comb(1,1) * math.factorial(4)
		total_unique_trials = n_AAA_trials + n_ABA_trials + n_ABB_trials
		total_trials = np.max([n_AAA_trials, n_ABA_trials, n_ABB_trials]) * 3
		log.info('Total possible test trials = ' + str(total_unique_trials) + '...')
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