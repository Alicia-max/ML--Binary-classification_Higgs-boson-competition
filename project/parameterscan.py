# -*- coding: utf-8 -*-
import numpy as np
#plot library
import seaborn as sns
import matplotlib.pyplot as plt

# I wanted to avoid this but its actually much simpler to use logging...
import os
import logging

## Our libraries
from helpers import *
from cross_val import *
from preprocessing import *
from plot import *
from run import *

# -*------------------------- Constants ---------------------------------*-
DEBUG = False
PATH_TRAIN= '../data/train.csv'
PATH_TEST = '../data/test.csv'
SEED = 1
K_FOLD = 5
MAX_ITERS = 100

# These have to be defined later on in the main
LOG_DIR = None
# Can be 0, 1, 2, or 3 (with group_ before)
GROUP = None
# Can be 'under', 'over' or None
SAMPLING = None
# Can be True or False
OFFSET = None

# -*------------------------- Helper ---------------------------------*-
def print_best_perfs(idxs, params, acc, std):
    for i in range(5):
        idx = idxs[i]
        logging.info('Ranked {}'.format(i+1))
        logging.info('Parameters\n %s', params[idx])
        logging.info('With validation accuracy {} std {}'.format(acc[idx], std[idx]))
        logging.info('\n')


# -*------------------------- Parameter Scan function ---------------------------------*-
def ParameterScan():
	logging.info('Starting parameter scan for: Group {}, sampling strategy {} and offset {}'.format(GROUP, SAMPLING, OFFSET))
	
	## Scanning the following parameters
	if DEBUG:
		degree = np.arange(1, 3)
		gamma = np.logspace(-8, -1, 3)
	else:
		degree = np.arange(1, 10)
		gamma = np.logspace(-8, -1, 8)
	lambdas = gamma
	
	####################################
	logging.info('--------------------------')
	logging.info('\nMethod: Least squares GD')
	method = least_squares_GD
	parameters_GD = []

	for d in degree:
		for g in gamma:
			parameters_GD.append({'gamma':g, 'degree':d, 'max_iters':MAX_ITERS, 'offset': OFFSET})

	acc_tr, acc_val, std_tr, std_val, idx_sorted = cross_tunning(y, x,
											 k_fold=K_FOLD, method=method , parameters=parameters_GD, seed=SEED, log=False)

	print_best_perfs(idx_sorted, parameters_GD, acc_val, std_val)
	best_GD = acc_val[idx_sorted[0]]
	
	####################################
	logging.info('--------------------------')
	logging.info('\nMethod: Least squares SGD')
	method = least_squares_SGD
	parameters_SGD = []

	for d in degree:
		for g in gamma:
			parameters_SGD.append({'gamma':g, 'degree':d, 'max_iters':MAX_ITERS, 'offset': OFFSET})

	acc_tr, acc_val, std_tr, std_val, idx_sorted = cross_tunning(y, x,
											 k_fold=K_FOLD, method=method , parameters=parameters_SGD, seed=SEED, log=False)

	print_best_perfs(idx_sorted, parameters_SGD, acc_val, std_val)
	best_SGD = acc_val[idx_sorted[0]]
	
	####################################
	logging.info('--------------------------')
	logging.info('\nMethod: Least squares')
	method = least_squares
	parameters_LS = []

	for d in degree:
			parameters_LS.append({'degree':d, 'offset': OFFSET})
	try:    
		acc_tr, acc_val, std_tr, std_val, idx_sorted = cross_tunning(y, x,
												 k_fold=K_FOLD, method=method , parameters=parameters_LS, seed=SEED, log=False)
		print_best_perfs(idx_sorted, parameters_LS, acc_val, std_val)
		best_LS = acc_val[idx_sorted[0]]
	except:
		logging.info('Least squares did not work')
		best_LS = .5
		
	####################################
	logging.info('--------------------------')
	logging.info('\nMethod: Ridge Regression')
	method = ridge_regression
	parameters_RR = []

	for d in degree:
		for l in lambdas:
			parameters_RR.append({'lambda_':l, 'degree':d,'offset': OFFSET})

	acc_tr, acc_val, std_tr, std_val, idx_sorted = cross_tunning(y, x,
											 k_fold=K_FOLD, method=method , parameters=parameters_RR, seed=SEED, log=False)

	print_best_perfs(idx_sorted, parameters_RR, acc_val, std_val)
	best_RR = acc_val[idx_sorted[0]]
	
	####################################
	logging.info('--------------------------')
	logging.info('\nMethod: Logistic Regression')
	method = logistic_regression
	parameters_LR = []

	for d in degree:
		for g in gamma:
			parameters_LR.append({'gamma':g, 'degree':d, 'max_iters':MAX_ITERS, 'offset': OFFSET, 'initial_w':None})

	acc_tr, acc_val, std_tr, std_val, idx_sorted = cross_tunning(y, x,
											 k_fold=K_FOLD, method=method , parameters=parameters_LR, seed=SEED, log=True)

	print_best_perfs(idx_sorted, parameters_LR, acc_val, std_val)
	best_LR = acc_val[idx_sorted[0]]
	
	#################################### 
	logging.info('--------------------------')
	logging.info('\nMethod: Regularized Logistic Regression')
	logging.info('\nFixing the learning rate gamma to the values of the best from logistic - otherwise too long')
	method = reg_logistic_regression
	lambdas = gamma
	parameters_RLR = []

	gamma = parameters_LR[idx_sorted[0]]['gamma']
	for d in degree:
		for l in lambdas:
			parameters_RLR.append({'gamma':gamma, 'lambda_':l, 'degree':d, 'max_iters':MAX_ITERS, 'offset': OFFSET, 'initial_w':None})

	acc_tr, acc_val, std_tr, std_val, idx_sorted = cross_tunning(y, x,
											 k_fold=K_FOLD, method=method , parameters=parameters_RLR, seed=SEED, log=True)

	print_best_perfs(idx_sorted, parameters_RLR, acc_val, std_val)
	best_RLR = acc_val[idx_sorted[0]]
	
	####################################
	logging.info('--------------------------')
	Methods = ['Least square GD', 'Least square SDG', 'Least square', 'Ridge Regression', 'Logistic', 'Regularized Logistic']
	Accuracy = [best_GD,best_SGD, best_LS, best_RR, best_LR, best_RLR]
	idx = np.argmax(np.array(Accuracy))
	logging.info('Best method: {}\n'.format(Methods[idx]))
	logging.info('--------------------------')

	ax = sns.relplot(x=Methods, y=Accuracy)
	ax.set_xticklabels(rotation = 80)
	plt.title("Test Accuracy Model Comparison \n Group {}, sampling {}, offset {}".format(GROUP, SAMPLING, OFFSET))
	plt.xlabel('Test Accuracy')
	plt.ylabel('Method')
	plt.tight_layout()
	plt.savefig(os.path.join(LOG_DIR, '{}_sampling_{}_offset_{}.png'.format(GROUP, SAMPLING, OFFSET)))
	

if __name__ == '__main__':
	# Configure logging
	GROUP = 'group_3'
	LOG_DIR = './parameterscan_' + GROUP
	os.makedirs(LOG_DIR, exist_ok=True)

	log_filename = os.path.join(LOG_DIR, 'logs.log')
	logging.basicConfig(format='%(message)s',
						filename=log_filename,
						filemode='w', # Allows to overwrite and not append to an old log
						level=logging.INFO)
	if DEBUG:
		logging.info('DEBUG SESSION')

	logging.info('Setting up Parameter Scan')
	
	logging.info('Loading data')
	y_train, tX_train, ids_train = load_csv_data(PATH_TRAIN, sub_sample=DEBUG)
	y_test, tX_test, ids_test = load_csv_data(PATH_TEST, sub_sample=DEBUG)
	logging.info('Data succesfully loaded')
	
	###########################################
	OFFSET = True
	###########################################
	# Can be 'under', 'over' or None
	SAMPLING = None

	preprocessed_X, _, preprocessed_y, _ = preprocess_data_new(tX_train, tX_test, y_train, sampling_strategy=SAMPLING)
	x, y = preprocessed_X[GROUP], preprocessed_y[GROUP]
	ParameterScan()

	###########################################
	# Can be 'under', 'over' or None
	SAMPLING = 'under'

	preprocessed_X, _, preprocessed_y, _ = preprocess_data_new(tX_train, tX_test, y_train, sampling_strategy=SAMPLING)
	x, y = preprocessed_X[GROUP], preprocessed_y[GROUP]
	ParameterScan()

	###########################################
	# Can be 'under', 'over' or None
	SAMPLING = 'over'

	preprocessed_X, _, preprocessed_y, _ = preprocess_data_new(tX_train, tX_test, y_train, sampling_strategy=SAMPLING)
	x, y = preprocessed_X[GROUP], preprocessed_y[GROUP]
	ParameterScan()

	###########################################
	OFFSET = False
	###########################################
	# Can be 'under', 'over' or None
	SAMPLING = None

	preprocessed_X, _, preprocessed_y, _ = preprocess_data_new(tX_train, tX_test, y_train, sampling_strategy=SAMPLING)
	x, y = preprocessed_X[GROUP], preprocessed_y[GROUP]
	ParameterScan()

	###########################################
	# Can be 'under', 'over' or None
	SAMPLING = 'under'

	preprocessed_X, _, preprocessed_y, _ = preprocess_data_new(tX_train, tX_test, y_train, sampling_strategy=SAMPLING)
	x, y = preprocessed_X[GROUP], preprocessed_y[GROUP]
	ParameterScan()

	###########################################
	# Can be 'under', 'over' or None
	SAMPLING = 'over'

	preprocessed_X, _, preprocessed_y, _ = preprocess_data_new(tX_train, tX_test, y_train, sampling_strategy=SAMPLING)
	x, y = preprocessed_X[GROUP], preprocessed_y[GROUP]
	ParameterScan()
	
	logging.info('Done!')

