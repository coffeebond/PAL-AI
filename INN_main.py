import sys, itertools, time, concurrent.futures, random, subprocess, math, argparse, re, concurrent.futures, random, os, shlex, skopt, matplotlib, h5py, scipy
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import inn_models

from time import time
from time import sleep
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats
from collections import OrderedDict

from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

from matplotlib import pyplot
matplotlib.use('Agg')

params_global = OrderedDict([
	('fa_string_filter', 'uncertain'), # if fasta entry contains this string, it will be filtered out
	('flag_input_header', True), # whether the input file with target values has a header
	('flag_initial_tl', False), # whether to use initial tail length as an input for learning
	('input_tag_min', 50), # minimal number of tags for each gene to be included from the input file
	('seq_3end_append', ''), # sequence appended at the 3'-end
	('len_max', 264), #1056 # maximal length of sequence (from 3' ends, after appending constant region, if applicable)
	('len_min', 10), # minimal length of 3' UTR sequence to be included, from 3' ends (those smaller than this won't be analyzed)
	('y_mode', 'none'), # how to transform target (tail length) value, can also be changed by input option '-t'
	('tl_cons', 35), # tail length constant to be subtracted if 'y_mode' is set to 'diff'
	('test_size', 0.1), # fraction of the dataset used as test set; also used for defining CV fold 
	('n_channel', 4), # number of channels/tracks (columns) as the input layer into the INN, can be either 4 (without CDS file) or 5 (with CDS file)
	('n_dimension', 2), # dimension of the input for the INN (2 means simple one-hot encoding, 3 means outer product of the one-hot encoding)
	('n_rep', 5), # number of repeated training with 1 cross-validation group
	('flag_print_model', True), # whether to print out the model summary
	('n_threads', 20),
	('chunk', 10000)
])

inn_params = OrderedDict([
	('input_shape', (1056,5)), # this will be automatically updated based on the input 
	('flag_initial_tl', False),
	#('flag_mfe', False),
	('n_Conv1D_filter', 64),
	('n_Conv1D_kernal', 5),
	('kernel_regularizer', None), 
	('activation_func', 'selu'), 
	('dropout_rate', 0.2), 
	('n_dense_neuron', 64),
	('rnn_func', 'GRU'),
	('n_rnn_units', 32),
	('l1_reg', 0.001),
	('l2_reg', 0.001),
	('optimizer', 'adam'),
	('learning_rate', 0.001),
	('loss', 'mse'),#custom_loss
	('metrics', 'mae'),
	('batch_size', 50),
	('epochs', 50),
	# resnet-specific parameters
	('resnet_trim', 'left'), # whether to trim input to the ResNet for addition
	('resnet_version', 1), 
	('resnet_n_group', 6), # number of ResNet groups
	('resnet_n_block', 4), # number of ResNet blocks per group
	('dilation_rate_lst', [1,2,4,4,2,1]),  
	# inn-specific parameters
	('n_Conv1D_MaxPool_block', 1),
	('n_pool_size', 2),
	# cinn-specific parameters
	('n_Conv2D_filter', 32),
	('n_Conv2D_kernel', 5),
	('n_Conv2D_MaxPool_block', 1)
])

inn_params_lst = [
	Categorical(categories=[32, 64, 96], name = 'n_Conv1D_filter'),
	Categorical(categories=[5, 6], name = 'n_Conv1D_kernal'),
	Categorical(categories=[1, 2], name = 'n_Conv1D_MaxPool_block'),
	Categorical(categories=[64], name = 'n_Conv2D_filter'),
	Categorical(categories=[5, 6, 7], name = 'n_Conv2D_kernal'),
	Categorical(categories=[1], name = 'n_Conv2D_MaxPool_block'),
	Categorical(categories=[32, 64, 96], name = 'n_dense_neuron'),
	Categorical(categories=[2], name = 'n_pool_size'),
	Categorical(categories=['GRU'], name = 'rnn_func'),
	Categorical(categories=[32, 64, 96], name = 'n_rnn_units'),
	Categorical(categories=['selu'], name='activation_func'), #'activation_func'
	Real(low=1e-6, high=1e-2, prior='log-uniform', name='l1_reg'), #'l1_reg'
	Real(low=1e-6, high=1e-2, prior='log-uniform', name='l2_reg'), #'l2_reg'
	Categorical(categories=[0.1, 0.2, 0.3], name = 'dropout_rate'),
	Categorical(categories=['adam'], name = 'optimizer'),
	Real(low=1e-5, high=1e-2, prior='log-uniform', name='learning_rate'), #'learning_rate' 
	#Categorical(categories=['mse'], name='loss'), #'loss'
	#Categorical(categories=['mae'], name='metrics'), #'metrics'
	Categorical(categories=[100], name = 'batch_size'),
	Categorical(categories=[10], name = 'epochs')
]



def timer(): # calculate runtime
	temp = str(time()-t_start).split('.')[0]
	temp =  '\t' + temp + 's passed...' + '\t' + str(datetime.now())
	return temp

def pwrite(f, text):
	f.write(text + '\n')
	print(text)

def make_log_file(filename, p_params = False, p_vars = False):
	f_log = open(filename, 'w')
	if isinstance(p_params, dict):
		pwrite(f_log, 'Updated global parameters:')
		for param in p_params:
			pwrite(f_log, param + ': ' + str(p_params[param]))
	if isinstance(p_vars, dict):
		pwrite(f_log, '\nInput arguments:')
		for var in p_vars:
			pwrite(f_log, var + ': ' + str(p_vars[var]))
		pwrite(f_log, '\n')	
	return(f_log)

def fasta_id_parser(line, sep = '\t'):
	lst = line.rstrip().lstrip('>').split(sep)
	return(lst)

def fasta_to_dict(file, len_min = 0, string_filter = None):
	temp_dict = {}
	temp_seq = ''
	current_id = None
	with open(file, 'r') as f:
		flag_seq_append = False
		while(True):
			line = f.readline()
			if line:
				if line[0] == '>':
					if current_id is not None:
						temp_dict[current_id] = temp_seq
					if (string_filter is None) or (string_filter not in line):
						current_id = fasta_id_parser(line, sep = '__', pos = 0)
						temp_seq = ''
						if current_id in temp_dict:
							sys.exit('Error! Input file (' + file + ') has non-unique IDs!')
					else:
						current_id = None
				else:
					if current_id is not None:
						temp_seq = temp_seq + line.strip()
			else:
				# last entry
				if current_id is not None:
					temp_dict[current_id] = temp_seq 
				break
	return(temp_dict)

def l2n(seq_len, n_kernal, n_pool_size, size_limit, n=0):
	# this helper function determines the number of repeated convolution-maxpooling layers from a given sequence length and output size limit
	if math.floor((seq_len - n_kernal + 1)/n_pool_size) <= size_limit:
		return(n)
	if n == 0:
		seq_len = seq_len - n_kernal + 1
	return(l2n(math.floor((seq_len - n_kernal + 1)/n_pool_size), n_kernal, n_pool_size, size_limit, n + 1))

def outer_concatenate(a1, a2, half_mask = False):
	# perform an "outer concatenation" of two 2D arrays to get a 3D array
	assert len(a1.shape) == 2 and len(a2.shape) == 2, 'Input arrays for "outer_concatenate" must be 2D!'
	out_array = np.zeros((a1.shape[0], a2.shape[0], a1.shape[1] * a2.shape[1]))
	for i in range(a1.shape[0]):
		for j in range(a2.shape[0]):
			if (not half_mask) or i <= j: 
				out_array[i, j, :] = np.outer(a1[i,:], a2[j,:]).flatten()
	return(out_array)

def encode_seq(seq, half_mask = False):
	# convert a DNA sequence to one-hot encoding matrix
	mat = np.zeros((len(seq), 4))
	for i in range(len(seq)):
		if seq[i] == 'A':
			mat[i,0] = 1.0
		elif seq[i] == 'C':
			mat[i,1] = 1.0
		elif seq[i] == 'G':
			mat[i,2] = 1.0
		elif seq[i] == 'T' or seq[i] == 'U':
			mat[i,3] = 1.0
	if params_global['n_dimension'] == 3:
		mat = outer_concatenate(mat, mat, half_mask = half_mask)
	return(mat) 

def process_seqs(line_lst):
	# convert sequences to features as X values and tail length to Y values
	X_mat_lst = []
	id_lst = []
	y_lst = []
	label_lst = []
	itl_lst = []
	n_drop_fa_string_filter = 0 # number of entries dropped because the name of each sequence entry in the input file contains a string ("uncertain" annotations)
	n_drop_short_utr = 0 # number of entries dropped because of short UTR sequence
	n_drop_unknown_nt = 0 # numbner of entries dropped becasue of unknown characters in the sequence
	for lines in line_lst:
		sele_id = fasta_id_parser(lines[0], sep = '__')[0]
		if len(fasta_id_parser(lines[0], sep = '__')) > 1:
			gene_id = fasta_id_parser(lines[0], sep = '__')[1] 
		else:
			gene_id = fasta_id_parser(lines[0], sep = '__')[0]

		# only convert sequences that have y values, or in "prediction mode", y values are not necessary
		if sele_id in y_dict or len(y_dict) == 0: 
			if len(y_dict) == 0:
				tl = 'NA'
			else:
				tl = y_dict[sele_id]

			if (params_global['fa_string_filter'] is None) or (params_global['fa_string_filter'] not in lines[0]): # exclude "uncertain" annotations
				seq = lines[1].rstrip()
				
				# concatenate the sequence, default is ''
				seq = seq + params_global['seq_3end_append']

				#seq = seq[:-132] # use this line to truncate 3'-UTR sequence from 3'-end
				if len(seq) >= params_global['len_min']: # UTR must be longer than this
					if len(seq) >= params_global['len_max']: 
						seq = seq[-params_global['len_max']:] # truncate long UTR sequences
						if args.c and params_global['n_dimension'] == 2:
							c5_ary = np.asarray([1 for i in range(len(seq))]) # 5th channel for labeling UTR (1) or non-UTR (0)
					else:
						if args.c and params_global['n_dimension'] == 2:
							c5_ary = np.asarray([0 for i in range(params_global['len_max']-len(seq))] + [1 for i in range(len(seq))]) # 5th channel for labeling UTR (1) or non-UTR (0)
							if gene_id in cds_dict:
								cds_seq = cds_dict[gene_id]
							else:
								cds_seq = 'N' * params_global['len_max']
							if len(cds_seq) >= (params_global['len_max']-len(seq)):
								seq = cds_seq[-(params_global['len_max']-len(seq)):] + seq
							else:
								seq = 'N'*(params_global['len_max']-len(cds_seq) - len(seq)) + cds_seq + seq
						else:
							seq = 'N'*(params_global['len_max']-len(seq)) + seq # pad 'N' to short UTR sequences
					
					# encode the sequence
					seq = seq.upper().replace('U', 'T')
					if re.search('(^[ACGTNU]+$)',seq):
						seq_mat = encode_seq(seq, half_mask = True)

						if params_global['n_dimension'] == 3:
							if args.f:
								if sele_id in fold_h5.keys():
									fold_ary = fold_h5[sele_id][:]
									fold_seq = fold_h5[sele_id].attrs['seq']
									if len(fold_seq) >= len(seq):
										fold_ary = fold_ary[(len(fold_seq) - len(seq)):, (len(fold_seq) - len(seq)):]
									else:
										fold_ary = np.pad(fold_ary, ((len(seq) - len(fold_seq),0),(len(seq) - len(fold_seq),0)), mode='constant', constant_values=0)
									
									X_mat_lst.append(np.concatenate((seq_mat, np.expand_dims(fold_ary, axis = 2)), axis = 2))
							else:
								X_mat_lst.append(seq_mat)

						else:
							if args.f:
								if sele_id in fold_dict:
									f_lst = list(map(float, fold_dict[sele_id][:]))
									if len(f_lst) < params_global['len_max']:
										f_lst = [0 for i in range(params_global['len_max'] - len(f_lst))] + f_lst
								else:
									sys.exit('Error! This pA site does not have RNA folding data: ' + sele_id)
								if args.c:
									X_mat_lst.append(np.hstack((seq_mat, c5_ary.reshape(-1,1), np.asarray(f_lst[-params_global['len_max']:]).reshape(-1,1))))
								else:
									X_mat_lst.append(np.hstack((seq_mat, np.asarray(f_lst[-params_global['len_max']:]).reshape(-1,1))))
							else:
								if args.c:
									X_mat_lst.append(np.hstack((seq_mat, c5_ary.reshape(-1,1))))
								else:
									X_mat_lst.append(seq_mat)

						id_lst.append(sele_id)
						y_lst.append(tl)
						if seq.rfind('TTTTA') >= 0 and (args.c == None or c5_ary[seq.rfind('TTTTA')] == 1): # add 'label' for whether the sequence contains CPE in the 3'-UTR
							label_lst.append(1)
						else:
							label_lst.append(0)

						if params_global['flag_initial_tl']:
							if sele_id in itl_dict:
								itl_lst.append(float(itl_dict[sele_id]))
							else:
								sys.exit('No initial tail length for this entry when the initial tail length is required: ' + sele_id)
					else:
						n_drop_unknown_nt += 1
				else:
					n_drop_short_utr += 1
			else:
				n_drop_fa_string_filter += 1
	if len(X_mat_lst) > 0:
		if params_global['flag_initial_tl']:
			return(
				np.stack(X_mat_lst, axis = 0), 
				pd.DataFrame({'id':id_lst, 'y':y_lst, 'label':label_lst, 'itl':itl_lst}), 
				n_drop_fa_string_filter, 
				n_drop_short_utr, 
				n_drop_unknown_nt)
		else:
			return(
				np.stack(X_mat_lst, axis = 0), 
				pd.DataFrame({'id':id_lst, 'y':y_lst, 'label':label_lst}), 
				n_drop_fa_string_filter, 
				n_drop_short_utr, 
				n_drop_unknown_nt)
	else:
		return(None)

def target_transform(x, method = 'none', inverse = False, offset = 0, tf = None):
	# x can be either a list of a 2-d array
	# a function to transform or inverse-transform target values
	# it returns the transformed values and the transformer (None value if not applicable)
	# if method is 'boxcox' or 'yeo-johnson' and 'inverse' is False, it fits and transforms if 'tf' is not provided, or it only transforms if 'tf' is provided 
	# if method is 'boxcox' or 'yeo-johnson' and 'inverse' is True, a fitted transformer must be provided to 'tf'
	if len(np.asarray(x).shape) == 1:
		flag_1d = True
	else:
		flag_1d = False
	if method == 'diff':
		if inverse:
			x_trans = x + params_global['tl_cons']
		else:
			x_trans = x - params_global['tl_cons']
	elif method == 'log':
		if inverse:
			x_trans = np.expm1(x) - offset
		else:
			x_trans = np.log1p(x + offset)
	elif method == 'sqrt':
		if inverse:
			x_trans = np.power(x,2) - offset
		else:
			x_trans = np.sqrt(x + offset)
	elif method == 'box-cox' or method == 'yeo-johnson':
		if method == 'box-cox':
			pt = PowerTransformer(method = 'box-cox')
		else:
			pt = PowerTransformer(method = 'yeo-johnson')
		if inverse:
			pt = tf
			if flag_1d:
				x_trans = (pt.inverse_transform(np.asarray(x).reshape((-1,1)))).ravel() - offset
			else:
				x_trans = pt.inverse_transform(x) - offset
		else:
			if flag_1d:
				x = np.asarray(x).reshape((-1,1))
			if tf != None: # if the transformer is provided, use it for transformation
				pt = tf
			else:
				pt.fit(x + offset)
				tf = pt
			if flag_1d:
				x_trans = (pt.transform(x + offset)).ravel()
			else:
				x_trans = pt.transform(x + offset)
	else:
		x_trans = x
	return(x_trans, tf)

def model_inn(x_train, y_train, x_test, y_test, params, best_model = False, early_stop = False):
	# make sure input shape is correct
	if params['flag_initial_tl']:
		params['input_shape'] = x_train[0].shape[1:]
	else:
		params['input_shape'] = x_train.shape[1:]

	if params_global['n_dimension'] == 3:
		model = inn_models.cinn_model(input_params = params)
	else:
		if args.n == 'resnet':
			# resnet model
			model = inn_models.resnet_model(input_params = params)
		else:
			# inn model
			model = inn_models.inn_model(input_params = params)

	if params_global['flag_print_model']:
		print(model.summary())
		model.summary(print_fn = lambda x: f_log.write(x + '\n'))
		params_global['flag_print_model'] = False

	# monitor metric for the best model
	callbacks = [keras.callbacks.ModelCheckpoint(args.o + 'Models/' + idf + '_callback_best_model.h5', monitor='val_loss', mode='min', save_best_only=True)]
	if early_stop:
		callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10))

	out = model.fit(x = x_train, y = y_train, batch_size = params['batch_size'], epochs = params['epochs'], verbose = args.v, shuffle = True, 
		callbacks = callbacks, validation_data=(x_test, y_test))

	if best_model:
		pwrite(f_log, 'Return the best model...')
		model = keras.models.load_model(args.o + 'Models/' + idf + '_callback_best_model.h5', custom_objects={'custom_metric': inn_models.custom_metric})
	else:
		pwrite(f_log, '\nReturn the final model...')

	return(out, model, params)

@use_named_args(inn_params_lst) 
def fitness(n_Conv1D_filter, n_Conv1D_kernal, n_Conv1D_MaxPool_block, 
	n_Conv2D_filter, n_Conv2D_kernal, n_Conv2D_MaxPool_block, 
	n_dense_neuron, n_pool_size, rnn_func, n_rnn_units, activation_func, l1_reg, l2_reg, 
	dropout_rate, optimizer, learning_rate, batch_size, epochs):
	# this is the objective function for optimization of hyperparameters
	params_dict = {
		'n_Conv1D_filter': n_Conv1D_filter,
		'n_Conv1D_kernal': int(n_Conv1D_kernal),
		'n_Conv1D_MaxPool_block': n_Conv1D_MaxPool_block,
		'n_Conv2D_filter': n_Conv2D_filter,
		'n_Conv2D_kernal': int(n_Conv2D_kernal),
		'n_Conv2D_MaxPool_block': n_Conv2D_MaxPool_block,
		'n_dense_neuron' : n_dense_neuron,
		'n_pool_size': int(n_pool_size),
		'rnn_func': rnn_func,
		'n_rnn_units': n_rnn_units, 
		'activation_func': activation_func,
		'l1_reg': l1_reg,
		'l2_reg': l2_reg,
		'dropout_rate' : dropout_rate,
		'optimizer' : optimizer,
		'learning_rate' : learning_rate,
		'batch_size' : batch_size,
		'epochs' : epochs
	}
	
	for key in inn_params:
		if key not in params_dict:
			params_dict.update({key: inn_params[key]})

	global opti_count
	opti_count += 1
	pwrite(f_log, '\nTry a new set of hyperparameters for model ' + str(opti_count) + ' out of ' + str(args.op) + '...' + timer())
	out, model, params = model_inn(X_train, Y_train_trans, X_test, Y_test_trans, params_dict, best_model = True, early_stop = True)
	Y_pred_trans = model.predict(X_test).ravel()
	Y_pred = target_transform(Y_pred_trans, method = params_global['y_mode'], offset = y_offset, inverse = True, tf = tf_fitted)[0]
	r2 = stats.linregress(Y_test_trans,Y_pred_trans).rvalue ** 2
	pwrite(f_log, 'R-squred: %.3f' % (r2) + '\n')
	global best_r2
	if r2 > best_r2:
		model.save(args.o + 'Models/' + idf + '_global_best_model.h5')
		best_r2 = r2

		# write out predictions
		out_pred = args.o + idf + '_test_prediction_results.txt'
		with open(out_pred, 'w') as f_res:
			f_res.write('id\ty\ty_pred\ty_trans\ty_pred_trans\tlabel\n')
			for k in range(Y_test_pd.shape[0]):
				f_res.write('\t'.join(list(map(str, [Y_test_pd['id'].to_numpy()[k], Y_test[k], Y_pred[k], 
					Y_test_trans[k], Y_pred_trans[k], Y_test_pd['label'].to_numpy()[k]]))) + '\n')

		if os.path.exists('Scatter_plot.R'):
			command = 'Rscript Scatter_plot.R ' + out_pred
			subprocess.Popen(shlex.split(command)).communicate()

	del model
	K.clear_session()
	return(-r2)

def loss_plot(train, test, fn):
	# plot training history
	pyplot.clf()
	pyplot.plot(train, label='train')
	pyplot.plot(test, label='test')
	pyplot.legend()
	pyplot.xlabel("Epochs")
	pyplot.ylabel("Loss (mse)")
	if not os.path.exists(args.o + 'Loss_plots'):
		os.mkdir(args.o + 'Loss_plots')
	pyplot.savefig(args.o + 'Loss_plots/' + fn)

######------------------------------------------------------------------------------------------------------------
t_start = time() # timer start
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--difference', dest = 'd', type = str, help = 'file with ID and tail-length difference as target values')
parser.add_argument('-i', '--initial', dest = 'i', type = str, help = 'file with ID and initial tail length')
parser.add_argument('-u', '--utr', dest = 'u', type = str, help = 'Fasta file for 3\'-UTR sequences')
parser.add_argument('-c', '--cds', dest = 'c', type = str, help = 'Fasta file for CDS sequences')
parser.add_argument('-f', '--fold', dest = 'f', type = str, help = 'file with base-paring probability predicted by RNAfold')
parser.add_argument('-md', '--model', dest = 'md', type = str, help = 'model file for prediction in h5 format')
parser.add_argument('-n', '--network', dest = 'n', type = str, default = 'inn', help = 'Network for the model')
parser.add_argument('-p', '--parameters', dest = 'p', type = str, help = 'text file with global and INN parameters (use ": " as delimiter)')
parser.add_argument('-o', '--output', dest = 'o', type = str, default = 'INN_out', help = 'folder for output files')
parser.add_argument('-l', '--lenMax', dest = 'l', type = int, help = 'Maximal length of sequence (from 3\'-end) used in the analysis')
parser.add_argument('-t', '--transform', dest = 't', type = str, default = 'none', help = 'how to transform Y, can be "none", "diff", log", "sqrt", "box-cox", "yeo-johnson"')
parser.add_argument('-v', '--verbose', dest = 'v', type = int, default = 0, help = 'verbose for INN')
parser.add_argument('-s', '--skip', dest = 's', action='store_true', help = 'whether to skip sequence conversion')
parser.add_argument('-x', '--x', dest = 'x', type = str, help = 'file name (.npy) for X sparse matrix (features)')
parser.add_argument('-y', '--y', dest = 'y', type = str, help = 'file name (.csv) for Y (target values and associated labels)')
parser.add_argument('-m', '--mode', dest = 'm', type = str, default = 'test', help = 'mode to run this script, can be "cv", "opti", "test", or "predict"')
parser.add_argument('-op', '--hyperparameters', dest = 'op', type = int, default = 100, help = 'Number of hyperparameter combinations')
args = parser.parse_args()


# output folder
if not os.path.exists(args.o):
	os.makedirs(args.o)
if not args.o.endswith('/'):
	args.o = args.o + '/'
if not os.path.exists(args.o + 'Models'):
	os.mkdir(args.o + 'Models')	

# update global parameters for a input file
if args.p:
	with open(args.p, 'r') as f:
		for line in f:
			lst = line.strip().split(': ')
			if lst[0] in params_global:
				params_global[lst[0]] = type(params_global[lst[0]])(lst[1])
			elif lst[0] in inn_params:
				inn_params[lst[0]] = type(inn_params[lst[0]])(lst[1])

# if the following paramters are specified in the input, update with these
# update the length 
if args.l:
	params_global['len_max'] = args.l

# update the number of channels
if args.c:
	params_global['n_channel'] += 1
if args.f and params_global['n_dimension'] == 2:
	params_global['n_channel'] += 1

# update the type of transformation for target values
if args.t not in ['none', 'diff', 'log', 'sqrt', 'box-cox', 'yeo-johnson']:
	pwrite(f_log, '\nWarning! No target transformation will be performed because the input transformation function is not supoorted: ' + args.t)
	params_global['y_mode'] = 'none'
else:
	params_global['y_mode'] = args.t

# construct out file prefix based on different input files
if args.s:
	if args.x is None or args.y is None:
		parser.error('In case of skipping sequence conversion, data for X (feature matrix) and Y (target values)!')
	idf = args.x.split('X_matrix_')[-1].split('.')[0]
	f_log = make_log_file(args.o + idf + '_simple_log.txt', p_params = params_global, p_vars = vars(args))
	pwrite(f_log, 'Sequence conversion skipped.')
else:
	if args.d is None:
		if args.u is None:
			parser.error('Input file with sequences is required!')
		else:
			idf = args.u.split('/')[-1].split('.')[0]
	else:
		idf = args.d.split('/')[-1].split('.')[0]
	f_log = make_log_file(args.o + idf + '_complete_log.txt', p_params = params_global, p_vars = vars(args))

###---------------------------------------------------------------------------------------------------------
if args.s is False: # if pre-converted data not provided
	
	# make a dictionary of target values (required input in this case)
	y_dict = {}
	if args.d:
		pwrite(f_log, '\nInput file with target values:\n' + args.d)
		pwrite(f_log, '\nMake a dictionary of target values...' + timer())
		with open(args.d, 'r') as f:
			if params_global['flag_input_header']:
				f.readline()
			for i, line in enumerate(f):
				lst = line.strip().split('\t')
				if len(lst) <= 2 or int(lst[2]) >= params_global['input_tag_min']: # the 3rd column is the number of tags
					if lst[0] in y_dict:
						sys.exit('Error! Redundant IDs in the input file: ' + lst[0])
					y_dict[lst[0]] = float(lst[1])
		pwrite(f_log, 'Number of total entries in the input file: ' + str(i+1))
		pwrite(f_log, 'Number of entries included in the downstream analysis: ' + str(len(y_dict)))
	else:
		pwrite(f_log, '\nNo targets values are provided. Likely in prediction mode...' + timer())

	# if inital tail length tail length is provided, make a disctionary
	if args.i:
		pwrite(f_log, '\nInput file with initial tail lengths:\n' + args.i)
		pwrite(f_log, '\nMake a dictionary...' + timer())
		itl_dict = {}
		with open(args.i, 'r') as f:
			f.readline() # header
			for line in f:
				lst = line.strip().split('\t')
				if lst[0] in itl_dict:
					sys.exit('Error! Redundant IDs in the file: ' + lst[0])
				itl_dict[lst[0]] = float(lst[1])
		if len(itl_dict) > 0:
			params_global['flag_initial_tl'] = True
			inn_params['flag_initial_tl'] = True
		pwrite(f_log, 'Number of entries in the file with initial tail lengths: ' + str(len(itl_dict)))

	# if CDS sequence file is provided, make a dictionary 
	if args.c:
		pwrite(f_log, '\nInput CDS file:\n' + args.c)
		pwrite(f_log, '\nMake a dictionary of CDS sequences...' + timer())
		cds_dict = fasta_to_dict(args.c)
		pwrite(f_log, 'Number of entries in the fasta file: ' + str(len(cds_dict)))
	else:
		pwrite(f_log, '\nNo CDS sequences are provided. Use only 3\'-UTR sequences...' + timer())

	# if RNA fold file is provided, make a dictionary 
	if args.f:
		pwrite(f_log, '\nInput RNA fold data:\n' + args.f)

		if params_global['n_dimension'] == 3:
			pwrite(f_log, '\nLoad RNA fold data...' + timer())
			fold_h5 = h5py.File(args.f)
		else:
			pwrite(f_log, '\nMake a dictionary of RNA fold data...' + timer())
			fold_dict = {}
			with open(args.f, 'r') as f:
				while(True):
					line = f.readline()
					if line:
						sele_id = fasta_id_parser(line, sep = '__', pos = 0)
						f.readline()
						pp_lst = f.readline().strip().split('\t')
						if sele_id not in fold_dict:
							fold_dict[sele_id] = pp_lst
						else:
							sys.exit('Error! Redundant IDs in the RNA fold data file: ' + sele_id)
					else:
						break
	else:
		pwrite(f_log, '\nNo RNA fold data is provided.' + timer())

	# convert sequence to 2D or 3D matrix
	pwrite(f_log, '\nRead in the fasta sequence file...')
	with open(args.u, 'r') as f:
		line_lst = []
		counting = 0
		n_drop_fa_string_filter = 0
		n_drop_short_utr = 0
		n_drop_unknown_nt = 0
		X_mat_lst = []
		Y_df_lst = []
		temp_seq = ''
		while(True):
			line = f.readline()
			if not line:
				line_lst.append([current_id_line, temp_seq]) # the last entry
				with concurrent.futures.ProcessPoolExecutor(min(params_global['n_threads'], len(line_lst))) as pool:
					futures = pool.map(process_seqs, np.array_split(line_lst, min(params_global['n_threads'], len(line_lst))))
					for future in futures:
						if future is not None:
							ary, df, n1, n2, n3 = future
							X_mat_lst.append(ary)
							Y_df_lst.append(df)
							n_drop_fa_string_filter += n1
							n_drop_short_utr += n2
							n_drop_unknown_nt += n3
					pwrite(f_log, str(counting) + ' sequences processed...' + timer())
				break
			else:
				if line[0] == '>':
					counting += 1
					if temp_seq != '':
						line_lst.append([current_id_line, temp_seq])
						temp_seq = ''
					current_id_line = line
					if counting % params_global['chunk'] == 0:
						with concurrent.futures.ProcessPoolExecutor(params_global['n_threads']) as pool:
							futures = pool.map(process_seqs, np.array_split(line_lst, params_global['n_threads']))
							for future in futures:
								if future is not None:
									ary, df, n1, n2, n3 = future
									X_mat_lst.append(ary)
									Y_df_lst.append(df)
									n_drop_fa_string_filter += n1
									n_drop_short_utr += n2
									n_drop_unknown_nt += n3
							pwrite(f_log, str(counting) + ' sequences processed...' + timer())
						line_lst = []
				else:
					temp_seq += line.strip()
	
	pwrite(f_log, 'Total number of sequences in the input file: ' + str(counting))
	pwrite(f_log, 'Number of sequences removed because sequences in the fasta file is uncertain: ' + str(n_drop_fa_string_filter))
	pwrite(f_log, 'Number of sequences removed because 3\'-UTR seqeunces shorter than ' + str(params_global['len_min']) +': ' + str(n_drop_short_utr))
	pwrite(f_log, 'Number of sequences removed because of un-recogenized characters in the sequence: ' + str(n_drop_unknown_nt))
	X_ary = np.vstack(X_mat_lst)
	Y_df = pd.concat(Y_df_lst, ignore_index = True)
	
	np.save(args.o + 'temp_X_matrix_' + idf + '.npy', X_ary)
	Y_df.to_csv(args.o + 'temp_Y_df_' + idf + '.csv')

else:
	X_ary = np.load(args.x)
	#params_global['n_channel'] = X_ary.shape[1]
	Y_df = pd.read_csv(args.y)

# check dimensionality
pwrite(f_log, 'Data matrix dimension, X: ' + str(X_ary.shape) + ' Y: ' + str(Y_df.shape))
if Y_df.shape[0] != X_ary.shape[0]:
	pwrite(f_log, 'Warining! Dimension of input matrices and the number of target values don\'t match!')

# update input shape
inn_params['input_shape'] = X_ary.shape[1:]

###---------------------------------------------------------------------------------------------------------
# check skewness of target Y values
if len(y_dict) != 0:
	if np.min(Y_df['y']) <= 0:
		y_offset = abs(math.floor(np.min(Y_df['y']))) + 1
		pwrite(f_log, "Target Y contains negative values. Apply offset by " + str(y_offset) + " for log, sqrt, boxcox transformations.")
	else:
		y_offset = 0
	pwrite(f_log, "\nCheck the overall skewness of target values if the following tranformation is applied: ")
	for tm in ['none', 'log', 'sqrt', 'box-cox', 'yeo-johnson']:
		val_skew = stats.skew(target_transform(Y_df['y'].to_numpy(), method = tm, offset = y_offset)[0])
		pwrite(f_log, tm + ": %.3f" % val_skew)

	pwrite(f_log, '\nType of transformation of target values: ' + params_global['y_mode'])
else:
	pwrite(f_log, '\nPrediction mode, skewness not checked.')

###---------------------------------------------------------------------------------------------------------
# train data with INN

# for optimization of hyperparameters
if args.m.startswith('op'):
	pwrite(f_log, '\nPerforming hyperparameter search...'+ timer())
	pwrite(f_log, 'Ratio between testing and training sets: ' + str(params_global['test_size']))
	params_global['flag_print_model'] = False

	# split data into training and testing groups
	X_train, X_test, Y_train_pd, Y_test_pd = train_test_split(X_ary, Y_df, test_size = params_global['test_size'], random_state = 57, shuffle = True, stratify = Y_df['label'])
	Y_train = Y_train_pd['y'].to_numpy()
	Y_test = Y_test_pd['y'].to_numpy()
	if params_global['flag_initial_tl']:
		X_train = [X_train, Y_train_pd['itl'].to_numpy()]
		X_test = [X_test, Y_test_pd['itl'].to_numpy()]

	# transform target values
	Y_train_trans, tf_fitted = target_transform(Y_train, method = params_global['y_mode'], offset = y_offset)
	Y_test_trans = target_transform(Y_test, method = params_global['y_mode'], offset = y_offset, tf = tf_fitted)[0]

	# optimize with objective function
	best_r2 = 0
	opti_count = 0 
	search_result = gp_minimize(func = fitness, dimensions = inn_params_lst, acq_func = 'EI', n_calls = args.op)
	
	# make a few plots for the parameters
	plot_convergence(search_result)
	pyplot.savefig(args.o + idf + '_hyperparameter_search_convergence_plot.png')
	#plot_evaluations(search_result)
	#pyplot.savefig(args.o + idf + '_hyperparameter_search_evaluation_plot.png')
	plot_objective(search_result)
	pyplot.savefig(args.o + idf + '_hyperparameter_search_objective_plot.png')
	
	# write out all parameters
	with open(args.o + idf + '_hyperparameter_search_stats.txt', 'w') as f:
		f.write('objective_value' + '\t' + '\t'.join([x.name for x in inn_params_lst]) + '\n')
		for obj, lst in sorted(zip(search_result.func_vals, search_result.x_iters)):
			f.write(str(obj) + '\t' + '\t'.join(list(map(str, lst))) + '\n')
	
	# use the best model to predict the test set
	model = keras.models.load_model(args.o + 'Models/' + idf + '_global_best_model.h5', custom_objects={'custom_metric': inn_models.custom_metric})
	Y_pred_trans = model.predict(X_test).ravel()
	Y_pred = target_transform(Y_pred_trans, method = params_global['y_mode'], offset = y_offset, inverse = True, tf = tf_fitted)[0]
	r_value = stats.linregress(Y_test, Y_pred).rvalue
	pwrite(f_log, 'R-squred the best model prediction: %.3f' % (r_value ** 2) + '\n')

	# write out predictions
	out_pred = args.o + idf + '_test_prediction_results.txt'
	with open(out_pred, 'w') as f_res:
		f_res.write('id\ty\ty_pred\ty_trans\ty_pred_trans\tlabel\n')
		for k in range(Y_test_pd.shape[0]):
			f_res.write('\t'.join(list(map(str, [Y_test_pd['id'].to_numpy()[k], Y_test[k], Y_pred[k], 
				Y_test_trans[k], Y_pred_trans[k], Y_test_pd['label'].to_numpy()[k]]))) + '\n')

# for training and testing with defined hyperparameters
elif args.m == 'cv' or args.m == 'test':
	pwrite(f_log, '\nUse a INN model for training and testing in a CV fold of ' + str(int(1/params_global['test_size'])) + ' with the following hyperparameters...' + timer())
	if not os.path.exists(args.o + 'Predictions'):
		os.mkdir(args.o + 'Predictions')	

	for hp in inn_params:
		pwrite(f_log, hp + ': ' + str(inn_params[hp]))
	
	# split data into k-fold, stratified by whether 
	sfk_split = StratifiedKFold(n_splits = int(1/params_global['test_size']), shuffle = True, random_state = 57).split(X_ary, Y_df['label'])

	out_pred = args.o + idf + '_train_test_CV_prediction_results.txt'
	with open(out_pred, 'w') as f_res, open(args.o + idf + '_train_test_CV_stats.txt', 'w') as f_stat:
		f_stat.write('group\trep\tr2\tloss\tvalue\n')
		f_res.write('group\tid\ty\ty_pred\ty_trans\ty_trans_pred\tlabel\n')
		for i, (train_idx, test_idx) in enumerate(sfk_split):
			best_r2 = 0
			for j in range(params_global['n_rep']):
				X_train = X_ary[train_idx]
				X_test = X_ary[test_idx]
				Y_train = Y_df.loc[train_idx]['y'].to_numpy()
				Y_test = Y_df.loc[test_idx]['y'].to_numpy()
				if params_global['flag_initial_tl']:
					X_train = [X_train, Y_df.loc[train_idx]['itl'].to_numpy()]
					X_test = [X_test, Y_df.loc[test_idx]['itl'].to_numpy()]

				# transform target values
				Y_train_trans, tf_fitted = target_transform(Y_train, method = params_global['y_mode'], offset = y_offset)
				Y_test_trans = target_transform(Y_test, method = params_global['y_mode'], offset = y_offset, tf = tf_fitted)[0]
				
				# train model
				pwrite(f_log, '\nTraining with CV group ' + str(i+1) + ' out of ' + str(int(1/params_global['test_size'])) + ', replicate ' + str(j+1) + ' out of ' + str(params_global['n_rep']))
				history, model, params = model_inn(X_train, Y_train, X_test, Y_test, inn_params, best_model = True, early_stop = False)
						
				# predict the test set and evaluate:
				model.save(args.o + 'Models/' + idf + '_best_model_CV_group_'+ str(i+1) + '_rep_' + str(j+1) + '.h5')
				scores = model.evaluate(X_test, Y_test_trans, batch_size=params['batch_size'], verbose = args.v)
				Y_pred_trans = model.predict(X_test, verbose = args.v).ravel()
				Y_pred = target_transform(Y_pred_trans, method = params_global['y_mode'], offset = y_offset, inverse = True, tf = tf_fitted)[0]
				r_value = stats.linregress(Y_test, Y_pred).rvalue
				pwrite(f_log, 'R-squred: %.3f' % (r_value ** 2) + '\n')
				if r_value ** 2 > best_r2:
					best_r2 = r_value ** 2
					best_model = model 
				del model
				K.clear_session()

				# output stats
				f_stat.write('\t'.join(list(map(str, [i+1, j+1, '%.3f' % (r_value ** 2)] + scores))) + '\n')
				
				# make train-test loss line plot
				loss_plot(train = history.history['loss'], test = history.history['val_loss'], 
					fn = idf + '_loss_over_epoch_line_plot_CV_group_' + str(i+1) + '_rep_' + str(j+1) +'.png')
				loss_plot(train = history.history['custom_metric'], test = history.history['val_custom_metric'], 
					fn = idf + '_metric_over_epoch_line_plot_CV_group_' + str(i+1) + '_rep_' + str(j+1) +'.png')

				if args.m != 'cv': # only do this once if not in 'cv' mode
					break

			# output preditions with the best model for this CV group
			best_model.save(args.o + 'Models/' + idf + '_best_model_CV_group_'+ str(i+1) + '.h5')
			Y_pred_trans = best_model.predict(X_test, verbose = args.v).ravel()
			Y_pred = target_transform(Y_pred_trans, method = params_global['y_mode'], offset = y_offset, inverse = True, tf = tf_fitted)[0]
			for k in range(len(Y_test)):
				f_res.write('\t'.join(list(map(str, [i+1, Y_df.loc[test_idx]['id'].to_numpy()[k], Y_test[k], Y_pred[k], 
					Y_test_trans[k], Y_pred_trans[k], Y_df.loc[test_idx]['label'].to_numpy()[k]]))) + '\n')

			if args.m != 'cv': # only do this once if not in 'cv' mode
				break

else: # args.m == 'predict'
	# Load pre-trained INN model
	pwrite(f_log, '\nPredict with pre-trained model.')
	pwrite(f_log, 'Input model file:\n' + args.md)
	model = keras.models.load_model(args.md, custom_objects={'custom_metric': inn_models.custom_metric})
	print(model.summary())
	model.summary(print_fn = lambda x: f_log.write(x + '\n'))
	model_input_shape = model.get_layer(index = 0).input_shape
	pwrite(f_log, 'model_input_shape: '+ str(model_input_shape))
	#if model_input_shape[1] != params_global['len_max']:
	#	sys.exit('Length of the input sequence (' + str(params_global['len_max']) + ') is incompatible with the model input (' + str(model_input_shape[1]) + ')!')

	# Predict with pre-trained INN
	if params_global['flag_initial_tl']:
		Y_pred_trans = model.predict([X_ary,Y_df['itl'].to_numpy()]).ravel()
	else:
		Y_pred_trans = model.predict(X_ary).ravel()

	# write out predictions
	out_pred = args.o + idf + '_prediction_results.txt'

	if 'NA' not in Y_df['y'].to_numpy():
		Y_pred = target_transform(Y_pred_trans, method = params_global['y_mode'], offset = y_offset, inverse = True)[0]
		y_trans = target_transform(Y_df['y'], method = params_global['y_mode'], offset = y_offset)[0]
		r_value = stats.linregress(Y_df['y'].to_numpy(), Y_pred).rvalue
		pwrite(f_log, 'R-squred the model prediction: %.3f' % (r_value ** 2) + '\n')

		with open(out_pred, 'w') as f_res:
			f_res.write('id\ty\ty_pred\ty_trans\ty_trans_pred\tlabel\n')
			for k in range(Y_df.shape[0]):
				f_res.write('\t'.join(list(map(str, [Y_df['id'].to_numpy()[k], Y_df['y'].to_numpy()[k], Y_pred[k], 
					y_trans[k], Y_pred_trans[k], Y_df['label'].to_numpy()[k]]))) + '\n')

	else:
		with open(out_pred, 'w') as f_res:
			f_res.write('id\ty_pred\tlabel\n')
			for k in range(Y_df.shape[0]):
				f_res.write('\t'.join(list(map(str, [Y_df['id'].to_numpy()[k], Y_pred_trans[k], Y_df['label'].to_numpy()[k]]))) + '\n')

####---------------------------------------------------------------------------------------------------------
# Make a plot for comparing measured and predicted values (require an R script "Scatter_plot.R")
if os.path.exists('Scatter_plot.R'):
	command = 'Rscript Scatter_plot.R ' + out_pred
	subprocess.Popen(shlex.split(command)).communicate()
else:
	pwrite(f_log, 'R script "Scatter_plot.R" not found. No plots made.')

pwrite(f_log, 'Finished...' + timer())
f_log.close()

