import sys, concurrent.futures, math, argparse, re, os, matplotlib, h5py, yaml, optuna, gzip, subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import inn_models, inn_utils

from inn_utils import timer, pwrite, make_log_file, fasta_id_parser, fasta_to_dict, encode_seq, target_transform, loss_plot, scatter_plot, update_dict_from_dict, save_opti_results, update_and_log

from time import time
from tensorflow import keras
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats
from collections import OrderedDict

matplotlib.use('Agg')

# Default global parameters can be updated by the input yaml file with the '-p' option
# some parameters can also be updated by the input options and they will override the default values and values provided in the yaml file
global_config = OrderedDict([
	('NAME', 'Global parameters'),
	('fa_string_filter', 'uncertain'), # if fasta entry contains this string, it will be filtered out
	('flag_input_header', True), # whether the input file with target values has a header
	('flag_initial_tl', False), # whether to use initial tail length as an input for learning
	('input_tag_min', 50), # minimal number of tags for each gene to be included from the input file
	('seq_3end_append', 'AAA'), # sequence appended at the 3'-end
	('len_max', 1000), # maximal length of sequence (from 3' ends, after appending the constant region, if applicable). This can also be updated with the '-l' option.
	('len_min', 10), # minimal length of 3' UTR sequence to be included, from 3' ends (those smaller than this won't be analyzed)
	('n_cpe_max', 4), # max number of CPEs for labeling each mRNA (if an mRNA has more than this number of CPEs, the 'label' will be set at this number) 
	('y_mode', 'none'), # how to transform target (tail length) value, can be one of 'none', 'diff', 'log', 'sqrt', 'box-cox', or 'yeo-johnson'
	('y_offset', 0), # an offset value applied to the target (tail length) value. Automatically modified based on the input data.
	('tl_cons', 35), # tail length constant to be subtracted if 'y_mode' is set to 'diff'
	('test_size', 0.1), # fraction of the dataset used as test set; also used for defining CV fold 
	('n_dimension', 2), # dimension of the input for the INN (2 means simple one-hot encoding, 3 means outer product of the one-hot encoding)
	('n_rep', 5), # number of repeated training with 1 cross-validation group
	('flag_print_model', True), # whether to print out the model summary
	('out_folder', 'Output'), # Folder for output files. If None, a default value 'Output' will be used. This can also be updated with the '-o' option.
	('idf', None), # unique prefix for output file names. If None, it will be automatically generated based on input files. 
	('verbose', 0), # output verbose for model training
	('n_threads', 20),
	('chunk', 10000)
])

inn_params = OrderedDict([
	('NAME', 'Neural Network parameters'),
	('input_shape', [1056,5]), # this will be automatically updated based on the input 
	('flag_initial_tl', False),
	#('flag_mfe', False),
	('n_Conv1D_filter', 64),
	('n_Conv1D_kernel', 5),
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
	('resnet_version', 2), 
	('resnet_n_group', 7), # number of ResNet groups
	('resnet_n_block', 4), # number of ResNet blocks per group
	('dilation_rate_lst', [1,2,4,8,4,2,1]), # dilation rate, length of the list must be the same as 'resnet_n_group'
	# inn-specific parameters
	('n_Conv1D_MaxPool_block', 1),
	('n_pool_size', 2),
	# cinn-specific parameters
	('n_Conv2D_filter', 32),
	('n_Conv2D_kernel', 5),
	('n_Conv2D_MaxPool_block', 1)
])

inn_params_opti = OrderedDict([
	('NAME', 'Optimization parameters'),
	# values are all categoriy lists except those for 'l1_reg', 'l2_reg', 'learning_rate'
	('n_Conv1D_filter', [32, 64, 128]),
	('n_Conv1D_kernel', [4, 5, 6]),
	('n_Conv1D_MaxPool_block', [1, 2, 3]),
 	('n_Conv2D_filter', [64]),
	('n_Conv2D_kernel', [5]),
	('n_Conv2D_MaxPool_block', [1]),
	('n_dense_neuron', [32, 64, 96]),
	('n_pool_size', [2]),
	('rnn_func', ['GRU', 'BiGRU', 'LSTM', 'BiLSTM']),
	('n_rnn_units', [32, 64, 128]),
	('resnet_version', [1, 2]),
	('resnet_n_group', [7]), 
	('resnet_n_block', [4]),
	('activation_func', ['selu', 'silu', 'leaky_relu']),
	('l1_reg', {'min': 0.000001, 'max': 0.01}),
	('l2_reg', {'min': 0.000001, 'max': 0.01}),
	('dropout_rate', [0.1, 0.2, 0.3]),
	('optimizer', ['adam', 'SGD']),
	('learning_rate', {'min': 0.00001, 'max': 0.01}),
	('loss', ['mse', 'mae']),
	('batch_size', [32, 64, 128]),
	('epochs', [30, 40, 50])
])

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
	n_utr_w_cds = 0 # number of UTRs with CDS sequences appended
	
	for lines in line_lst:
		sele_id = fasta_id_parser(lines[0], sep = '__')[0]
		if len(fasta_id_parser(lines[0], sep = '__')) > 1:
			transcript_id = fasta_id_parser(lines[0], sep = '__')[1] 
		else:
			transcript_id = fasta_id_parser(lines[0], sep = '__')[0]

		# only convert sequences that have y values, or in "prediction mode", y values are not necessary
		global y_dict
		if sele_id in y_dict or len(y_dict) == 0: 
			if len(y_dict) == 0:
				tl = 'NA'
			else:
				tl = y_dict[sele_id]

			if (global_config['fa_string_filter'] is None) or (global_config['fa_string_filter'] not in lines[0]): # exclude "uncertain" annotations
				seq = lines[1].rstrip()
				
				# append the 3' end sequence (default: 3 nt from the poly(A) tail)
				seq = seq + global_config['seq_3end_append']

				#seq = seq[:-100] # use this line to truncate 3'-UTR sequence from 3'-end
				
				if len(seq) >= global_config['len_min']: # UTR must be longer than this

					# truncate long UTR sequences and also pad short UTR sequences with N at the 5' ends
					utr_seq = seq 
					if len(seq) >= global_config['len_max']: 
						seq = seq[-global_config['len_max']:] 
					else:	
						seq = 'N' * (global_config['len_max'] - len(seq)) + seq

					if global_config['n_dimension'] == 2: 
						# add the CDS track only if the architecture is in 2D

						# add the CDS track 
						if args.c:
							if len(utr_seq) >= global_config['len_max']:
								anno_track = [1 for i in range(len(seq))]
							else:
								anno_track = [0 for i in range(global_config['len_max']-len(utr_seq))] + [1 for i in range(len(utr_seq))]
								
								# add the CDS sequence if provided
								if 'cds_dict' in globals():
									if transcript_id in cds_dict or len(cds_dict) == 1: # if only one entry in the CDS dictionary, treat this as universal CDS for all UTR3 variants
										n_utr_w_cds += 1
										if transcript_id in cds_dict:
											cds_seq = cds_dict[transcript_id]
										else:
											cds_seq = list(cds_dict.values())[0]
											
										if len(cds_seq) > global_config['len_max']-len(utr_seq):
											seq = (cds_seq + utr_seq)[-global_config['len_max']:]
										else:
											seq = 'N'*(global_config['len_max']-len(cds_seq) - len(utr_seq)) + cds_seq + utr_seq

						# encode the sequence
						seq = seq.upper().replace('U', 'T')
						if re.search('(^[ACGTNU]+$)',seq):
							seq_mat = encode_seq(seq, n_dim = global_config['n_dimension'], half_mask = True)
						else:
							pwrite(f_log, 'Error! Unknown characters in the sequence of this entry: ' + sele_id)
							n_drop_unknown_nt += 1
							break

						# add the cds track if provided
						if args.c:
							seq_mat = np.hstack((seq_mat, np.asarray(anno_track).reshape((-1,1))))

						# add the splice site track if provided
						if args.f:
							global fold_dict
							if sele_id in fold_dict:
								f_lst = list(map(float, fold_dict[sele_id][:]))
								if len(f_lst) < global_config['len_max']:
									f_lst = [0 for i in range(global_config['len_max'] - len(f_lst))] + f_lst
								else:
									f_lst = f_lst[-global_config['len_max']:]
								seq_mat = np.hstack((seq_mat, np.asarray(f_lst).reshape((-1,1))))
							else:
								sys.exit('Error! This pA site does not have RNA folding data: ' + sele_id)
							
					else:
						# if the architecture is in 3D, ignore the CDS track

						# encode the sequence
						seq = seq.upper().replace('U', 'T')
						if re.search('(^[ACGTNU]+$)',seq):
							seq_mat = encode_seq(seq, n_dim = global_config['n_dimension'], half_mask = True)
						else:
							pwrite(f_log, 'Error! Unknown characters in the sequence of this entry: ' + sele_id)
							n_drop_unknown_nt += 1
							break
							
						if args.f:
							global fold_h5
							if sele_id in fold_h5.keys():
								fold_ary = fold_h5[sele_id][:] # pairing probability of each nucleotide pair
								fold_seq = fold_h5[sele_id].attrs['sequence']
								if len(fold_seq) >= len(seq):
									fold_ary = fold_ary[(len(fold_seq) - len(seq)):, (len(fold_seq) - len(seq)):]
								else:
									fold_ary = np.pad(fold_ary, ((len(seq) - len(fold_seq),0),(len(seq) - len(fold_seq),0)), mode='constant', constant_values=0)
								
								seq_mat = np.concatenate((seq_mat, np.expand_dims(fold_ary, axis = 2)), axis = 2)
							else:
								sys.exit('Error! This pA site does not have RNA folding data: ' + sele_id)

					X_mat_lst.append(seq_mat)
					id_lst.append(sele_id)
					y_lst.append(tl)

					# label each entry with the number of CPE (TTTTA) it contains in the 3' UTR
					cpe_start_lst = [x.start() for x in re.finditer('TTTTA', seq)]
					if args.c and len(cpe_start_lst) > 0:
						cpe_start_lst = [x for x in cpe_start_lst if anno_track[x] == 1]
					label_lst.append(len(cpe_start_lst) if len(cpe_start_lst) < global_config['n_cpe_max'] else global_config['n_cpe_max'])
					
					# add the list of starting tail length if provided
					if global_config['flag_initial_tl']:
						global itl_dict
						if sele_id in itl_dict:
							itl_lst.append(float(itl_dict[sele_id]))
						else:
							sys.exit('No initial tail length for this entry when the initial tail length is required: ' + sele_id)
				else:
					n_drop_short_utr += 1
			else:
				n_drop_fa_string_filter += 1
	if len(X_mat_lst) > 0:
		if global_config['flag_initial_tl']:
			return(
				np.stack(X_mat_lst, axis = 0), 
				pd.DataFrame({'id':id_lst, 'y':y_lst, 'label':label_lst, 'itl':itl_lst}), 
				n_drop_fa_string_filter, 
				n_drop_short_utr, 
				n_drop_unknown_nt,
				n_utr_w_cds)
		else:
			return(
				np.stack(X_mat_lst, axis = 0), 
				pd.DataFrame({'id':id_lst, 'y':y_lst, 'label':label_lst}), 
				n_drop_fa_string_filter, 
				n_drop_short_utr, 
				n_drop_unknown_nt,
				n_utr_w_cds)
	else:
		return(None)

def model_nn(x_train, y_train, x_val, y_val, params, model_type = 'inn', best_model = False, early_stop = False, gpu_id = None):
	'''
	Train a neural network model with the specified parameters.

	Args:
		x_train, y_train: Training data and target values.
		x_val, y_val: Validation data and target values.
		params: Dictionary of hyperparameters.
		model_type: Type of model to train ('inn', 'resnet', etc.).
		best_model: If True, loads the best saved model after training.
		early_stop: If True, uses early stopping.
		gpu_id: GPU identifier.

	Returns:
		Tuple (training history, trained model, updated params).
	'''

	# Ensure global variables are initialized
	global global_config, inn_params, f_log
	
	# update missing parameters with defaults
	for key in inn_params:
		if key not in params:
			params.update({key: inn_params[key]})

	# make sure input shape is correct
	if params['flag_initial_tl']:
		assert params['input_shape'] == x_train[0].shape[1:]
	else:
		assert params['input_shape'] == x_train.shape[1:]
	
	# load the model for training
	if global_config['n_dimension'] == 3:
		model = inn_models.cinn_model(input_params = params)
	else:
		if model_type == 'resnet':
			# resnet model
			model = inn_models.resnet_model(input_params = params)
		else:
			# inn model
			model = inn_models.inn_model(input_params = params)

	# Print model summary if required
	if global_config['flag_print_model']:
		print(model.summary())
		model.summary(print_fn = lambda x: f_log.write(x + '\n'))
		global_config['flag_print_model'] = False

	# Define callbacks
	device = '_CPU' if gpu_id is None else f'_GPU_{gpu_id}'
	f_best_model = os.path.join(global_config['out_folder'], 'Models', f'{global_config["idf"]}{device}_callback_best_model.keras')
	callbacks = [keras.callbacks.ModelCheckpoint(f_best_model, monitor='val_loss', mode='min', save_best_only=True)]
	if early_stop:
		callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10))

	history = model.fit(
		x = x_train, y = y_train, 
		batch_size = params['batch_size'], 
		epochs = params['epochs'], 
		verbose = global_config['verbose'], 
		shuffle = True, 
		callbacks = callbacks, 
		validation_data=(x_val, y_val)
	)

	if best_model:
		pwrite(f_log, 'Return the best model...\n')
		try:
			model = keras.models.load_model(f_best_model)
			#model = keras.models.load_model(f_best_model, custom_objects={'custom_metric': inn_models.custom_metric})
		except Exception as e:
			pwrite(f_log, f"Error loading best model: {e}\n")
			pwrite(f_log, 'Return the final model...\n')
	else:
		pwrite(f_log, 'Return the final model...\n')

	return(history, model, params)
	
def optuna_objective(trial, x_train, y_train, x_val, y_val, params, model_type = 'inn', gpus = None, opti_log = None):
	'''
	Optuna objective function for hyperparameter optimization.

	Args:
		trial: Optuna trial object.
		x_train, y_train: Training data and target values.
		x_val, y_val: Validation data aand target values.
		params: Dictionary of hyperparameters.
		model_type: Type of model to train ('inn', 'resnet', etc.).
		gpus: List of available GPUs (optional).
		opti_log: Path to the log file for optimization results.

	Returns:
		Negative R-squared value for Optuna optimization.
	'''

	# Ensure global variables are initialized
	global global_config, tf_fitted, best_neg_r2, f_log
	
	# Assign GPU (round-robin method)
	if gpus:
		gpu_id = trial.number % len(gpus)  # Assign GPU based on trial number
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Limit TensorFlow to this GPU
	else:
		gpu_id = None
		#os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Use CPU if no GPUs are available

	# Sample hyperparameters
	params_opti = OrderedDict()
	for param in params:
		if param in ['l1_reg', 'l2_reg', 'learning_rate']:
			if params[param]['min'] < params[param]['max']:
				params_opti[param] = trial.suggest_float(param, params[param]['min'], params[param]['max'], log = True)
		else:
			if isinstance(params[param], list) and len(params[param]) > 1:
				params_opti[param] = trial.suggest_categorical(param, params[param])
			else:
				params_opti[param] = params[param][0] if isinstance(params[param], list) else params[param]

	# Train model using these hyperparameters
	history, out_model, out_params = model_nn(
		x_train, y_train, x_val, y_val, params_opti, 
		model_type = model_type, best_model = True, early_stop = True, gpu_id = gpu_id
	)

	# Predict using the trained model
	y_pred_trans = out_model.predict(x_val).ravel()

	# Transform y-values back to original scale
	try:
		y = target_transform(y_val, method=global_config['y_mode'], offset=global_config['y_offset'], inverse=True, transformer=tf_fitted, constant = global_config['tl_cons'])[0]
		y_pred = target_transform(y_pred_trans, method=global_config['y_mode'], offset=global_config['y_offset'], inverse=True, transformer=tf_fitted, constant = global_config['tl_cons'])[0]
	except Exception as e:
		raise ValueError(f"Error in target transformation: {e}")

	# Compute Negative R-squared for minimization
	neg_r2 = -stats.linregress(y,y_pred).rvalue ** 2	

	# Save trial results
	if opti_log:
		if not os.path.exists(opti_log):	
			with open(opti_log, 'w') as f:
				f.write('Trial number\tGPU ID\t' + '\t'.join(list(trial.params.keys())) + '\tneg_r2\tepochs (actual/planned)\n')
				f.write(f'{trial.number + 1}\t{"NA" if gpu_id is None else gpu_id}\t' +
					'\t'.join([str(trial.params[param]) for param in trial.params]) + '\t' +
					f'{neg_r2}\t{len(history.history["loss"])}/{out_params["epochs"]}\n')
		else:
			with open(opti_log, 'a') as f:	
				f.write(f'{trial.number + 1}\t{"NA" if gpu_id is None else gpu_id}\t' +
					'\t'.join([str(trial.params[param]) for param in trial.params]) + '\t' +
					f'{neg_r2}\t{len(history.history["loss"])}/{out_params["epochs"]}\n')

	if neg_r2 < best_neg_r2:
		best_neg_r2 = neg_r2
		os.makedirs(os.path.join(global_config['out_folder'], 'Scatter_plots'), exist_ok=True)
		os.makedirs(os.path.join(global_config['out_folder'], 'Loss_plots'), exist_ok=True)
		os.makedirs(os.path.join(global_config['out_folder'], 'Models'), exist_ok=True)

		# Generate scatter plot
		scatter_plot(y, y_pred, fn = os.path.join(global_config["out_folder"], 'Scatter_plots', 'Optimization_best_model_predicted_vs_measured_scatter_plot.png'))

		# Generate train-test loss plots
		loss_plot(train = history.history['loss'], test = history.history['val_loss'], 
			fn = os.path.join(global_config["out_folder"], 'Loss_plots', 'Optimization_best_model'), y_label = 'Loss')
		loss_plot(train = history.history[out_params['metrics']], test = history.history[f"val_{out_params['metrics']}"], 
			fn = os.path.join(global_config["out_folder"], 'Loss_plots', 'Optimization_best_model'), y_label = 'Metric')

		# save best model
		out_model.save(os.path.join(global_config["out_folder"], 'Models', f'{global_config["idf"]}_global_best_model.keras'))
	
	pwrite(f_log, f"Current metric for trial number {trial.number + 1} is {neg_r2:.3f}. Best metric so far is {best_neg_r2:.3f}.\n")
	return(neg_r2)


######------------------------------------------------------------------------------------------------------------
t_start = time() # timer start
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--difference', dest = 'd', type = str, help = 'file with ID and tail-length difference as target values')
parser.add_argument('-i', '--initial', dest = 'i', type = str, help = 'file with ID and initial tail length')
parser.add_argument('-u', '--utr', dest = 'u', type = str, help = 'Fasta file for 3\'-UTR sequences')
parser.add_argument('-c', '--cds', dest = 'c', nargs = '?', const = True, help = 'Fasta file for CDS sequences')
parser.add_argument('-f', '--fold', dest = 'f', type = str, help = 'file with base-paring probability predicted by RNAfold')
parser.add_argument('-md', '--model', dest = 'md', type = str, help = 'either a folder of the model files or a model file (keras format) for prediction')
parser.add_argument('-n', '--network', dest = 'n', type = str, default = 'inn', help = 'Network for the model')
parser.add_argument('-p', '--parameters', dest = 'p', type = str, help = 'yaml file for updating parameters')
parser.add_argument('-o', '--outfolder', dest = 'o', type = str, help = 'folder for output files')
parser.add_argument('-l', '--length', dest = 'l', type = str, help = 'maximal length of sequence')
parser.add_argument('-x', '--x', dest = 'x', type = str, help = 'file name (.npy) for X sparse matrix (features)')
parser.add_argument('-y', '--y', dest = 'y', type = str, help = 'file name (.csv) for Y (target values and associated labels)')
parser.add_argument('-m', '--mode', dest = 'm', type = str, default = 'test', help = 'mode to run this script, can be "cv", "opti", "test", or "predict"')
parser.add_argument('-op', '--hyperparameters', dest = 'op', type = int, default = 100, help = 'Number of hyperparameter combinations')
args = parser.parse_args()

###---------------------------------------------------------------------------------------------------------
# This part update parameters 

# update parameters from the input yaml file
if args.p:
	with open(args.p, 'r') as f:
		config = yaml.safe_load(f) or {}

		# update global parameters
		update_dict_from_dict(global_config, config.get('global_config', {}))
		
		if args.l:
			global_config['len_max'] = int(args.l)

		if args.o:
			global_config['out_folder'] = args.o
			
		# update nn parameters
		update_dict_from_dict(inn_params, config.get('inn_params', {}))
		
		# update parameters for optimization 
		if args.m.startswith('op'):
			for key, values in config.get('inn_params_opti', {}).items():
				if key in inn_params_opti:
					if key in ['l1_reg', 'l2_reg', 'learning_rate']:
						inn_params_opti[key]['min'] = values['min']
						inn_params_opti[key]['max'] = values['max']
					else:
						inn_params_opti[key] = values

# make output folders
if global_config['out_folder'] is None:
	global_config['out_folder'] = 'Output'
os.makedirs(global_config['out_folder'], exist_ok=True)
for sub_folder in ['Data', 'Models', 'Loss_plots', 'Scatter_plots', 'Predictions']:
	os.makedirs(os.path.join(global_config['out_folder'], sub_folder), exist_ok=True)
	
# make a log file
# also construct a prefix for output files based on different input files (only if 'idf' not specified in the yaml file)
if args.x and args.y:
	if global_config['idf'] is None:
		global_config['idf'] = args.x.split('X_matrix_')[-1].split('.')[0]
	f_log = make_log_file(os.path.join(global_config['out_folder'], global_config['idf'] + '_simple_log.txt'), p_vars = vars(args))
	pwrite(f_log, 'Note: both X and Y values are provided, so the sequence conversion step will be skipped.')
else:
	if args.d is None:
		if args.u is None:
			parser.error('Input file with sequences is required!')
		else:
			if global_config['idf'] is None:
				global_config['idf'] = args.u.split('/')[-1].split('.')[0]
	else:
		if global_config['idf'] is None:
			global_config['idf'] = args.d.split('/')[-1].split('.')[0]
	f_log = make_log_file(os.path.join(global_config['out_folder'], global_config['idf'] + '_complete_log.txt'), p_vars = vars(args))


# output updated parameters in the log file
pwrite(f_log, '\n' + ('').join(['-']*100))
if args.p:
	pwrite(f_log, 'Parameters have been updated with the yaml file.')
else:
	pwrite(f_log, 'No yaml file is not provided. Default parametered will be used.')
	
pwrite(f_log, '\nGlobal parameters:')
for param in global_config:
	pwrite(f_log, f'\t{param}: {global_config[param]}')
pwrite(f_log, '\nNeural Network parameters:')
for param in inn_params:
	pwrite(f_log, f'\t{param}: {inn_params[param]}')
if args.m.startswith('op'):
	pwrite(f_log, '\nOptimization parameters:')
	for param in inn_params_opti:
		pwrite(f_log, f'\t{param} = {inn_params_opti[param]}')

###---------------------------------------------------------------------------------------------------------
# Set available GPUs
pwrite(f_log, '\n' + ('').join(['-']*100))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
	pwrite(f_log, f'Detected {len(gpus)} GPU cores.')
	n_jobs = len(gpus)
else:
	pwrite(f_log, f'No GPU core is detected. Use CPU for this session. ')
	n_jobs = 1

###---------------------------------------------------------------------------------------------------------
# in prediction mode, load model and update 'len_max' in global_config before data preparation
if args.m == 'predict':
	# Load the pre-trained model
	pwrite(f_log, '\n' + ('').join(['-']*100))
	pwrite(f_log, 'Predict with pre-trained models:')
	pwrite(f_log, 'Input model folder/file:\n' + args.md)
	models = inn_utils.load_keras_models(args.md) 
	pwrite(f_log, f'Number of models loaded: {len(models)}\n')
	models[0].summary(print_fn = lambda x: pwrite(f_log, x))
	model_input_shape = models[0].inputs[0].shape
	pwrite(f_log, f'Model\'s input shape: {model_input_shape}\n')
	if model_input_shape[1] != global_config['len_max']:
		pwrite(f_log, 'Length of the input sequence (' + str(global_config['len_max']) + ') is different from the model input (' + str(model_input_shape[1]) + ')!')
		update_and_log(global_config, 'len_max', model_input_shape[1], f_log)

###---------------------------------------------------------------------------------------------------------
# This part prepares data 
pwrite(f_log, '\n' + ('').join(['-']*100))

if args.x is None or args.y is None: # if pre-converted data not provided
	pwrite(f_log, 'Prepare data for analysis...')
	# make a dictionary of target values (required input in this case)
	y_dict = {}
	if args.d:
		pwrite(f_log, '\nInput file with target values:\n' + args.d)
		pwrite(f_log, '\nMake a dictionary of target values...' + timer(t_start))
		with open(args.d, 'r') as f:
			if global_config['flag_input_header']:
				f.readline()
			for i, line in enumerate(f):
				lst = line.strip().split('\t')
				if len(lst) <= 2 or int(lst[2]) >= global_config['input_tag_min']: # the 3rd column is the number of tags
					if lst[0] in y_dict:
						sys.exit('Error! Redundant IDs in the input file: ' + lst[0])
					y_dict[lst[0]] = float(lst[1])
		pwrite(f_log, 'Number of total entries in the input file: ' + str(i+1))
		pwrite(f_log, 'Number of entries included in the downstream analysis: ' + str(len(y_dict)))
	else:
		pwrite(f_log, '\nNo targets values are provided. Likely in prediction mode...' + timer(t_start))

	# if inital tail length tail length is provided, make a dictionary
	if args.i:
		pwrite(f_log, '\nInput file with initial tail lengths:\n' + args.i)
		pwrite(f_log, '\nMake a dictionary...' + timer(t_start))
		itl_dict = {}
		with open(args.i, 'r') as f:
			f.readline() # header
			for line in f:
				lst = line.strip().split('\t')
				if lst[0] in itl_dict:
					sys.exit('Error! Redundant IDs in the file: ' + lst[0])
				itl_dict[lst[0]] = float(lst[1])
		if len(itl_dict) > 0:
			global_config['flag_initial_tl'] = True
			inn_params['flag_initial_tl'] = True
		pwrite(f_log, 'Number of entries in the file with initial tail lengths: ' + str(len(itl_dict)))

	# if '-c' is present but the the CDS sequence file is not provided, a CDS annotation track is added to the input sequences arrays.
	# if the CDS sequence file is not provided, make a dictionary 
	if args.c:
		if args.c is True:
			pwrite(f_log, '\nNo CDS sequences are provided, but the CDS annotation track will be added.' + timer(t_start))
		else:
			pwrite(f_log, '\nInput CDS file:\n' + args.c)
			pwrite(f_log, '\nMake a dictionary of CDS sequences...' + timer(t_start))
			cds_dict = fasta_to_dict(args.c, key_parser_sep = '__', key_parser_pos = 0)
			pwrite(f_log, 'Number of entries in the fasta file: ' + str(len(cds_dict)))
	else:
		pwrite(f_log, '\nNo CDS sequences are provided. Use only 3\'-UTR sequences...' + timer(t_start))

	# if RNA fold file is provided, make a dictionary 
	if args.f:
		pwrite(f_log, '\nInput RNA fold data:\n' + args.f)

		if global_config['n_dimension'] == 3:
			pwrite(f_log, '\nLoad RNA fold data...' + timer(t_start))
			fold_h5 = h5py.File(args.f)
		else:
			pwrite(f_log, '\nMake a dictionary of RNA fold data...' + timer(t_start))
			fold_dict = {}
			with gzip.open(args.f, 'rt') as f:
				while(True):
					line = f.readline()
					if line:
						sele_id = fasta_id_parser(line, sep = '__')[0]
						f.readline()
						pp_lst = f.readline().strip().split('\t')
						if sele_id not in fold_dict:
							fold_dict[sele_id] = pp_lst
						else:
							sys.exit('Error! Redundant IDs in the RNA fold data file: ' + sele_id)
					else:
						break
	else:
		pwrite(f_log, '\nNo RNA fold data is provided.' + timer(t_start))

	# convert sequence to 2D or 3D matrix
	pwrite(f_log, '\nRead in the fasta sequence file...')
	with open(args.u, 'r') as f:
		line_lst = []
		counting = 0
		n_drop_fa_string_filter = 0
		n_drop_short_utr = 0
		n_drop_unknown_nt = 0
		n_utr_w_cds = 0
		X_mat_lst = []
		Y_df_lst = []
		temp_seq = ''
		while(True):
			line = f.readline()
			if not line:
				line_lst.append([current_id_line, temp_seq]) # the last entry
				with concurrent.futures.ProcessPoolExecutor(min(global_config['n_threads'], len(line_lst))) as pool:
					futures = pool.map(process_seqs, np.array_split(line_lst, min(global_config['n_threads'], len(line_lst))))
					for future in futures:
						if future is not None:
							ary, df, n1, n2, n3, n4 = future
							X_mat_lst.append(ary)
							Y_df_lst.append(df)
							n_drop_fa_string_filter += n1
							n_drop_short_utr += n2
							n_drop_unknown_nt += n3
							n_utr_w_cds += n4
					pwrite(f_log, str(counting) + ' sequences processed...' + timer(t_start))
				break
			else:
				if line[0] == '>':
					counting += 1
					if temp_seq != '':
						line_lst.append([current_id_line, temp_seq])
						temp_seq = ''
					current_id_line = line
					if counting % global_config['chunk'] == 0:
						with concurrent.futures.ProcessPoolExecutor(global_config['n_threads']) as pool:
							futures = pool.map(process_seqs, np.array_split(line_lst, global_config['n_threads']))
							for future in futures:
								if future is not None:
									ary, df, n1, n2, n3, n4 = future
									X_mat_lst.append(ary)
									Y_df_lst.append(df)
									n_drop_fa_string_filter += n1
									n_drop_short_utr += n2
									n_drop_unknown_nt += n3
									n_utr_w_cds += n4
							pwrite(f_log, str(counting) + ' sequences processed...' + timer(t_start))
						line_lst = []
				else:
					temp_seq += line.strip()
	
	pwrite(f_log, 'Total number of sequences in the input file: ' + str(counting))
	pwrite(f_log, 'Number of sequences removed because sequences in the fasta file is uncertain: ' + str(n_drop_fa_string_filter))
	pwrite(f_log, 'Number of sequences removed because 3\'-UTR seqeunces shorter than ' + str(global_config['len_min']) +': ' + str(n_drop_short_utr))
	pwrite(f_log, 'Number of sequences removed because of un-recogenized characters in the sequence: ' + str(n_drop_unknown_nt))
	if args.c:
		pwrite(f_log, 'Number of 3\'-UTRs with CDS sequences added to the 5\' end: ' + str(n_utr_w_cds))
	
	X_ary = np.vstack(X_mat_lst)
	Y_df = pd.concat(Y_df_lst, ignore_index = True)
	pwrite(f_log, 'Number of sequences remained after filtering and encoding: ' + str(X_ary.shape[0]))
	
	np.save(os.path.join(global_config['out_folder'], 'Data', 'temp_X_matrix_' + global_config['idf'] + '.npy'), X_ary)
	Y_df.to_csv(os.path.join(global_config['out_folder'], 'Data', 'temp_Y_df_' + global_config['idf'] + '.csv'))

else:
	X_ary = np.load(args.x)
	Y_df = pd.read_csv(args.y)
	# update parameters based on input data
	if 'itl' in Y_df.columns:
		update_and_log(global_config, 'flag_initial_tl', True, f_log)
		update_and_log(inn_params, 'flag_initial_tl', True, f_log)

# get the statitics of the count for each class (by label)
pwrite(f_log, '\nNumber of sequences by CPE count (for stratification):')
label_count_df = Y_df['label'].value_counts().sort_index()
for i in range(label_count_df.shape[0]):
	pwrite(f_log, f'{label_count_df.index[i]}: {label_count_df.iloc[i]}')

# check dimensionality
pwrite(f_log, '\nData matrix dimension, X: ' + str(X_ary.shape) + ' Y: ' + str(Y_df.shape))
if Y_df.shape[0] != X_ary.shape[0]:
	pwrite(f_log, 'Warining! Dimension of input matrices and the number of target values don\'t match!')

# update input shape
update_and_log(inn_params, 'input_shape', X_ary.shape[1:], f_log)

###---------------------------------------------------------------------------------------------------------
# check skewness of target Y values
pwrite(f_log, '\n' + ('').join(['-']*100))
if len(Y_df) != 0 and "NA" not in Y_df['y'].to_numpy():
	if np.min(Y_df['y']) <= 0:
		global_config['y_offset'] = abs(math.floor(np.min(Y_df['y']))) + 1
		pwrite(f_log, "Target Y contains negative values. An offset of " + str(global_config['y_offset']) + " will be used if log, sqrt, or boxcox transformation is applied.")
	else:
		global_config['y_offset'] = 0
	pwrite(f_log, "Check the overall skewness of target values if the following tranformation is applied: ")
	for tm in ['none', 'log', 'sqrt', 'box-cox', 'yeo-johnson']:
		val_skew = stats.skew(target_transform(Y_df['y'].to_numpy(), method = tm, offset = global_config['y_offset'], constant = global_config['tl_cons'])[0])
		pwrite(f_log, tm + ': %.3f' % val_skew)

	pwrite(f_log, 'Type of transformation of target values: ' + global_config['y_mode'])
else:
	pwrite(f_log, 'Prediction mode, skewness not checked.')

###---------------------------------------------------------------------------------------------------------
# This section performs machine learning with neural networks
pwrite(f_log, '\n' + ('').join(['-']*100) + '\n')

# Mode 1 (args.m == 'op') for optimization of hyperparameters
if args.m.startswith('op'):
	pwrite(f_log, '\nPerforming hyperparameter search...'+ timer(t_start))
	pwrite(f_log, 'Ratio between testing and training sets: ' + str(global_config['test_size']))
	global_config['flag_print_model'] = False
	os.makedirs(os.path.join(global_config['out_folder'], 'Optimization'), exist_ok=True)

	# split data into two groups for training and validation
	X_train, X_val, Y_train_df, Y_val_df = train_test_split(X_ary, Y_df, test_size = global_config['test_size'], random_state = 57, shuffle = True, stratify = Y_df['label'])
	Y_train = Y_train_df['y'].to_numpy()
	Y_val = Y_val_df['y'].to_numpy()
	if global_config['flag_initial_tl']:
		X_train = [X_train, Y_train_df['itl'].to_numpy()]
		X_val= [X_val, Y_val_df['itl'].to_numpy()]

	# transform target values
	Y_train_trans, tf_fitted = target_transform(Y_train, method = global_config['y_mode'], offset = global_config['y_offset'], constant = global_config['tl_cons'])
	Y_val_trans = target_transform(Y_val, method = global_config['y_mode'], offset = global_config['y_offset'], transformer = tf_fitted, constant = global_config['tl_cons'])[0]

	# optimize with objective function
	best_neg_r2 = 0
	f_opti = os.path.join(global_config["out_folder"], 'Optimization', f'{global_config["idf"]}_hyperparameter_search_log.txt')
	study_storage = 'sqlite:///' + os.path.join(global_config["out_folder"], 'Optimization', f'{global_config["idf"]}_hyperparameter_search_optuna_study.db')
	
	try:
		optuna.delete_study(study_name = 'PAL-AI_optuna', storage = study_storage)
		pwrite(f_log, '\nAn optuna study with the same name exists. This old one will be overwritten by the new study.')
	except KeyError:
		pwrite(f_log, '\nA new optuna study is created for optimization.')

	study = optuna.create_study(
		study_name = 'PAL-AI_optuna',
		direction = 'minimize', 
		sampler = optuna.samplers.TPESampler(),
		storage = study_storage
		)
	
	study.optimize(
		lambda trial: optuna_objective(trial, x_train = X_train, y_train = Y_train_trans, x_val = X_val, y_val = Y_val_trans, 
			params = inn_params_opti, model_type = args.n, gpus = gpus, opti_log = f_opti), 
		n_trials = args.op, n_jobs = n_jobs, gc_after_trial = True
		)

	# save optimization results
	save_opti_results(study = study, fn = os.path.join(global_config["out_folder"], 'Optimization', f'{global_config["idf"]}_hyperparameter_search'))


# Mode 2 (args.m == 'cv' or 'test') for cross validation or test (one round of CV)
elif args.m == 'cv' or args.m == 'test':
	# for training and testing with defined hyperparameters
	pwrite(f_log, '\nTraining and testing in a CV fold of ' + str(int(1/global_config['test_size'])) + '...' + timer(t_start))	
	
	# split data into k-fold, stratified by the 'label' in the Y dataframe (whether the input sequence contains a CPE)
	sfk_split = StratifiedKFold(n_splits = int(1/global_config['test_size']), shuffle = True, random_state = 57).split(X_ary, Y_df['label'])

	# prediction results
	out_pred = os.path.join(global_config['out_folder'], 'Predictions', global_config['idf'] + '_train_test_CV_prediction_results.txt')

	with open(out_pred, 'w') as f_res, open(os.path.join(global_config['out_folder'], global_config['idf'] + '_train_test_CV_stats.txt'), 'w') as f_stat:
		# output files for the statistics and predictions
		f_stat.write('cv_group\trep\tr2\tr_pearson\tr_spearman\tloss\tmetric\tepochs\n')
		f_res.write('cv_group\tid\ty\ty_pred\tlabel\n')
		
		for i, (train_val_idx, test_idx) in enumerate(sfk_split):
			X_train_val = X_ary[train_val_idx]
			X_test = X_ary[test_idx]
			Y_train_val_df = Y_df.iloc[train_val_idx]
			Y_test_df = Y_df.iloc[test_idx]
			
			# get target values 
			Y_test = Y_test_df['y'].to_numpy()

			r2_lst = [] # a list to save r2 values for all reps
			for j in range(global_config['n_rep']):
				if global_config['flag_initial_tl']:
					X_train_val = [X_train_val, Y_df.iloc[train_val_idx]['itl'].to_numpy()]
					X_test = [X_test, Y_df.iloc[test_idx]['itl'].to_numpy()]

				# train a model for this CV and rep
				pwrite(f_log, f"\nTraining with CV group {i+1} out of {int(1/global_config['test_size'])}, replicate {j+1} out of {global_config['n_rep']}" + timer(t_start))
				
				# split the training data into two groups for training and validation
				X_train, X_val, Y_train_df, Y_val_df = train_test_split(X_train_val, Y_train_val_df, test_size = global_config['test_size'], random_state = i*10+j, shuffle = True, stratify = Y_train_val_df['label'])
				
				# get target values for each group
				Y_train = Y_train_df['y'].to_numpy()
				Y_val = Y_val_df['y'].to_numpy()

				# transform target values
				Y_train_trans, tf_fitted = target_transform(Y_train, method = global_config['y_mode'], offset = global_config['y_offset'], constant = global_config['tl_cons'])
				Y_val_trans = target_transform(Y_val, method = global_config['y_mode'], offset = global_config['y_offset'], transformer = tf_fitted, constant = global_config['tl_cons'])[0]
				Y_test_trans = target_transform(Y_test, method = global_config['y_mode'], offset = global_config['y_offset'], transformer = tf_fitted, constant = global_config['tl_cons'])[0]

				# fit the model
				history, model, model_params = model_nn(X_train, Y_train_trans, X_val, Y_val_trans, inn_params, model_type = args.n, best_model = True, early_stop = True, gpu_id = 0 if gpus else None)
						
				# save the model
				model.save(os.path.join(global_config['out_folder'], 'Models', global_config['idf'] + '_best_model_CV_group_'+ str(i+1) + '_rep_' + str(j+1) + '.keras'))
				
				# predict the test set and evaluate the model:
				scores = model.evaluate(X_test, Y_test_trans, batch_size = model_params['batch_size'], verbose = global_config['verbose'])
				Y_pred_trans = model.predict(X_test, verbose = global_config['verbose']).ravel()
				Y_pred = target_transform(Y_pred_trans, method = global_config['y_mode'], offset = global_config['y_offset'], inverse = True, transformer = tf_fitted, constant = global_config['tl_cons'])[0]
				r_pearson = stats.pearsonr(Y_test, Y_pred).statistic
				r_spearmam = stats.spearmanr(Y_test, Y_pred).statistic
				r2_lst.append(r_pearson ** 2)
				pwrite(f_log, f'R squared: {r_pearson ** 2: .2f} (Rp = {r_pearson: .3f}, Rs = {r_spearmam: .3f})\n')

				# output stats
				f_stat.write(
					f'{i+1}\t{j+1}\t{r_pearson ** 2: .2f}\t{r_pearson: .3f}\t{r_spearmam: .3f}\t' + 
					'\t'.join(list(map(str, scores))) + 
					f'\t{len(history.history["loss"])}/{model_params["epochs"]}\n')
				
				# make train-test loss line plot
				loss_plot(train = history.history['loss'], test = history.history['val_loss'], 
					fn = os.path.join(global_config['out_folder'], 'Loss_plots', global_config['idf'] + '_CV_group_' + str(i+1) + '_rep_' + str(j+1)), y_label = 'Loss')
				loss_plot(train = history.history[model_params['metrics']], test = history.history[f"val_{model_params['metrics']}"], 
					fn = os.path.join(global_config['out_folder'], 'Loss_plots', global_config['idf'] + '_CV_group_' + str(i+1) + '_rep_' + str(j+1)), y_label = 'Metric')

				if args.m != 'cv': # only do this once if not in 'cv' mode
					for k in range(len(Y_test)):
						f_res.write('\t'.join(list(map(str, [i+1, Y_df.iloc[test_idx]['id'].to_numpy()[k], Y_test[k], Y_pred[k], Y_df.iloc[test_idx]['label'].to_numpy()[k]]))) + '\n')
					break
				
				# combine predicted values from each rep for each cv split
				if j == 0:
					Y_pred_cv = np.array(Y_pred)
				else:
					Y_pred_cv = np.vstack((Y_pred_cv, np.array(Y_pred)))

			if args.m != 'cv': # only do this once if not in 'cv' mode
				break
				
			# average predicted values among all reps for each cv split
			Y_pred_avg = np.mean(Y_pred_cv, axis = 0)
			
			# output preditions for each cv split 
			for k in range(len(Y_test)):
				f_res.write('\t'.join(list(map(str, [i+1, Y_df.iloc[test_idx]['id'].to_numpy()[k], Y_test[k], Y_pred_avg[k], Y_df.iloc[test_idx]['label'].to_numpy()[k]]))) + '\n')

			# make directory to save the best model for each split
			os.makedirs(os.path.join(global_config['out_folder'], 'Models', 'Best_models', f'CV_{i+1}'), exist_ok=True)

			# copy the best model (by r2) to the "Best_models" folder
			j_best = np.argmax(r2_lst)
			subprocess.call(
				['cp', 
				  os.path.join(global_config['out_folder'], 'Models', global_config['idf'] + '_best_model_CV_group_'+ str(i+1) + '_rep_' + str(j_best+1) + '.keras'),
				  os.path.join(global_config['out_folder'], 'Models', 'Best_models', f'CV_{i+1}', global_config['idf'] + '_best_model_CV_group_'+ str(i+1) + '_rep_' + str(j_best+1) + '.keras')])

else: # args.m == 'predict'
	pwrite(f_log, '\nPredict with pre-trained models:')
	
	# Predict with pre-trained INN
	for i in range(len(models)):
		if global_config['flag_initial_tl']:
			Y_pred_trans = models[i].predict([X_ary,Y_df['itl'].to_numpy()], verbose = global_config['verbose']).ravel()
		else:
			Y_pred_trans = models[i].predict(X_ary, verbose = global_config['verbose']).ravel()
		if i == 0:
			Y_pred_trans_ary = np.array(Y_pred_trans)
		else:
			Y_pred_trans_ary = np.vstack((Y_pred_trans_ary, np.array(Y_pred_trans)))
	
	# write out predictions
	out_pred = os.path.join(global_config['out_folder'], 'Predictions', global_config['idf'] + '_prediction_results.txt')
	out_pred_all = os.path.join(global_config['out_folder'], 'Predictions', global_config['idf'] + '_prediction_all_models.txt')

	# Take the average of the predictions from all models
	Y_pred_trans = np.mean(Y_pred_trans_ary, axis = 0)

	if 'NA' not in Y_df['y'].to_numpy():
		Y_pred = target_transform(Y_pred_trans, method = global_config['y_mode'], offset = global_config['y_offset'], inverse = True, constant = global_config['tl_cons'])[0]
		Y_trans = target_transform(Y_df['y'], method = global_config['y_mode'], offset = global_config['y_offset'], constant = global_config['tl_cons'])[0]
		r_value = stats.linregress(Y_df['y'].to_numpy(), Y_pred).rvalue
		pwrite(f_log, 'R-squared between the measured and the model-predicted: %.3f' % (r_value ** 2) + '\n')

		with open(out_pred, 'w') as f_res:
			f_res.write('id\ty\ty_pred\ty_trans\ty_trans_pred\tlabel\n')
			for k in range(Y_df.shape[0]):
				f_res.write('\t'.join([Y_df['id'].to_numpy()[k], f"{Y_df['y'].to_numpy()[k]}", f"{Y_pred[k]:.3f}", 
					f"{Y_trans[k]:.3f}", f"{Y_pred_trans[k]:.3f}", f"{Y_df['label'].to_numpy()[k]}"]) + '\n')

		with open(out_pred_all, 'w') as f:
			f.write('id\ty\t' + '\t'.join([f'model_{x}' for x in range(len(models))]) + '\n')
			for k in range(Y_df.shape[0]):
				f.write('\t'.join([Y_df['id'].to_numpy()[k], f"{Y_df['y'].to_numpy()[k]}"]) + '\t' + 
					'\t'.join([f'{x:.3f}' for x in Y_pred_trans_ary[:,k]]) + '\n')

	else:
		with open(out_pred, 'w') as f_res:
			f_res.write('id\ty_pred\tlabel\n')
			for k in range(Y_df.shape[0]):
				f_res.write('\t'.join([Y_df['id'].to_numpy()[k], f"{Y_pred_trans[k]:.3f}", f"{Y_df['label'].to_numpy()[k]}"]) + '\n')

		with open(out_pred_all, 'w') as f:
			f.write('id\t' + '\t'.join([f'model_{x}' for x in range(len(models))]) + '\n')
			for k in range(Y_df.shape[0]):
				f.write(str(Y_df['id'].to_numpy()[k]) + '\t' + '\t'.join([f'{x:.3f}' for x in Y_pred_trans_ary[:,k]]) + '\n')


####---------------------------------------------------------------------------------------------------------
# Make a plot for comparing measured and predicted values 
if 'out_pred' in globals() and os.path.exists(out_pred):
	out_df = pd.read_csv(out_pred, sep = '\t')
	if 'y' in out_df.columns and 'y_pred' in out_df.columns:
		scatter_plot(out_df['y'], out_df['y_pred'], fn = os.path.join(global_config['out_folder'], 'Scatter_plots', global_config['idf'] + '_final_scatter_plot.png'))

pwrite(f_log, 'Finished...' + timer(t_start))
f_log.close()

