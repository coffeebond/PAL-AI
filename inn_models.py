import math
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from collections import OrderedDict

def custom_loss(y_true, y_pred):
	loss_w = tf.where(K.abs(y_true - K.mean(y_true)) < 3 * K.std(y_true), 0.01, 1)
	loss = K.mean(loss_w * K.square(y_true - y_pred))
	#loss = K.mean((K.abs(y_true - tl_mu) + 1) * K.square(y_true - y_pred))
	return(loss)

def custom_metric(y_true, y_pred):
	y_true_mean = K.mean(y_true, axis=0)
	y_pred_mean = K.mean(y_pred, axis=0)
	r_numerator = K.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
	x_square_sum = K.sum((y_true - y_true_mean) * (y_true - y_true_mean))
	y_square_sum = K.sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
	r_denominator = K.sqrt(x_square_sum * y_square_sum)
	r = r_numerator / r_denominator
	metric_out = -(K.mean(r) ** 2) 
	'''
	sele = K.abs(y_true - K.mean(y_true)) > 3 * K.std(y_true)
	sele = K.cast(sele, dtype = 'float32')
	metric_out = K.sum(K.abs(y_true * sele - y_pred * sele)) / K.sum(sele)
	'''
	return(metric_out)
	
def resnet_block_v1(inputs, filters, kernel_size, dilation_rate = 1, strides = 1, kernel_regularizer = None, activation_func = 'selu', dropout_rate = 0, trim = 'left', group_idx = 1, block_idx = 1):
	if trim is False:
		padding = 'same'
	else:
		padding = 'valid'
	name_prefix = 'resnet_v1_group_' + str(group_idx) + '_block_' + str(block_idx)
	res = inputs
	res = layers.Conv1D(filters = filters, kernel_size = kernel_size, kernel_regularizer = kernel_regularizer, dilation_rate = dilation_rate, padding = padding, name = name_prefix + '_conv1D_1')(res)
	res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_1')(res)
	#res = layers.LayerNormalization(name = name_prefix + '_layer_nor_1')(res)
	res = layers.BatchNormalization(name = name_prefix + '_batch_nor_1')(res)
	res = layers.Activation(activation_func, name = name_prefix + 'activate_1')(res)
	res = layers.Conv1D(filters = filters, kernel_size = kernel_size, kernel_regularizer = kernel_regularizer, dilation_rate = dilation_rate, padding = padding, name = name_prefix + '_conv1D_2')(res)
	res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_2')(res)
	#res = layers.LayerNormalization(name = name_prefix + '_layer_nor_2')(res)
	res = layers.BatchNormalization(name = name_prefix + '_batch_nor_2')(res)

	# crop the inputs so inputs are the same dimension as the output
	if trim == 'left': 
		inputs = layers.Cropping1D(cropping = ((kernel_size-1)*dilation_rate*2,0))(inputs)
	elif trim == 'right':
		inputs = layers.Cropping1D(cropping = (0, (kernel_size-1)*dilation_rate*2))(inputs)
	elif trim == 'both':
		inputs = layers.Cropping1D(cropping = (math.ceiling((kernel_size-1)*dilation_rate)*2, math.floor((kernel_size-1)*dilation_rate*2)))

	res = layers.Add(name = name_prefix + '_skip_1')([inputs, res])
	res = layers.Activation(activation_func, name = name_prefix + '_activate_2')(res)
	return(res)

def resnet_block_v2(inputs, filters, kernel_size, dilation_rate = 1, strides = 1, kernel_regularizer = None, activation_func = 'selu', dropout_rate = 0, trim = 'left', group_idx = 1, block_idx = 1):
	if trim is False:
		padding = 'same'
	else:
		padding = 'valid'
	name_prefix = 'resnet_v2_group_' + str(group_idx) + '_block_' + str(block_idx)
	res = inputs
	res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_1')(res)
	res = layers.LayerNormalization(name = name_prefix + '_layer_nor_1')(res)
	res = layers.Activation(activation_func, name = name_prefix + 'activate_1')(res)
	res = layers.Conv1D(filters = filters, kernel_size = kernel_size, kernel_regularizer = kernel_regularizer, dilation_rate = dilation_rate, padding = padding, name = name_prefix + '_conv1D_1')(res)
	res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_2')(res)
	res = layers.LayerNormalization(name = name_prefix + '_layer_nor_2')(res)
	res = layers.Conv1D(filters = filters, kernel_size = kernel_size, kernel_regularizer = kernel_regularizer, dilation_rate = dilation_rate, padding = padding, name = name_prefix + '_conv1D_2')(res)

	# crop the inputs so inputs are the same dimension as the output
	if trim == 'left': 
		inputs = layers.Cropping1D(cropping = ((kernel_size-1)*dilation_rate*2,0))(inputs)
	elif trim == 'right':
		inputs = layers.Cropping1D(cropping = (0, (kernel_size-1)*dilation_rate*2))(inputs)
	elif trim == 'both':
		inputs = layers.Cropping1D(cropping = (math.ceiling((kernel_size-1)*dilation_rate*2), math.floor((kernel_size-1)*dilation_rate*2)))(inputs)

	res = layers.Add()([inputs, res])
	return(res)

def resnet_group(inputs, filters, kernel_size, dilation_rate = 1, strides = 1, kernel_regularizer = None, activation_func = 'selu', dropout_rate = 0, trim = 'left', version = 2, n_block = 4, group_idx = 1):
	x = inputs
	for i in range(n_block):
		if version == 1:
			x = resnet_block_v1(inputs = x, filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, strides = strides, kernel_regularizer = kernel_regularizer, 
				activation_func = activation_func, dropout_rate = dropout_rate, trim = trim, group_idx = group_idx, block_idx = i + 1)
		else:
			x = resnet_block_v2(inputs = x, filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, strides = strides, kernel_regularizer = kernel_regularizer, 
				activation_func = activation_func, dropout_rate = dropout_rate, trim = trim, group_idx = group_idx, block_idx = i + 1)
	return(x)

def resnet_model(input_params):
	params = OrderedDict([
		('input_shape', (1056,5)),
		('n_Conv1D_filter', 32),
		('n_Conv1D_kernel', 5),
		('kernel_regularizer', None), 
		('activation_func', 'selu'), 
		('dropout_rate', 0), 
		('resnet_trim', 'left'), # whether to trim input to the ResNet for addition
		('resnet_version', 2), 
		('resnet_n_group', 7), # number of ResNet groups
		('resnet_n_block', 4), # number of ResNet blocks per group
		('dilation_rate_lst', [1,2,4,8,4,2,1]),  
		('rnn_func', 'GRU'), 
		('n_rnn_units', 32), 
		('n_dense_neuron', 64),
		('optimizer', 'adam'), 
		('l1_reg', 0.001),
		('l2_reg', 0.001),
		('learning_rate', 0.001), 
		('loss', 'mse'), 
		('metrics', 'mae')
	])
	
	# update network parameters
	for key in input_params:
		if key in params:
			params[key] = input_params[key]

	# input layer
	inn_input = layers.Input(shape = params['input_shape'])

	# first convolution layer
	res = layers.Conv1D(filters = params['n_Conv1D_filter'], kernel_size = params['n_Conv1D_kernel'], kernel_regularizer = params['kernel_regularizer'], name = 'initial_conv1d')(inn_input)
	res = layers.LayerNormalization(name = 'initial_layer_nor')(res)
	res = layers.Activation(params['activation_func'], name = 'initial_activate')(res)
	
	# resnet layers
	assert params['resnet_n_group'] == len(params['dilation_rate_lst'])
	for i in range(params['resnet_n_group']):
		res = resnet_group(inputs = res, filters = params['n_Conv1D_filter'], kernel_size = params['n_Conv1D_kernel'], dilation_rate = params['dilation_rate_lst'][i], 
			kernel_regularizer = params['kernel_regularizer'], activation_func = params['activation_func'], dropout_rate = params['dropout_rate'], 
			trim = params['resnet_trim'], version = params['resnet_version'], n_block = params['resnet_n_block'], group_idx = i + 1)

	# rnn layers
	if params['rnn_func'] == 'LSTM':
		res = layers.LSTM(units = params['n_rnn_units'], go_backwards = False, kernel_regularizer = params['kernel_regularizer'], name = 'LSTM')(res)
		res = layers.BatchNormalization(name = 'LSTM_batch_nor')(res)
		res = layers.Activation(params['activation_func'], name = 'LSTM_activate')(res)
	elif params['rnn_func'] == 'GRU':
		res = layers.GRU(units = params['n_rnn_units'], go_backwards = False, kernel_regularizer = params['kernel_regularizer'], name = 'GRU')(res)
		res = layers.BatchNormalization(name = 'GRU_batch_nor')(res)
		res = layers.Activation(params['activation_func'], name = 'GRU_activate')(res)
	elif params['rnn_func'] == 'BiLSTM':
		res = layers.Bidirectional(layers.LSTM(units = params['n_rnn_units'], kernel_regularizer = params['kernel_regularizer']), name = 'BiLSTM')(res)
		res = layers.BatchNormalization(name = 'BiLSTM_batch_nor')(res)
		res = layers.Activation(params['activation_func'], name = 'BiLSTM_activate')(res)
	elif params['rnn_func'] == 'BiGRU':
		res = layers.Bidirectional(layers.GRU(units = params['n_rnn_units'], kernel_regularizer = params['kernel_regularizer']), name = 'BiGRU')(res)
		res = layers.BatchNormalization(name = 'BiGRU_batch_nor')(res)
		res = layers.Activation(params['activation_func'], name = 'BiGRU_activate')(res)
	else:
		res = layers.Flatten(name = 'flatten_no_RNN')(res)

	res = layers.Dense(params['n_dense_neuron'], kernel_regularizer=params['kernel_regularizer'], name = 'dense')(res)
	res = layers.Dropout(params['dropout_rate'], name = 'dropout_dense')(res)
	res = layers.BatchNormalization(name = 'layer_nor_dense')(res)
	res = layers.Activation(params['activation_func'], name = 'activate_dense')(res)
	inn_output = layers.Dense(1, name = 'final_dense')(res)
	
	model = keras.Model(inputs = inn_input, outputs = inn_output)

	if params['optimizer'] == 'adam':
		model.compile(optimizer=keras.optimizers.Adam(learning_rate = params['learning_rate']), loss= params['loss'], metrics=[params['metrics'], custom_metric]) #[params['metrics']])#, 
	else:
		model.compile(optimizer=keras.optimizers.SGD(learning_rate = params['learning_rate'], momentum = 0.9, nesterov = True, decay = 1e-6), loss = params['loss'], metrics=[params['metrics'], custom_metric]) #[params['metrics']])#,
	
	return(model)

def Conv1D_MaxPool_block(inputs, filters, kernel_size, kernel_regularizer = None, dropout_rate = 0, pool_size = 2, activation_func = 'selu'):
	res = layers.Conv1D(filters = filters, kernel_size = kernel_size, kernel_regularizer = kernel_regularizer)(inputs)
	res = layers.Dropout(dropout_rate)(res)
	res = layers.MaxPooling1D(pool_size=pool_size)(res)
	res = layers.LayerNormalization()(res)
	res = layers.Activation(activation_func)(res)	
	return(res)

def Conv2D_MaxPool_block(inputs, filters, kernel_size, kernel_regularizer = None, dropout_rate = 0, pool_size = 2, activation_func = 'selu'):
	res = layers.Conv2D(filters = filters, kernel_size = kernel_size, kernel_regularizer = kernel_regularizer)(inputs)
	res = layers.Dropout(dropout_rate)(res)
	res = layers.MaxPooling2D(pool_size=pool_size)(res)
	res = layers.LayerNormalization()(res)
	res = layers.Activation(activation_func)(res)	
	return(res)

def get_shape_Conv2D_MaxPool_block(l, kernel_size, pool_size, n_block): # calculate the shape of the output 
	if n_block != 0:
		for i in range(n_block):
			l = ((l - kernel_size + 1) - pool_size)//pool_size + 1
	return(l)

def inn_model(input_params):
	params = OrderedDict([
		('input_shape', (1056,5)),
		('flag_initial_tl', False),
		#('flag_mfe', False),
		('n_Conv1D_filter', 32),
		('n_Conv1D_kernel', 5),
		('kernel_regularizer', None), 
		('activation_func', 'selu'), 
		('dropout_rate', 0), 
		('n_Conv1D_MaxPool_block', 1),
		('n_pool_size', 2),
		('rnn_func', 'GRU'), 
		('n_rnn_units', 32), 
		('n_dense_neuron', 64),
		('optimizer', 'adam'), 
		('l1_reg', 0.001),
		('l2_reg', 0.001),
		('learning_rate', 0.001), 
		('loss', 'mse'), 
		('metrics', 'mae')
	])
	
	# update network parameters
	for key in input_params:
		if key in params:
			params[key] = input_params[key]

	# input layer
	inn_input = layers.Input(shape = params['input_shape'])


	# first convolution layer
	res = layers.Conv1D(filters = params['n_Conv1D_filter'], kernel_size = params['n_Conv1D_kernel'], kernel_regularizer = params['kernel_regularizer'])(inn_input)
	res = layers.LayerNormalization()(res)
	res = layers.Activation(params['activation_func'])(res)
	
	if params['n_Conv1D_MaxPool_block'] > 0:
		for i in range(params['n_Conv1D_MaxPool_block']):
			res = Conv1D_MaxPool_block(inputs = res, filters = params['n_Conv1D_filter'], kernel_size = params['n_Conv1D_kernel'], kernel_regularizer = params['kernel_regularizer'], 
				dropout_rate = params['dropout_rate'], pool_size = params['n_pool_size'], activation_func = params['activation_func'])

	if params['rnn_func'] == 'LSTM':
		res = layers.LSTM(units = params['n_rnn_units'], go_backwards = False, kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']))(res)
		res = layers.BatchNormalization()(res)
		res = layers.Activation(params['activation_func'])(res)
	elif params['rnn_func'] == 'GRU':
		res = layers.GRU(units = params['n_rnn_units'], go_backwards = False, kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']))(res)
		res = layers.BatchNormalization()(res)
		res = layers.Activation(params['activation_func'])(res)
	elif params['rnn_func'] == 'BiLSTM':
		res = layers.Bidirectional(layers.LSTM(units = params['n_rnn_units'], kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))(res)
		res = layers.BatchNormalization()(res)
		res = layers.Activation(params['activation_func'])(res)
	elif params['rnn_func'] == 'BiGRU':
		res = layers.Bidirectional(layers.GRU(units = params['n_rnn_units'], kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))(res)
		res = layers.BatchNormalization()(res)
		res = layers.Activation(params['activation_func'])(res)
	else:
		res = layers.Flatten()(res)

	res = layers.Dense(params['n_dense_neuron'], kernel_regularizer=keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']))(res)
	res = layers.Dropout(params['dropout_rate'])(res)
	res = layers.BatchNormalization()(res)
	res = layers.Activation(params['activation_func'])(res)

	if params['flag_initial_tl']:
		itl_input = layers.Input(1)
		res = layers.Concatenate(axis = 1)([res, itl_input])
		inn_output = layers.Dense(1)(res)
		model = keras.Model(inputs = [inn_input, itl_input], outputs = inn_output)
	else:
		inn_output = layers.Dense(1)(res)
		model = keras.Model(inputs = inn_input, outputs = inn_output)

	if params['optimizer'] == 'adam':
		model.compile(optimizer=keras.optimizers.Adam(learning_rate = params['learning_rate']), loss=params['loss'], metrics=[params['metrics'], custom_metric]) #[params['metrics']])#, 
	else:
		model.compile(optimizer=keras.optimizers.SGD(learning_rate = params['learning_rate'], momentum = 0.9, nesterov = True, decay = 1e-6), loss=params['loss'], metrics=[params['metrics'], custom_metric]) #[params['metrics']])#, 

	return(model)

def cinn_model(input_params):
	params = OrderedDict([
		('input_shape', (81,81,16)),
		('flag_initial_tl', False),
		#('flag_mfe', False),
		('n_Conv2D_filter', 32),
		('n_Conv2D_kernel', 5),
		('kernel_regularizer', None), 
		('activation_func', 'selu'), 
		('dropout_rate', 0), 
		('n_Conv2D_MaxPool_block', 1),
		('n_pool_size', 2),
		('rnn_func', 'GRU'), 
		('n_rnn_units', 32), 
		('n_dense_neuron', 64),
		('optimizer', 'adam'), 
		('l1_reg', 0.001),
		('l2_reg', 0.001),
		('learning_rate', 0.001), 
		('loss', 'mse'), 
		('metrics', 'mae')
	])
	
	# update network parameters
	for key in input_params:
		if key in params:
			params[key] = input_params[key]

	# input layer
	inn_input = layers.Input(shape = params['input_shape'])


	# first convolution layer
	res = layers.Conv2D(filters = params['n_Conv2D_filter'], kernel_size = params['n_Conv2D_kernel'], kernel_regularizer = params['kernel_regularizer'])(inn_input)
	res = layers.LayerNormalization()(res)
	res = layers.Activation(params['activation_func'])(res)
	
	if params['n_Conv2D_MaxPool_block'] > 0:
		for i in range(params['n_Conv2D_MaxPool_block']):
			res = Conv2D_MaxPool_block(inputs = res, filters = params['n_Conv2D_filter'], kernel_size = params['n_Conv2D_kernel'], kernel_regularizer = params['kernel_regularizer'], 
				dropout_rate = params['dropout_rate'], pool_size = params['n_pool_size'], activation_func = params['activation_func'])

	# convert to 1D so it can be feed into RNN
	len_d2 = get_shape_Conv2D_MaxPool_block(params['input_shape'][0] - params['n_Conv2D_kernel'] + 1 , params['n_Conv2D_kernel'], params['n_pool_size'], params['n_Conv2D_MaxPool_block'])
	res = layers.Conv2D(filters = params['n_Conv2D_filter'], kernel_size = (params['n_Conv2D_kernel'],len_d2), kernel_regularizer = params['kernel_regularizer'])(res)
	res = layers.LayerNormalization()(res)
	res = layers.Activation(params['activation_func'])(res)
	res = layers.Reshape((len_d2 - params['n_Conv2D_kernel'] + 1, params['n_Conv2D_filter']))(res)

	if params['rnn_func'] == 'LSTM':
		res = layers.LSTM(units = params['n_rnn_units'], go_backwards = False, kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']))(res)
		res = layers.BatchNormalization()(res)
		res = layers.Activation(params['activation_func'])(res)
	elif params['rnn_func'] == 'GRU':
		res = layers.GRU(units = params['n_rnn_units'], go_backwards = False, kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']))(res)
		res = layers.BatchNormalization()(res)
		res = layers.Activation(params['activation_func'])(res)
	elif params['rnn_func'] == 'BiLSTM':
		res = layers.Bidirectional(layers.LSTM(units = params['n_rnn_units'], kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))(res)
		res = layers.BatchNormalization()(res)
		res = layers.Activation(params['activation_func'])(res)
	elif params['rnn_func'] == 'BiGRU':
		res = layers.Bidirectional(layers.GRU(units = params['n_rnn_units'], kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))(res)
		res = layers.BatchNormalization()(res)
		res = layers.Activation(params['activation_func'])(res)
	else:
		res = layers.Flatten()(res)

	res = layers.Dense(params['n_dense_neuron'], kernel_regularizer=keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']))(res)
	res = layers.Dropout(params['dropout_rate'])(res)
	res = layers.BatchNormalization()(res)
	res = layers.Activation(params['activation_func'])(res)

	if params['flag_initial_tl']:
		itl_input = layers.Input(1)
		res = layers.Concatenate(axis = 1)([res, itl_input])
		inn_output = layers.Dense(1)(res)
		model = keras.Model(inputs = [inn_input, itl_input], outputs = inn_output)
	else:
		inn_output = layers.Dense(1)(res)
		model = keras.Model(inputs = inn_input, outputs = inn_output)

	if params['optimizer'] == 'adam':
		model.compile(optimizer=keras.optimizers.Adam(learning_rate = params['learning_rate']), loss=params['loss'], metrics=[params['metrics'], custom_metric]) #[params['metrics']])#, 
	else:
		model.compile(optimizer=keras.optimizers.SGD(learning_rate = params['learning_rate'], momentum = 0.9, nesterov = True, decay = 1e-6), loss=params['loss'], metrics=[params['metrics'], custom_metric]) #[params['metrics']])#, 

	return(model)
