import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import Callback
from collections import OrderedDict

def custom_loss(y_true, y_pred):
	loss_w = tf.where(tf.abs(y_true - tf.reduce_mean(y_true)) < 3 * tf.math.reduce_std(y_true)(y_true), 0.01, 1)
	loss = tf.reduce_mean(loss_w * tf.square(y_true - y_pred))
	#loss = K.mean((K.abs(y_true - tl_mu) + 1) * K.square(y_true - y_pred))
	return(loss)

def custom_metric(y_true, y_pred):
	y_true_mean = tf.reduce_mean(y_true, axis=0) 
	y_pred_mean = tf.reduce_mean(y_pred, axis=0)
	r_numerator = tf.reduce_sum((y_true - y_true_mean) * (y_pred - y_pred_mean), axis=0)  
	x_square_sum = tf.reduce_sum(tf.square(y_true - y_true_mean), axis=0)  
	y_square_sum = tf.reduce_sum(tf.square(y_pred - y_pred_mean), axis=0)  
	r_denominator = tf.sqrt(x_square_sum * y_square_sum) + tf.keras.backend.epsilon()  
	r = r_numerator / r_denominator
	return(-tf.reduce_mean(tf.square(r)))

def resnet_block_v1(inputs, filters, kernel_size, dilation_rate = 1, strides = 1, kernel_regularizer = None, kernel_initializer = 'he_normal', activation_func = 'leaky_relu', dropout_rate = 0, trim = 'left', group_idx = 1, block_idx = 1):
	if trim is False:
		padding = 'same'
	else:
		padding = 'valid'
	name_prefix = 'resnet_v1_group_' + str(group_idx) + '_block_' + str(block_idx)
	res = inputs
	res = layers.Conv1D(
		filters = filters, 
		kernel_size = kernel_size, 
		kernel_regularizer = kernel_regularizer, 
		kernel_initializer = kernel_initializer,
		dilation_rate = dilation_rate, 
		padding = padding, 
		name = name_prefix + '_conv1D_1')(res)
	
	if activation_func == 'selu':
		res = layers.AlphaDropout(dropout_rate, name = name_prefix + '_alphadropout_1')(res)
		res = layers.LayerNormalization(name = name_prefix + '_layer_nor_1')(res)
	elif activation_func == 'silu':
		res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_1')(res)
		res = layers.LayerNormalization(name = name_prefix + '_layer_nor_1')(res)
	else:
		res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_1')(res)
		res = layers.BatchNormalization(name = name_prefix + '_batch_nor_1')(res)
	res = layers.Activation(activation_func, name = name_prefix + 'activate_1')(res)
	
	res = layers.Conv1D(
		filters = filters, 
		kernel_size = kernel_size, 
		kernel_regularizer = kernel_regularizer, 
		kernel_initializer = kernel_initializer,
		dilation_rate = dilation_rate, 
		padding = padding, 
		name = name_prefix + '_conv1D_2')(res)
	if activation_func == 'selu':
		res = layers.AlphaDropout(dropout_rate, name = name_prefix + '_alphadropout_2')(res)
		res = layers.LayerNormalization(name = name_prefix + '_layer_nor_2')(res)
	elif activation_func == 'silu':
		res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_2')(res)
		res = layers.LayerNormalization(name = name_prefix + '_layer_nor_2')(res)
	else:
		res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_2')(res)
		res = layers.BatchNormalization(name = name_prefix + '_batch_nor_2')(res)

	# crop the inputs so inputs are the same dimension as the output
	if trim == 'left': 
		inputs = layers.Cropping1D(cropping = ((kernel_size-1)*dilation_rate*2,0))(inputs)
	elif trim == 'right':
		inputs = layers.Cropping1D(cropping = (0, (kernel_size-1)*dilation_rate*2))(inputs)
	elif trim == 'both':
		inputs = layers.Cropping1D(cropping = (math.ceiling((kernel_size-1)*dilation_rate)*2, math.floor((kernel_size-1)*dilation_rate*2)))

	res = layers.Add(name = name_prefix + '_skip')([inputs, res])
	res = layers.Activation(activation_func, name = name_prefix + '_activate_2')(res)
	return(res)

def resnet_block_v2(inputs, filters, kernel_size, dilation_rate = 1, strides = 1, kernel_regularizer = None, kernel_initializer = 'he_normal', activation_func = 'leaky_relu', dropout_rate = 0, trim = 'left', group_idx = 1, block_idx = 1):
	if trim is False:
		padding = 'same'
	else:
		padding = 'valid'
	name_prefix = 'resnet_v2_group_' + str(group_idx) + '_block_' + str(block_idx)
	res = inputs

	if activation_func == 'selu':
		res = layers.AlphaDropout(dropout_rate, name = name_prefix + '_alphadropout_1')(res)
		res = layers.LayerNormalization(name = name_prefix + '_layer_nor_1')(res)
	elif activation_func == 'silu':
		res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_1')(res)
		res = layers.LayerNormalization(name = name_prefix + '_layer_nor_1')(res)
	else:
		res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_1')(res)
		res = layers.BatchNormalization(name = name_prefix + '_batch_nor_1')(res)
	res = layers.Activation(activation_func, name = name_prefix + 'activate_1')(res)

	res = layers.Conv1D(
		filters = filters, 
		kernel_size = kernel_size, 
		kernel_regularizer = kernel_regularizer, 
		kernel_initializer = kernel_initializer,
		dilation_rate = dilation_rate, 
		padding = padding, 
		name = name_prefix + '_conv1D_1')(res)
	
	if activation_func == 'selu':
		res = layers.AlphaDropout(dropout_rate, name = name_prefix + '_alphadropout_2')(res)
		res = layers.LayerNormalization(name = name_prefix + '_layer_nor_2')(res)
	elif activation_func == 'silu':
		res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_2')(res)
		res = layers.LayerNormalization(name = name_prefix + '_layer_nor_2')(res)
	else:
		res = layers.Dropout(dropout_rate, name = name_prefix + '_dropout_2')(res)
		res = layers.BatchNormalization(name = name_prefix + '_batch_nor_2')(res)
	res = layers.Activation(activation_func, name = name_prefix + 'activate_2')(res)
	
	res = layers.Conv1D(
		filters = filters, 
		kernel_size = kernel_size, 
		kernel_regularizer = kernel_regularizer, 
		kernel_initializer = kernel_initializer,
		dilation_rate = dilation_rate, 
		padding = padding, 
		name = name_prefix + '_conv1D_2')(res)

	# crop the inputs so inputs are the same dimension as the output
	if trim == 'left': 
		inputs = layers.Cropping1D(cropping = ((kernel_size-1)*dilation_rate*2,0))(inputs)
	elif trim == 'right':
		inputs = layers.Cropping1D(cropping = (0, (kernel_size-1)*dilation_rate*2))(inputs)
	elif trim == 'both':
		inputs = layers.Cropping1D(cropping = (math.ceiling((kernel_size-1)*dilation_rate*2), math.floor((kernel_size-1)*dilation_rate*2)))(inputs)

	res = layers.Add(name = name_prefix + '_skip')([inputs, res])
	return(res)

def resnet_group(inputs, filters, kernel_size, dilation_rate = 1, strides = 1, kernel_regularizer = None, kernel_initializer = 'he_normal', activation_func = 'leaky_relu', dropout_rate = 0, trim = 'left', version = 2, n_block = 4, group_idx = 1):
	x = inputs
	for i in range(n_block):
		if version == 1:
			x = resnet_block_v1(
				inputs = x, 
				filters = filters, 
				kernel_size = kernel_size, 
				dilation_rate = dilation_rate, 
				strides = strides, 
				kernel_regularizer = kernel_regularizer, 
				kernel_initializer = kernel_initializer,
				activation_func = activation_func, 
				dropout_rate = dropout_rate, 
				trim = trim, 
				group_idx = group_idx, 
				block_idx = i + 1)
		else:
			x = resnet_block_v2(
				inputs = x, 
				filters = filters, 
				kernel_size = kernel_size, 
				dilation_rate = dilation_rate, 
				strides = strides, 
				kernel_regularizer = kernel_regularizer, 
				kernel_initializer = kernel_initializer,
				activation_func = activation_func, 
				dropout_rate = dropout_rate, 
				trim = trim, 
				group_idx = group_idx, 
				block_idx = i + 1)
	return(x)

def resnet_model(input_params):
	params = OrderedDict([
		('input_shape', (1056,5)),
		('flag_initial_tl', False),
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

	# use the optimal initializers for different activation functions
	if params['activation_func'] == 'silu':
		params['kernel_initializer'] = 'he_normal'
	elif params['activation_func'] == 'selu':
		params['kernel_initializer'] = 'lecun_normal'
	else: # any other input will be restored to the default
		params['activation_func'] == 'leaky_relu'
		params['kernel_initializer'] = 'he_normal'

	# input layer
	inn_input = layers.Input(shape = params['input_shape'])

	# first convolution layer
	res = layers.Conv1D(
		filters = params['n_Conv1D_filter'], 
		kernel_size = params['n_Conv1D_kernel'], 
		kernel_initializer = params['kernel_initializer'],
		kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']),
		name = 'initial_conv1d'
		)(inn_input)
	res = layers.LayerNormalization(name = 'initial_layer_nor')(res)
	res = layers.Activation(params['activation_func'], name = 'initial_activate')(res)
	
	# resnet block
	assert params['resnet_n_group'] == len(params['dilation_rate_lst'])
	for i in range(params['resnet_n_group']):
		res = resnet_group(
			inputs = res, 
			filters = params['n_Conv1D_filter'], 
			kernel_size = params['n_Conv1D_kernel'], 
			dilation_rate = params['dilation_rate_lst'][i], 
			kernel_regularizer = params['kernel_regularizer'], 
			kernel_initializer = params['kernel_initializer'],
			activation_func = params['activation_func'], 
			dropout_rate = params['dropout_rate'], 
			trim = params['resnet_trim'], 
			version = params['resnet_version'], 
			n_block = params['resnet_n_block'], 
			group_idx = i + 1)

	# Recurrent block
	res = rnn_block(
		inputs = res,
		rnn_func = params['rnn_func'], 
		rnn_units = params['n_rnn_units'], 
		kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']), 
		kernel_initializer = 'he_normal', 
		activation_func = 'leaky_relu')

	# Dense block
	res = Dense_block(
		inputs = res,
		n_dense_neuron = params['n_dense_neuron'], 
		dropout_rate = params['dropout_rate'], 
		kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']), 
		kernel_initializer = params['kernel_initializer'], 
		activation_func = params['activation_func'])

	# Final output layer
	if params['flag_initial_tl']:
		itl_input = layers.Input((1,))
		res = layers.Concatenate(axis = 1)([res, itl_input])
		inn_output = layers.Dense(units = 1, name = 'final_dense')(res)
		model = keras.Model(inputs = [inn_input, itl_input], outputs = inn_output)
	else:
		inn_output = layers.Dense(units = 1, name = 'final_dense')(res)
		model = keras.Model(inputs = inn_input, outputs = inn_output)

	if params['optimizer'] == 'adam':
		model.compile(
		optimizer = keras.optimizers.Adam(learning_rate = params['learning_rate']), 
		loss = params['loss'],
		metrics = [params['metrics']]
		#metrics = [params['metrics'], custom_metric]
		)
	else:
		model.compile(
		optimizer=keras.optimizers.SGD(learning_rate = params['learning_rate'], momentum = 0.9, nesterov = True, decay = 1e-6), 
		loss = params['loss'], 
		metrics = [params['metrics']]
		#metrics=[params['metrics'], custom_metric]
		) 
	
	return(model)

def Conv1D_MaxPool_block(inputs, filters, kernel_size, kernel_regularizer = None, kernel_initializer = 'he_normal', dropout_rate = 0, pool_size = 2, activation_func = 'leaky_relu'):
	res = layers.Conv1D(filters = filters, kernel_size = kernel_size, kernel_regularizer = kernel_regularizer, kernel_initializer = kernel_initializer)(inputs)
	res = layers.LayerNormalization()(res)
	res = layers.Activation(activation_func)(res)	
	res = layers.MaxPooling1D(pool_size=pool_size)(res)
	res = layers.Dropout(dropout_rate)(res)
	return(res)

def Conv2D_MaxPool_block(inputs, filters, kernel_size, kernel_regularizer = None, kernel_initializer = 'he_normal', dropout_rate = 0, pool_size = 2, activation_func = 'leaky_relu'):
	res = layers.Conv2D(filters = filters, kernel_size = kernel_size, kernel_regularizer = kernel_regularizer, kernel_initializer = kernel_initializer)(inputs)
	res = layers.LayerNormalization()(res)
	res = layers.Activation(activation_func)(res)
	res = layers.MaxPooling2D(pool_size=pool_size)(res)
	res = layers.Dropout(dropout_rate)(res)
	return(res)

def rnn_block(inputs, rnn_func, rnn_units, kernel_regularizer = None, kernel_initializer = 'he_normal', activation_func = 'leaky_relu'):
	if rnn_func == 'LSTM':
		res = layers.LSTM(units = rnn_units, go_backwards = False, kernel_regularizer = kernel_regularizer, kernel_initializer = kernel_initializer, name = 'LSTM_activate')(inputs)
		res = layers.LayerNormalization()(res)
		res = layers.Activation(activation_func)(res)
	elif rnn_func == 'GRU':
		res = layers.GRU(units = rnn_units, go_backwards = False, kernel_regularizer = kernel_regularizer, kernel_initializer = kernel_initializer, name = 'GRU_activate')(inputs)
		res = layers.LayerNormalization()(res)
		res = layers.Activation(activation_func)(res)
	elif rnn_func == 'BiLSTM':
		res = layers.Bidirectional(layers.LSTM(units = rnn_units, kernel_regularizer = kernel_regularizer, kernel_initializer = kernel_initializer), name = 'BiLSTM_activate')(inputs)
		res = layers.LayerNormalization()(res)
		res = layers.Activation(activation_func)(res)
	elif rnn_func == 'BiGRU':
		res = layers.Bidirectional(layers.GRU(units = rnn_units, kernel_regularizer = kernel_regularizer, kernel_initializer = kernel_initializer), name = 'BiGRU_activate')(inputs)
		res = layers.LayerNormalization()(res)
		res = layers.Activation(activation_func)(res)
	else:
		res = layers.Flatten(name = 'flatten_no_RNN')(inputs)
	return(res)	

def Dense_block(inputs, n_dense_neuron, dropout_rate = 0, kernel_regularizer = None, kernel_initializer = 'he_normal', activation_func = 'leaky_relu'):
	res = layers.Dense(
		units = n_dense_neuron, 
		kernel_regularizer = kernel_regularizer,
		kernel_initializer = kernel_initializer,
		name = 'dense')(inputs)

	if activation_func == 'selu':
		res = layers.AlphaDropout(dropout_rate, name = 'dropout_dense')(res)
		res = layers.LayerNormalization(name = 'layer_nor_dense')(res)
	elif activation_func == 'silu':
		res = layers.Dropout(dropout_rate, name = 'dropout_dense')(res)
		res = layers.LayerNormalization(name = 'layer_nor_dense')(res)
	else:
		res = layers.Dropout(dropout_rate, name = 'dropout_dense')(res)
		res = layers.BatchNormalization(name = 'batch_nor_dense')(res)
	
	res = layers.Activation(activation_func, name = 'activate_dense')(res)
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
		('n_Conv1D_filter', 32),
		('n_Conv1D_kernel', 5),
		('kernel_initializer', 'he_normal'),
		('activation_func', 'leaky_relu'), 
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

	# use the optimal initializers for different activation functions

	if params['activation_func'] == 'silu':
		params['kernel_initializer'] = 'he_normal'
	elif params['activation_func'] == 'selu':
		params['kernel_initializer'] = 'lecun_normal'
	else: # any other input will be restored to the default
		params['activation_func'] == 'leaky_relu'
		params['kernel_initializer'] = 'he_normal'

	# input layer
	inn_input = layers.Input(shape = params['input_shape'])

	# first convolution layer
	res = layers.Conv1D(
		filters = params['n_Conv1D_filter'], 
		kernel_size = params['n_Conv1D_kernel'], 
		kernel_initializer = params['kernel_initializer'],
		kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])
		)(inn_input)
	res = layers.LayerNormalization()(res)
	res = layers.Activation(params['activation_func'])(res)
	
	# Convolution block
	if params['n_Conv1D_MaxPool_block'] > 0:
		for i in range(params['n_Conv1D_MaxPool_block']):
			res = Conv1D_MaxPool_block(
				inputs = res, 
				filters = params['n_Conv1D_filter'], 
				kernel_size = params['n_Conv1D_kernel'], 
				kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']), 
				kernel_initializer = params['kernel_initializer'],
				dropout_rate = params['dropout_rate'], 
				pool_size = params['n_pool_size'], 
				activation_func = params['activation_func'])

	# Recurrent block
	res = rnn_block(
		inputs = res,
		rnn_func = params['rnn_func'], 
		rnn_units = params['n_rnn_units'], 
		kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']), 
		kernel_initializer = 'he_normal', 
		activation_func = 'leaky_relu')

	# Dense block
	res = Dense_block(
		inputs = res,
		n_dense_neuron = params['n_dense_neuron'], 
		dropout_rate = params['dropout_rate'], 
		kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']), 
		kernel_initializer = params['kernel_initializer'], 
		activation_func = params['activation_func'])

	# Final output layer
	if params['flag_initial_tl']:
		itl_input = layers.Input((1,))
		res = layers.Concatenate(axis = 1)([res, itl_input])
		inn_output = layers.Dense(1, name = 'final_dense')(res)
		model = keras.Model(inputs = [inn_input, itl_input], outputs = inn_output)
	else:
		inn_output = layers.Dense(1, name = 'final_dense')(res)
		model = keras.Model(inputs = inn_input, outputs = inn_output)

	# Use either Adam or SGD for optimization
	if params['optimizer'] == 'adam':
		model.compile(
			optimizer = keras.optimizers.Adam(learning_rate = params['learning_rate']), 
			loss = params['loss'],
			metrics = [params['metrics']]
			#metrics=[params['metrics'], custom_metric]
			) 
	else:
		model.compile(
			optimizer = keras.optimizers.SGD(learning_rate = params['learning_rate'], momentum = 0.9, nesterov = True, decay = 1e-6), 
			loss = params['loss'], 
			metrics = [params['metrics']]
			#metrics=[params['metrics'], custom_metric]
			)  

	return(model)

def cinn_model(input_params):
	# this model allow a 3D input tensor

	params = OrderedDict([
		('input_shape', (81,81,16)),
		('flag_initial_tl', False),
		('n_Conv2D_filter', 32),
		('n_Conv2D_kernel', 5),
		('kernel_regularizer', None), 
		('kernel_initializer', 'he_normal'),
		('activation_func', 'leaky_relu'), 
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
	res = layers.Conv2D(
		filters = params['n_Conv2D_filter'], 
		kernel_size = params['n_Conv2D_kernel'], 
		kernel_regularizer = params['kernel_regularizer'])(inn_input)
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

	# Dense block
	res = Dense_block(
		inputs = res,
		n_dense_neuron = params['n_dense_neuron'], 
		dropout_rate = params['dropout_rate'], 
		kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']), 
		kernel_initializer = params['kernel_initializer'], 
		activation_func = params['activation_func'])

	# Final output layer
	if params['flag_initial_tl']:
		itl_input = layers.Input((1,))
		res = layers.Concatenate(axis = 1)([res, itl_input])
		inn_output = layers.Dense(1, name = 'final_dense')(res)
		model = keras.Model(inputs = [inn_input, itl_input], outputs = inn_output)
	else:
		inn_output = layers.Dense(1, name = 'final_dense')(res)
		model = keras.Model(inputs = inn_input, outputs = inn_output)

	if params['optimizer'] == 'adam':
		model.compile(
			optimizer = keras.optimizers.Adam(learning_rate = params['learning_rate']), 
			loss = params['loss'], 
			metrics = [params['metrics']]
			#metrics=[params['metrics'], custom_metric]
			) 
	else:
		model.compile(
			optimizer = keras.optimizers.SGD(learning_rate = params['learning_rate'], momentum = 0.9, nesterov = True, decay = 1e-6), 
			loss=params['loss'], 
			metrics = [params['metrics']]
			#metrics=[params['metrics'], custom_metric]
			) 

	return(model)

class train_data_generator(keras.utils.Sequence):
	'''
	Custom batch generator for multi-group training.

	Parameters:
		- x_ary_lst: [data_input, group_input, ...], where:
			- data_input: Feature matrix (numpy array)
			- group_input: 1D array of integers indicating dataset groups
			- ...: other input (such as initial tail length)
		- y_ary: 1D array of targets
		- n_group: Number of dataset groups
		- batch_size: Number of samples per batch (default: 50)
		- shuffle: Whether to shuffle data at the end of each epoch
		- sample: If True, downsample larger groups to match the smallest one
	'''
	def __init__(self, x_ary_lst, y_ary, n_group = 1, batch_size = 50, shuffle = True, sample = False):
		# initialization 
		super().__init__()
		self.x, self.y = x_ary_lst, y_ary
		self.n_group = n_group
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.sample = sample
		self.group_ary = x_ary_lst[1]	# Extracts group indices
		self.on_epoch_end()

	def __len__(self):
		# returns the number of batches per epoch

		if self.n_group == 1:
			return(math.ceil(self.y.shape[0] / self.batch_size))
		
		if self.sample:
			# all groups have the same (downsampled) size
			n_entry_per_group = len(self.idx_lst_dict[0])  # same for all groups
			n_total_entries = n_entry_per_group * self.n_group
			return math.ceil(n_total_entries / self.batch_size)
		else:
			# compute batch count per group
			self.n_batch_lst = [
				math.ceil(np.sum(self.group_ary == i) / self.batch_size) for i in range(self.n_group)
			]

			# Determine the smallest batch count to balance training
			n_batch_min = min(self.n_batch_lst)
			
			# Adjust batch sizes for larger groups
			self.batch_size_lst = [
				math.ceil(np.sum(self.group_ary == i) / n_batch_min) for i in range(self.n_group)
			]

			return(n_batch_min * self.n_group)

	def __getitem__(self, idx):
		# Generate a batch at index `idx`
		if self.n_group == 1:
			batch_x = [m[self.idx_lst[idx * self.batch_size:(idx + 1) * self.batch_size]] for m in self.x]
			batch_y = self.y[self.idx_lst[idx * self.batch_size:(idx + 1) * self.batch_size]]
		else:
			# Determine which group to pull from
			batch_idx = math.floor(idx / self.n_group)  # Batch number within the group
			current_group = idx % self.n_group  # Which dataset group to use
			
			if self.sample:
				current_batch_size = self.batch_size
			else:
				current_batch_size = self.batch_size_lst[current_group]

			batch_idx_start = batch_idx * current_batch_size
			batch_idx_end = (batch_idx + 1) * current_batch_size

			# Extract data from the corresponding group
			batch_indices = self.idx_lst_dict[current_group][batch_idx_start:batch_idx_end]
			batch_x = [m[batch_indices] for m in self.x]
			batch_y = self.y[batch_indices]

		return (tuple(batch_x), batch_y)
		
	def on_epoch_end(self):
		# shuffle indices and (if enabled) downsample to balance dataset sizes.
		self.idx_lst = np.arange(self.y.shape[0])

		if self.shuffle == True:
			np.random.shuffle(self.idx_lst)

		if self.n_group > 1:
			# group-based shuffling
			self.idx_lst_dict = {
				i: np.random.permutation([j for j in self.idx_lst if self.group_ary[j] == i])
				for i in range(self.n_group)
			}

			if self.sample:
				# downsample all groups to match the smallest one
				n_entry_min = min(len(v) for v in self.idx_lst_dict.values())
				self.idx_lst_dict = {i: v[:n_entry_min] for i, v in self.idx_lst_dict.items()}

class SelectiveValidationCallback(Callback):
	def __init__(self, x_val_lst, y_val, monitor_group_lst = None, verbose = 0):
		'''
		Parameters:
			- x_val_lst: [data_input, group_input, ...], where:
				- data_input: Feature matrix (numpy array)
				- group_input: 1D array of integers indicating dataset groups
				- ...: other input (such as initial tail length)
			- y_val: 1D array of targets
			- group_ids: List of group indices to monitor (e.g., [0, 1]). If None, monitors all groups.
		'''
		super().__init__()
		self.x = x_val_lst
		self.y = y_val
		self.group_labels = x_val_lst[1]  # Extract group indices from second input
		self.verbose = verbose
		if monitor_group_lst is None:
			self.monitor_group_lst = np.unique(self.group_labels)
		else:
			self.monitor_group_lst = np.array([g for g in np.unique(self.group_labels) if g in monitor_group_lst])
			if len(self.monitor_group_lst) == 0:
				self.monitor_group_lst = np.unique(self.group_labels)

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}

		# Filter validation data for the monitored group
		group_mask = np.isin(self.group_labels, self.monitor_group_lst)
		x_group = [m[group_mask] for m in self.x]
		y_group = self.y[group_mask]

		val_loss, val_metric = self.model.evaluate(x_group, y_group, verbose = self.verbose)

		# Print custom validation loss and metric
		#print(f"Epoch {epoch+1}: Validation Loss for Group {','.join([str(g) for g in self.monitor_group_lst])} = {val_loss:.3f}")
		#print(f"Epoch {epoch+1}: Validation Metric for Group {','.join([str(g) for g in self.monitor_group_lst])} = {val_metric:.3f}")

		# Add custom validation loss and metric to logs without overriding standard metrics
		logs[f"val_loss_group_{','.join([str(g) for g in self.monitor_group_lst])}"] = val_loss
		logs[f"val_metric_group_{','.join([str(g) for g in self.monitor_group_lst])}"] = val_metric

@keras.utils.register_keras_serializable(package="CustomLayers")
class CustomGatherLayer(layers.Layer):
	def call(self, inputs):
		'''
		inputs[0]: List of tensors (outputs), each with shape (batch_size, 1)
		inputs[1]: Indices tensor (group_input), with shape (batch_size, 1)
		'''
		# stack the list of tensors along axis=1 to create a tensor of shape (batch_size, n_group, 1)
		stacked = tf.stack(inputs[0], axis=1)  # shape: (batch_size, n_group, 1)

		# squeeze the last dimension to make it (batch_size, n_group)
		stacked = tf.squeeze(stacked, axis=-1)  # shape: (batch_size, n_group)

		# make sure the inputs[1] has the correct dtype and shape
		indices = tf.squeeze(tf.cast(inputs[1], dtype = tf.int32), axis = -1)

		# gather elements based on group_input
		gathered = tf.gather(stacked, inputs[1], batch_dims=1)  # shape: (batch_size, 1)

		return(gathered)

	def compute_output_shape(self, input_shape):
		'''
		input_shape: List of shapes for each input tensor
		input_shape[0]: Shape of the first input (list of tensors, each with shape (batch_size, 1))
		input_shape[1]: Shape of the second input (group_input, with shape (batch_size, 1))
		'''
		# output shape is (batch_size, 1)
		return(input_shape[1][0], 1)

	def get_config(self):
		'''ensures proper serialization/deserialization'''
		config = super().get_config()
		return(config)

	@classmethod
	def from_config(cls, config):
		'''recreates the layer from its config'''
		return(cls(**config))  # uses default constructor (no custom args)

def minn_model(input_params):
	# this model allow multi-group training

	params = OrderedDict([
		('input_shape', (1056,5)),
		('flag_initial_tl', False),
		('n_group', 1),
		('n_Conv1D_filter', 32),
		('n_Conv1D_kernel', 5),
		('kernel_initializer', 'he_normal'),
		('activation_func', 'leaky_relu'),  
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

	# use the optimal initializers for different activation functions

	if params['activation_func'] == 'silu':
		params['kernel_initializer'] = 'he_normal'
	elif params['activation_func'] == 'selu':
		params['kernel_initializer'] = 'lecun_normal'
	else: # any other input will be restored to the default
		params['activation_func'] == 'leaky_relu'
		params['kernel_initializer'] = 'he_normal'

	# input layer
	data_input = layers.Input(shape = params['input_shape'], name="data_input")
	group_input = layers.Input(shape=(1,), dtype="int32", name="group_input")  # Dataset indicator
	input_lst = [data_input, group_input]

	# first convolution layer
	res = layers.Conv1D(
		filters = params['n_Conv1D_filter'], 
		kernel_size = params['n_Conv1D_kernel'], 
		kernel_initializer = params['kernel_initializer'],
		kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])
		)(data_input)
	res = layers.LayerNormalization()(res)
	res = layers.Activation(params['activation_func'])(res)
	
	# Convolution block
	if params['n_Conv1D_MaxPool_block'] > 0:
		for i in range(params['n_Conv1D_MaxPool_block']):
			res = Conv1D_MaxPool_block(
				inputs = res, 
				filters = params['n_Conv1D_filter'], 
				kernel_size = params['n_Conv1D_kernel'], 
				kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']), 
				kernel_initializer = params['kernel_initializer'],
				dropout_rate = params['dropout_rate'], 
				pool_size = params['n_pool_size'], 
				activation_func = params['activation_func'])

	# Recurrent block
	res = rnn_block(
		inputs = res,
		rnn_func = params['rnn_func'], 
		rnn_units = params['n_rnn_units'], 
		kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']), 
		kernel_initializer = 'he_normal', 
		activation_func = 'leaky_relu')

	# Dense block
	res = Dense_block(
		inputs = res,
		n_dense_neuron = params['n_dense_neuron'], 
		dropout_rate = params['dropout_rate'], 
		kernel_regularizer = keras.regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg']), 
		kernel_initializer = params['kernel_initializer'], 
		activation_func = params['activation_func'])

	# add the initial tail length if necessary
	if params['flag_initial_tl']:
		itl_input = layers.Input(shape = (1,), dtype = 'float32', name = 'initial_tail_length_input')
		res = layers.Concatenate(axis = 1)([res, itl_input])
		input_lst.append(itl_input)

	# Multiple output layers, one per dataset (len = n_group, shape for each element: (batch_size, 1))
	res_lst = [layers.Dense(1, name=f'output_list_{i}')(res) for i in range(input_params['n_group'])]

	# Select the correct output dynamically 
	final_output = CustomGatherLayer(name="final_output")([res_lst, group_input])
	
	model = keras.Model(inputs = input_lst, outputs = final_output)

	if params['optimizer'] == 'adam':
		model.compile(
			optimizer = keras.optimizers.Adam(learning_rate = params['learning_rate']), 
			loss = params['loss'], 
			metrics = [params['metrics']]
			#metrics=[params['metrics'], custom_metric]
			) 
	else:
		model.compile(
			optimizer = keras.optimizers.SGD(learning_rate = params['learning_rate'], momentum = 0.9, nesterov = True, decay = 1e-6), 
			loss=params['loss'], 
			metrics = [params['metrics']]
			#metrics=[params['metrics'], custom_metric]
			) 

	return(model)		
