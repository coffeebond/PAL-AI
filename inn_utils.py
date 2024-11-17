# updated 20241023

import numpy as np
from time import time
from datetime import datetime
from sklearn.preprocessing import PowerTransformer
from matplotlib import pyplot
from scipy.stats import gaussian_kde, spearmanr, pearsonr

def timer(t_start): 
	# a function to calculate runtime
	temp = str(time()-t_start).split('.')[0]
	temp =  '\t' + temp + 's passed...' + '\t' + str(datetime.now())
	return(temp)

def pwrite(f, text):
	# a function to write printed output to a file 
	f.write(text + '\n')
	print(text)

def make_log_file(filename, p_params = False, p_vars = False):
	# a function to generate log file
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

def update_dict_from_file(d, fn, sep = '='):
	with open(fn, 'r') as f:
		for line in f:
			if sep in line:
				key, val = line.strip().split(sep, 1)
				if key in d:
					if type(d[key]) == bool:
						d[key] = (val.lower() == 'true')
					else:
						try:
							d[key] = type(d[key])(val)
						except ValueError:
							print(f"Warning: Cannot convert '{val}' to {type(d[key])}. Keeping original value.")

def fasta_id_parser(line, sep = '\t'):
	# a function to parse the id line in a fasta file
	lst = line.rstrip().lstrip('>').split(sep)
	return(lst)

def fasta_to_dict(file, string_filter = None, key_parser_sep = '\t', key_parser_pos = 0):
	# a function to make a dictionary from a fasta file 
	# use "string_filter" to filter out entries 
	# use "key_parser_sep" and "key_parser_pos" to determine how to extract the keys from the id lines
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
						current_id = line.rstrip().lstrip('>').split(key_parser_sep)[key_parser_pos]
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

def outer_concatenate(a1, a2, half_mask = False):
	# perform an "outer concatenation" of two 2D arrays to get a 3D array
	assert len(a1.shape) == 2 and len(a2.shape) == 2, 'Input arrays for "outer_concatenate" must be 2D!'
	out_array = np.zeros((a1.shape[0], a2.shape[0], a1.shape[1] * a2.shape[1]))
	for i in range(a1.shape[0]):
		for j in range(a2.shape[0]):
			if (not half_mask) or i <= j: 
				out_array[i, j, :] = np.outer(a1[i,:], a2[j,:]).flatten()
	return(out_array)

def encode_seq(seq, n_dim = 2, half_mask = False):
	# convert a DNA sequence to one-hot encoding matrix
	# N is [0,0,0,0]
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
	if n_dim == 3:
		mat = outer_concatenate(mat, mat, half_mask = half_mask)
	return(mat) 

def encode_num(n, ndim):
	# a function to one-hot encode a number
	assert ndim >= n
	m = np.zeros(ndim)
	m[n] = 1
	return(m)

def decode_num(ary):
	# a function to turn one-hot encoded array back to a number (reverse of "encode_num")
	ary = np.asarray(ary)
	assert np.sum(ary) == 1
	return(np.where(ary == 1)[0][0])

def target_transform(x, method = 'none', inverse = False, offset = 0, tf = None, constant = 0):
	# This is a function to transform or inverse-transform target values
	# x can be either a list of a 2-d array
	# it returns the transformed values and the transformer (None value if not applicable)
	# if method is 'boxcox' or 'yeo-johnson' and 'inverse' is False, it fits and transforms if 'tf' is not provided, or it only transforms if 'tf' is provided 
	# if method is 'boxcox' or 'yeo-johnson' and 'inverse' is True, a fitted transformer must be provided to 'tf'
	if len(np.asarray(x).shape) == 1:
		flag_1d = True
	else:
		flag_1d = False
	if method == 'diff':
		if inverse:
			x_trans = x + constant
		else:
			x_trans = x - constant
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


def loss_plot(train, test, fn):
	# a function to make a line plot for monitoring training history
	pyplot.clf()
	pyplot.plot(train, label='train')
	pyplot.plot(test, label='test')
	pyplot.legend()
	pyplot.xlabel("Epochs")
	pyplot.ylabel("Loss or metric")
	pyplot.savefig(fn)

def scatter_plot(x, y, fn):

	# Calculate point density
	xy = np.vstack([x, y])
	z = gaussian_kde(xy)(xy)

	# Calculate Spearman and Pearson correlation coefficients
	spearman_corr, _ = spearmanr(x, y)
	pearson_corr, _ = pearsonr(x, y)

	# Calculate the number of points
	n = len(x)

	# Create scatter plot
	pyplot.figure(figsize=(8, 6))
	scatter = pyplot.scatter(x, y, c=z, cmap='viridis', s=20)

	# Add labels
	pyplot.xlabel('Measured data')
	pyplot.ylabel('Predicted data')

	# Add color bar
	pyplot.colorbar(scatter, label='Density')

	# Ensure x and y axes have the same scale
	pyplot.axis('equal')

	# Plot dashed line at x = y
	min_val = min(min(x), min(y))
	max_val = max(max(x), max(y))
	pyplot.plot([min_val, max_val], [min_val, max_val], '--', color='red', linewidth=1)

	# Add correlation and sample size text in the bottom right corner
	pyplot.text(0.95, 0.05, f'n={n}\nPearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}',
		horizontalalignment='right', verticalalignment='bottom',
		transform=pyplot.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

	# Save the plot to a PNG file
	pyplot.savefig(fn, dpi=300, bbox_inches='tight')

