import numpy as np
import re, optuna, os, joblib
from time import time
from datetime import datetime
from sklearn.preprocessing import PowerTransformer
from matplotlib import pyplot
from scipy.stats import gaussian_kde, spearmanr, pearsonr
from matplotlib.ticker import MultipleLocator
from tensorflow.keras.models import load_model

def timer(t_start): 
	# a function to calculate runtime
	temp = str(time()-t_start).split('.')[0]
	temp =  '\t' + temp + 's passed...' + '\t' + str(datetime.now())
	return(temp)

def pwrite(f, text, timestamp=None):
	"""Write printed output to a file and print it to console."""
	log_text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {text}" if timestamp else text
	if f:
		f.write(log_text + '\n')
		f.flush()
	print(log_text)

def make_log_file(filename, p_params = False, p_vars = False):
	# a function to generate log file
	f_log = open(filename, 'w')
	if isinstance(p_params, dict):
		pwrite(f_log, 'Default parameters:')
		for param in p_params:
			pwrite(f_log, f'\t{param}: {p_params[param]}')
	if isinstance(p_vars, dict):
		pwrite(f_log, '\nInput arguments:')
		for var in p_vars:
			pwrite(f_log, f'\t{var}: {p_vars[var]}')
	return(f_log)

def update_dict_from_dict(default_dict, new_dict, exclude_keys = list(), error_log = None):
	# a function to update the default dictionary in place, with the option to not update select members specified in the "exclude_keys" list
	for key in new_dict:
		if key in default_dict and key not in exclude_keys:
			if type(default_dict[key]) == type(new_dict[key]) or default_dict[key] is None:
				default_dict[key] = new_dict[key]
			else:
				error_msg = f'Warning: different type in the input parameters from that of the default for the parameter: "{key}"! Keeping original value.'
				if error_log:
					pwrite(error_log, error_msg)
				else:
					print(error_msg)

def update_and_log(d, key, value, log = None):
	# a function to update a value in a dictionary and log the change to a file (or print out)
	if key in d:
		d[key] = value
		if 'NAME' in d:
			try:
				pwrite(log, '\n' + d['NAME'] + ' (updated):')
				pwrite(log, key + ': ' + f"{value}")
			except:
				print('Updated ' +d['NAME'])
				print(key + ': ' + f"{value}")

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

def target_transform(x, method = 'none', offset = 0, transformer=None, constant = 0, inverse = False):
	# legacy function
	# This is a function to transform or inverse-transform target values
	# x can be either a list of a 2-d array
	# it returns the transformed values and the transformer (None value if not applicable)
	# if method is 'boxcox' or 'yeo-johnson' and 'inverse' is False, it fits and transforms if 'tf' is not provided, or it only transforms if 'tf' is provided 
	# if method is 'boxcox' or 'yeo-johnson' and 'inverse' is True, a fitted transformer must be provided to 'tf'
	x = np.asarray(x)
	flag_1d = (x.ndim == 1)
	
	if method == 'diff':
		x_trans = x + constant if inverse else x - constant
	
	elif method == 'log':
		x_trans = np.expm1(x) - offset if inverse else np.log1p(x + offset)
	
	elif method == 'sqrt':
		x_trans = np.power(x, 2) - offset if inverse else np.sqrt(x + offset)
	
	elif method in ['box-cox', 'yeo-johnson']:
		if method == 'box-cox':
			new_transformer = PowerTransformer(method = 'box-cox', standardize=True)
		else:
			new_transformer = PowerTransformer(method = 'yeo-johnson', standardize=True)
		
		if transformer is not None:
			new_transformer = transformer

		x_input = x.reshape(-1, 1) if flag_1d else x

		if inverse:
			if method == 'box-cox':
				x_trans = new_transformer.inverse_transform(x_input) - offset
			else:
				x_trans = new_transformer.inverse_transform(x_input)
			
		else:
			if method == 'box-cox':
				x_input += offset
				
			if transformer is None: 
				new_transformer.fit(x_input)
			x_trans = new_transformer.transform(x_input)
			
		if flag_1d:
			x_trans = x_trans.ravel()

		return(x_trans, new_transformer if not inverse else transformer)
	else:
		x_trans = x
	
	return(x_trans, transformer)

class Target_transformer:
	def __init__(self, method = 'none', offset = 0, constant = 0):
		'''
		method : str (default='none')
			transformation method: 'none', 'diff', 'log', 'sqrt', 'box-cox', or 'yeo-johnson'
		offset : float (default=0)
			offset value for log/sqrt/box-cox transformations
		constant : float (default=0)
			constant for difference transformation
		'''
		self.method = method
		self.offset = offset
		self.constant = constant
		self.transformer = None

	def fit(self, x):
		'''
		fit the transformer if needed (for box-cox/yeo-johnson).
		x : array-like input data to fit the transformer
		'''
		x = np.asarray(x)
		if self.method in ['box-cox', 'yeo-johnson']:
			self.transformer = PowerTransformer(
				method = self.method,
				standardize = True
			)
			x_input = x.reshape(-1, 1) if x.ndim == 1 else x
			if self.method == 'box-cox':
				x_input = x_input + self.offset
			self.transformer.fit(x_input)
		return(self)

	def transform(self, x):
		'''
		apply the transformation to the input data.
		x : array-like data to transform  
		Returns: array-like transformed data
		'''
		x = np.asarray(x)
		flag_1d = (x.ndim == 1)

		if self.method == 'diff':
			return(x - self.constant)
		    
		elif self.method == 'log':
			return(np.log1p(x + self.offset))

		elif self.method == 'sqrt':
			return(np.sqrt(x + self.offset))

		elif self.method in ['box-cox', 'yeo-johnson']:
			if self.transformer is None:
				raise ValueError("Transformer not fitted. Call fit() first.")

			x_input = x.reshape(-1, 1) if flag_1d else x
			if self.method == 'box-cox':
				x_input = x_input + self.offset
		
			x_trans = self.transformer.transform(x_input)
			return(x_trans.ravel() if flag_1d else x_trans)
		
		else:
			return(x)

	def inverse_transform(self, x):
		'''
		reverse the transformation.
		x : array-like transformed data to inverse transform
		Returns: array-like original scale data
		'''
		x = np.asarray(x)
		flag_1d = (x.ndim == 1)

		if self.method == 'diff':
			return (x + self.constant)

		elif self.method == 'log':
			return(np.expm1(x) - self.offset)

		elif self.method == 'sqrt':
			return(np.power(x, 2) - self.offset)

		elif self.method in ['box-cox', 'yeo-johnson']:
			if self.transformer is None:
				raise ValueError("Transformer not fitted. Call fit() first.")

			x_input = x.reshape(-1, 1) if flag_1d else x
			x_trans = self.transformer.inverse_transform(x_input)

			if self.method == 'box-cox':
				x_trans = x_trans - self.offset

			return(x_trans.ravel() if flag_1d else x_trans)

		else:
			return(x)
			
	def fit_transform(self, x):
		return self.fit(x).transform(x)

def load_keras_models(path):
	if os.path.isdir(path):
		# If it's a folder, load all models in the folder (except the callback models)
		models = []
		for file_name in os.listdir(path):
			if 'callback' not in file_name:
				file_path = os.path.join(path, file_name)
				if os.path.isfile(file_path) and file_name.endswith('.keras'):
					models.append(load_model(file_path))
		return(models)

	elif os.path.isfile(path) and path.endswith('.keras'):
		# If it's a file, load the model
		return([load_model(path)])
	else:
		raise ValueError("The provided path is neither a valid folder nor a Keras model file.")

def load_models_and_transformers(path):
	'''
	Load Keras model(s) and associated transformer(s) from a .keras file or .pkl file, or from a folder containing them.

	Returns a list of dictionaries, each with keys:
		- 'model': the Keras model
		- 'transformer': the transformer (or None if not present)
	'''
	loaded_items = []

	if os.path.isdir(path):
		for file_name in os.listdir(path):
			if 'callback' in file_name:
				continue

			file_path = os.path.join(path, file_name)
			if file_name.endswith('.keras') and os.path.isfile(file_path):
				model = load_model(file_path)
				loaded_items.append({'model': model, 'transformer': None})

			elif file_name.endswith('.pkl') and os.path.isfile(file_path):
				bundle = joblib.load(file_path)
				model = bundle.get('model', None)
				transformer = bundle.get('transformer', None)
				loaded_items.append({'model': model, 'transformer': transformer})
	elif os.path.isfile(path):
		if path.endswith('.keras'):
			model = load_model(path)
			loaded_items.append({'model': model, 'transformer': None, 'path': path})

		elif path.endswith('.pkl'):
			bundle = joblib.load(path)
			model = bundle.get('model', None)
			transformer = bundle.get('transformer', None)
			loaded_items.append({'model': model, 'transformer': transformer, 'path': path})
		else:
			raise ValueError("The provided file is neither a .keras model nor a .pkl bundle.")
	else:
		raise ValueError("The provided path is neither a valid folder nor a recognized file.")


def save_opti_results(study, fn = 'Optimization_reuslts'):
	study.trials_dataframe().to_csv(f'{fn}_results.csv')

	# make a few plots
	# optimization history
	plt = optuna.visualization.matplotlib.plot_optimization_history(study).figure
	plt.savefig(f'{fn}_optimization_history.png', dpi = 300, bbox_inches='tight')

	# param slices
	plt = optuna.visualization.matplotlib.plot_slice(study)[0].figure
	plt.savefig(f'{fn}_search_slice.png', dpi = 300, bbox_inches='tight')

	# param importances
	plt = optuna.visualization.matplotlib.plot_param_importances(study).figure
	plt.savefig(f'{fn}_param_importances.png', dpi = 300, bbox_inches='tight')

	# timeline
	plt = optuna.visualization.matplotlib.plot_timeline(study).figure
	plt.savefig(f'{fn}_timeline.png', dpi = 300, bbox_inches='tight')

	# contour plots
	n_params = len(list(study.best_trial.params.keys()))
	fig_size = max(3, n_params * 3)
	plt = optuna.visualization.matplotlib.plot_contour(study)[0,0].figure
	plt.set_size_inches(fig_size, fig_size) 
	plt.savefig(f'{fn}_contour_map.png', dpi = 300, bbox_inches='tight')


def loss_plot(train, test, fn, y_label = 'Loss'):
	# a function to save loss of metric values of training history and make a line plot
	fn = re.sub(r'\..*', '', fn)
	with open(f'{fn}_train_test_{y_label.lower()}_by_epochs_history.txt', 'w') as f:
		f.write('Epochs\tTrain\tTest\n')
		for i in range(len(train)):
			f.write(f'{i+1}\t{train[i]}\t{test[i]}\n')
	pyplot.clf()
	pyplot.plot(train, label='train')
	pyplot.plot(test, label='test')
	pyplot.legend()
	pyplot.xlabel("Epochs")
	pyplot.ylabel(y_label)
	pyplot.gca().xaxis.set_major_locator(MultipleLocator(5))
	pyplot.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
	pyplot.savefig(f'{fn}_train_test_{y_label.lower()}_by_epochs_history.png')

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

def density_plot(df, col_x, col_group = None, fn = 'plot.png'):
	pyplot.figure(figsize=(8, 6))

	if col_group is None:
		# Case when no grouping is needed - plot single density
		data = df[col_x].dropna()

		# Main density plot
		density = gaussian_kde(data)
		x_vals = np.linspace(data.min(), data.max(), 500)
		y_vals = density(x_vals)
		pyplot.plot(x_vals, y_vals, color='blue', linewidth=2)
		pyplot.fill_between(x_vals, y_vals, color='blue', alpha=0.2)

		# Mean line
		mean_val = data.mean()
		pyplot.axvline(mean_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='Mean')

		# Plot decorations
		pyplot.title('Density Distribution with Mean Line', fontsize=14)
		pyplot.xlabel(col_x, fontsize=12)
	else:
		# Get unique groups and assign colors
		groups = df[col_group].unique()
		colors = pyplot.cm.viridis(np.linspace(0, 1, len(groups)))  # Color spectrum
		
		# Plot density and mean for each group
		for group, color in zip(groups, colors):
			group_data = df[df[col_group] == group][col_x].dropna()  # Clean data

			# 1. Density plot
			density = gaussian_kde(group_data)
			x_vals = np.linspace(group_data.min(), group_data.max(), 500)
			y_vals = density(x_vals)
			pyplot.plot(x_vals, y_vals, color=color, label=f'Group {group}', linewidth=2)
			pyplot.fill_between(x_vals, y_vals, color=color, alpha=0.2)

			# 2. Mean line (new addition)
			mean_val = group_data.mean()
			pyplot.axvline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8, label=f'Group {group} Mean')

		# Add plot decorations
		pyplot.title('Density Distribution with Mean Lines', fontsize=14)
		pyplot.xlabel(col_x, fontsize=12)
	
	pyplot.ylabel('Density', fontsize=12)

	# Combine legends and adjust position
	handles, labels = pyplot.gca().get_legend_handles_labels()
	# Remove duplicate labels (keeps order)
	unique_labels = dict(zip(labels, handles))  # Preserves last occurrence
	pyplot.legend(unique_labels.values(), unique_labels.keys(), title='Groups & Means', framealpha=0.9)

	pyplot.grid(True, alpha=0.3)
	pyplot.tight_layout()
	# Save the plot to a PNG file
	pyplot.savefig(fn, dpi=300, bbox_inches='tight')

