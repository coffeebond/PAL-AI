# PAL-AI
**Predicts poly(A) tail-length changes during oocyte maturation from mRNA sequences using integrated neural networks.**

---

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [PAL-AI-s vs. PAL-AI-m](#pal-ai-s-versus-pal-ai-m)
- [Input Options](#input-options)
- [Running PAL-AI](#running-pal-ai)
  - [Training Mode](#1-training-mode)  
  - [Prediction Mode](#2-prediction-mode)  
  - [Optimization Mode](#3-optimization-mode) 
- [Pre-trained Models](#pre-trained-models)   
- [Configuration](#configuring-pal-ai-with-the-yaml-file)  
  - [Global Parameters](#global-parameters-global_config)  
  - [Model Parameters](#model-parameters-inn_params)  
  - [Optimization Parameters](#parameters-for-optimization-inn_params_opti)
- [Data Information](#data-information)  
- [Citation](#citation)

---

## System Requirements
PAL-AI has been tested on a **Ubuntu 20.04** with:

- **CPU**: Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz (4 cores)
- **RAM**: 128 GB
- **GPU**: GeForce RTX 2080 Ti (recommended for 5–100× speedup)  

**Runtime Examples**:
| Mode    | Model    | Runtime     |
| ------- | -------- | ----------- |
| cv      | PAL-AI-s | 1 hr 30 min |
| cv      | PAL-AI-m | 3 hr 20 min |
| predict | PAL-AI-s | 31 sec      |
| predict | PAL-AI-m | 33 sec      |
| opti    | PAL-AI-s | 61 hr       |
| opti    | PAL-AI-m | 71 hr       | 

---

## Installation

1.  Install `virtualenv` (if it's not already installed).

```         
pip install virtualenv
```

2.  Create a virtual environment.

```         
virtualenv PAL_AI_env --python=python3.11
```

3.  Activate the environment and install required packages listed in the `PAL_AI_environment.txt` file.

```         
source PAL_AI_env/bin/activate
pip install -r PAL_AI_environment.txt
```

Once installed, the virtual environment can be activated anytime by

```         
source PAL_AI_env/bin/activate
```

---

## PAL-AI-s versus PAL-AI-m

**PAL-AI-s** is the original PAL-AI, which can be trained with a single dataset and predict poly(A) tail-length changes from mRNA sequences.

**PAL-AI-m** can be trained with multiple datasets, with dataset-specific output heads. This allows the model to learn common features of different datasets, while making dataset-specific predictions.

---

## Input options

#### options shared by PAL-AI-s and PAL-AI-m

- `-p`: A YAML file for updating parameters (examples provided in `INN_config.yaml` and `MINN_config.yaml`, for **PAL-AI-s** and **PAL-AI-m**, respectively). Note that some parameters can also be updated using options passed through the python script (as specified below), and in such cases, values of these parameters, either default or defined in the YAML file, will be **overwritten by the input options**.
- `-o`: Folder name for output files. This can also be defined by `out_folder` in the YAML file.
- `-l`: Maximal length of sequnence used for the model. This can also be defined by `len_max` in the YAML file.
- `-x`: Input file in `.npy` format with converted matrices from sequences. This intermediate file is generated when running the script. Loading such a file can skip the data-conversion step. 
- `-y`: Input file in `.csv` format with target values and associated labels. This intermediate file is generated when running the script. Loading such a file can skip the data-conversion step. 
- `-n`: Choice of neural network for the model (default: `inn`). An alternative option is a ResNet model: `resnet`.
- `-m`: Mode to run this script, choice of `cv` (_**cross-validation**_), `opti` (_**optimization**_), `test` (_**testing**_), or `predict` (_**prediction**_).
- `-op`: Number of hyperparameter combinations (default: `100`).. Only used in _**optimization**_ mode.
- `-md`: A model file (in `.keras` format) or a model-file-containing folder for predictions. By default, PAL-AI-trained models are saved in `.keras` format. In _**prediction**_ mode, pre-trained models can be loaded via this option. When a folder is provided, all model files within that folder will be used to make predictions.


#### PAL-AI-s-specific options

- `-d`: Input file with 3' UTR ID and tail-length difference as target values. This file should be tab-delimited. Two columns (with column names by default) must be present. The first column contains the unique identifier for each entry (3' UTR ID), and the second column contains corresponding target values (usually poly(A) tail-length changes or translational changes) for the model to train. An optional third column may be included, which can be used as a metric for filtering. It usually contains the number of poly(A) tags (reflecting the abundance) for each entry.
- `-u`: Input (3'-UTR) sequences in FASTA format. The description line for each entry (mRNA) must contain the same unique identifiers (3'-UTR ID) as those in the input file specified by `-d`. Additional information can be present in the description line, but it must follow 3'-UTR ID and be separated by double underscores `__`. When sequences of the coding region are used (specified by `-c`), the gene ID must follow the 3'-UTR ID, separated by double underscores `__`. Additional information that follow the gene ID must also be separated by double underscores `__`. 
- `-c`: Input (coding region) sequences in FASTA format (optional). The description line for each entry (gene) must contain unique identifiers (gene ID). Additional information in the description line must follow the gene ID and be separated by double underscores `__`. Note that each gene can have multiple different 3' UTRs.
- `-f`: Input file containing per-base unpairing probability for each 3' UTR (optional). The format of this file should be similar to the input file with 3'-UTR ID, except that for each 3' UTR, after the description line and the sequence line, a third line is included with per-base unpairing probability value for each nucleotide in the sequence. This information can be generated with RNA structure prediction tools, such as [RNAfold](https://www.tbi.univie.ac.at/RNA/RNAfold.1.html).
- `-i`: Input file with 3'-UTR ID and initial poly(A)-tail lengths (optional). The identifier (3'-UTR ID) must be the same as those in the input file specified by `-d`. The global parameter `flag_initial_tl` will be automatically set to `True` for this application.

#### PAL-AI-m-specific options
- `-i`: A text file with information about input files. This file should be tab-delimited, with a total of three columns. The first row is column names. The first column specifies the group number of each dataset for the input, starting from 1, 2, ..., and so forth. The second column specifies the type of the input file and the third column points to the path of the input file. Value of the second column be one of the following: `x_utr` (input file with 3'-UTR sequence, as `-u` for **PAL-AI-s**), `x_cds` (input file with coding sequences, as `-c` for **PAL-AI-s**), `x_fold` (input file with per-base unpairing probability, as `-f` for **PAL-AI-s**), and `y` (input file with target values, as `-d` for **PAL-AI-s**).

---

## Running PAL-AI

### 1. Training mode

The _**training**_ mode can be specified by the option `-m` with the value of either `cv`, when PAL-AI performs training and testing with 10-fold cross-validation, or `test`, when PAL-AI performs training and testing once. The `cv` mode is used to generate an ensemble of models for predictions, while the `test` model is useful for debugging and testing parameters.

#### PAL-AI-s

An input file containing target values (specified by `-d`) and an input file containing the sequences (in fasta format, specified by `-u`) must be provided.

An example run can be performed using the provided data in the `Data` folder as below:
```         
python -u INN_main.py \
            -d Data/XL_oo_pp_7h_tail_length_change.txt \
            -u Data/XL_utr3.fa \
            -c Data/XL_cds.fa \
            -p INN_config.yaml  \
            -o CV_output/PAL_AI_s \
            -l 2000 \
            -m cv 
```
#### PAL-AI-m

A text file (specified by `-i`) containing information about dataset-specific input files must be provided.

An example run can be performed using the provided data in the `Data` folder as below:
```         
python -u INN_minn.py \
            -i Data/Datasets_XL_HS_MM_wCDS.txt \
            -p MINN_config.yaml \
            -o CV_output/PAL_AI_m \
            -l 2000 \
            -m cv

```

#### OUTPUT

The following files will be generated in the output:
1. Under `Data` folder: 
    1. Converted input matrices in `.npy` format
    2. Target values and associated labels in a `.csv` file 
2. Under `Loss_plots` folder:
    1. Loss by epochs for all cross-validation models
    2. Metrics by epochs for all cross-validation models
3. Under `Models` folder:
    1. All cross-validation models in `.keras` format
    2. Best-performing model from each fold also under the `Best_models` subfolder
4. Under `Predictions` folder:
    - A text file with the following columns:
        1. `cv_group`: the fold number of the cross-validation
        2. `id`: unique ID for each mRNA isoform
        3. `y`: target values (measured) for predictions
        4. `y_pred`: model-predicted values
        5. `label`: group labels for data stratification
5. Under `Scatter_plots` folder:
    - A scatter plot comparing measured and predicted values
6. Under `Density_plots` folder:
    - Density plots showing distributions of the target values before and after transformations
7. A text file reporting the statistics of all trained models in the cross-validation process
8. A log file reporting information about the run

### 2. Prediction mode

The _**prediction**_ mode can be specified by the option `-m` with the value of `prediction`.

#### PAL-AI-s

An example run can be performed using the provided data in the `Data` folder and pre-trained models in the `Models` folder as below:

```         
python -u INN_main.py \
            -u Data/XL_F044_utr3.fa \
            -c Data/XL_F044_cds.fa \
            -p INN_config.yaml \
            -o Predictions/XL_F044/PAL_AI_s_frog_mRNAs \
            -l 2000 \
            -m predict \
            -md Models/PAL_AI_s_frog_mRNAs
```

#### PAL-AI-m

Because PAL-AI-m has multiple output heads, the desired choice must be specified by the parameter `predict_model_index` under `inn_params` in the input YAML file. Note that output head `n` corresponds to value `n-1` for `predict_model_index`.

An example run can be performed using the provided data in the `Data` folder and pre-trained models in the `Models` folder as below (here `predict_model_index` should be set to value `0`):

```
python -u INN_minn.py \
            -i Data/Datasets_F044_wCDS.txt \
            -p MINN_config.yaml \
            -o Predictions/XL_F044/PAL_AI_m/Group_1 \
            -l 2000 \
            -m predict \
            -md Models/PAL_AI_m/Group_1
```

#### OUTPUT

The following files will be generated in the output:
1. Under `Data` folder: 
    1. Converted input matrices in `.npy` format
    2. Target values and associated labels in a `.csv` file 
2. Under `Predictions` folder:
    1. A text file with predicted values from all models (one column per model)
    2. A text file with ensemble-averaged predicted values (from all models) with the following columns:
        1. `id`: unique ID for each mRNA isoform
        2. `y`: target values (measured) for predictions, only if provided in the input 
        3. `y_pred`: model-predicted values
        4. `label`: group labels for data stratification
3. Under `Scatter_plots` folder:
    - A scatter plot comparing measured and predicted values (only if target values are provided in the input)
4. A log file reporting information about the run

### 3. Optimization mode

The **optimization** mode can be specified by the option `-m` with the value of `opti`. 

The hyperparameters that can be optimized are specified under the parameter group `inn_params_opti` in the YAML file. The optimization process is performed using the [`Optuna`](https://optuna.readthedocs.io/en/stable/#) framework. The number of searches can be specified by the `-op` option. 

#### PAL-AI-s

An input file containing target values (specified by `-d`) and an input file containing the sequences (in fasta format, specified by `-u`) must be provided.

An example run can be performed using the provided data in the `Data` folder as below:
```         
python -u INN_main.py \
            -d Data/XL_oo_pp_7h_tail_length_change.txt \
            -u Data/XL_utr3.fa \
            -c Data/XL_cds.fa \
            -p INN_config.yaml \
            -o Optimizations/PAL_AI_s \
            -l 2000 \
            -m opti \
            -op 1000
```
#### PAL-AI-m

A text file (specified by `-i`) containing information about dataset-specific input files must be provided.

An example run can be performed using the provided data in the `Data` folder as below:

```         
python -u INN_minn.py \
            -i Data/Datasets_XL_HS_MM_wCDS.txt \
            -p MINN_config.yaml \
            -o Optimizations/PAL_AI_m \
            -l 2000 \
            -m opti \
            -op 1000
```

#### OUTPUT

The following files will be generated in the output:
1. Under `Data` folder: 
    1. Converted input matrices in `.npy` format
    2. Target values and associated labels in a `.csv` file 
2. Under `Loss_plots` folder:
    1. Loss by epochs for the best-performing model
    2. Metrics by epochs for the best-performing model
3. Under `Models` folder:
    1. The best-performing model in `.keras` format
    2. A callback model (only used during training, not relevant for output)
4. Under `Optimization` folder:
    1. A `.db` SQLite database containing infomation about the optmization process
    2. A `_log.txt` file containing statistics of all optimization trials
    3. A `.csv` file with all hyperparameters used in each trial
    4. A few plots visualizing the optimization process and results
5. Under `Scatter_plots` folder:
    - A scatter plot comparing measured and predicted values
6. Under `Density_plots` folder:
    - Density plots showing distributions of the target values before and after transformations
7. A log file reporting information about the run

---

## Pre-trained models

The following pre-trained models are provided in the `Models` folder.
1. **PAL-AI-s** trained with **frog endogenous mRNAs**: `PAL_AI_s_frog_mRNAs`
2. **PAL-AI-s** trained with **N60(LC)-PAS<sup>_mos_</sup>** mRNA library: `PAL-AI_s_N60_LC_PASmos`
3. **PAL-AI-m** trained with **frog endogenous mRNAs** (output head 1) and **N60(LC)-PAS<sup>_mos_</sup>** mRNA library (output head 2): `PAL_AI_m`
4. **PAL-AI-ms** trained with endogenous mRNAs from **frog** (output head 1), **human** (output head 2), and **mouse** (output head 3) oocytes: `PAL_AI_ms`

---

## Configuring PAL-AI 

PAL-AI can be configured with the following parameters in the YAML file (`-p` option):

### Global Parameters (`global_config`)

These parameters control preprocessing, input configuration, training behavior, and output handling. They apply across different modules and affect model behavior globally.

##### Input Filtering and Preprocessing

- `fa_string_filter`: *String*. Filters out entries in the input FASTA file (3′ UTR) whose headers contain this string. Useful for excluding low-confidence annotations.
- `flag_input_header`: *Boolean*. If `True`, the first line of the input file (typically a header) will be skipped.
- `flag_initial_tl`: *Boolean*. If `True`, initial tail length values will be incorporated into the input. This flag is automatically set to `True` when a file containing initial tail lengths is provided via the `-i` option (for **PAL-AI-s**) or in the input file specified with `-i` (for **PAL-AI-m**).
- `input_tag_min`: *Int*. If the input file includes a third column (e.g., read counts), entries with a value below this threshold will be excluded.
- `seq_3end_append`: *String*. Nucleotide sequence to append to the 3′ ends of input sequences.
- `len_min`: *Int*. Minimum length (after appending `seq_3end_append`) for a 3′ UTR sequence to be included.
- `len_max`: *Int*. Maximum input sequence length after 3′ end appending. Overridden by the `-l` option.

##### Target Value Transformation

- `y_mode`: *String*. Method for transforming target values. Options:
  - `none`: No transformation.
  - `diff`: Subtract constant defined by `tl_cons`.
  - `log`, `sqrt`, `box-cox`: Apply respective function after shifting targets to be all positive.
  - `yeo-johnson`: Apply Yeo-Johnson transformation.
- `y_offset`: *Int*. Offset applied to target values. Automatically set when using `log`, `sqrt`, or `box-cox`.
- `tl_cons`: *Int*. Constant subtracted from target values when `y_mode` is `diff`.

##### Model Training and Evaluation

- `test_size`: *Float*. Fraction of data used as the test set (and for defining cross-validation splits). Must be between 0 and 1.
- `n_rep`: *Int*. Number of repetitions performed during training for each cross-validation fold.
- `n_dimension`: *Int*. Input dimensionality:
  - `2`: Simple one-hot encoding.
  - `3`: Outer product of one-hot encoding (used when base-pairing probabilities from RNAfold are included).

##### Output and Verbosity

- `flag_print_model`: *Boolean*. If `True`, prints the model architecture summary.
- `out_folder`: *String*. Output directory path. Defaults to `'Output'` unless overridden by the `-o` option.
- `idf`: *String*. Identifier prefix for output files. Automatically generated if `None`.
- `verbose`: *Int*. Logging verbosity level. Options: `0` (silent), `1` (default), `2` (detailed). Passed to `model.fit()` in TensorFlow.

##### Parallel Processing

- `n_threads`: *Int*. Number of CPU cores used to process input data in parallel.
- `chunk`: *Int*. Number of entries each thread processes in a batch during input preprocessing.

##### Stratification 

- `n_cpe_max`: *Int*. Maximum number of CPEs (cytoplasmic polyadenylation elements) considered when generating mRNA stratification labels. mRNAs with more CPEs than this value are grouped together.

### Model Parameters (`inn_params`)

These parameters define the architecture and training behavior of PAL-AI. Parameters fall into general architecture, recurrent layers, regularization/training, and specialized modules like ResNet and 3D convolution.

##### General Architecture

- `input_shape`: *List or tuple*. Automatically inferred from input data. Typically left unspecified by the user.
- `flag_initial_tl`: *Boolean*. Must match `flag_initial_tl` in the global configuration (`global_config`).
- `n_Conv1D_filter`: *Int*. Number of filters in each Conv1D layer.
- `n_Conv1D_kernel`: *Int*. Kernel size (width) used in Conv1D layers.
- `n_Conv1D_MaxPool_block`: *Int*. Number of Conv1D blocks, where each block consists of Conv1D, activation, and pooling layers. If `n = 3`, the block is repeated twice.
- `n_pool_size`: *Int*. Pooling size used in MaxPooling1D layers.
- `dropout_rate`: *Float*. Dropout rate applied after selected layers for regularization.
- `n_dense_neuron`: *Int*. Number of units in the final fully connected (Dense) layer.
- `activation_func`: *String*. Activation function used in the model. Options: `selu`, `silu`, `leaky_relu`.

##### Recurrent Layer Parameters

- `rnn_func`: *String*. Type of RNN layer to use. Options:
  - `GRU`: Gated Recurrent Unit
  - `BiGRU`: Bidirectional GRU
  - `LSTM`: Long Short-Term Memory
  - `BiLSTM`: Bidirectional LSTM
- `n_rnn_units`: *Int*. Number of hidden units in the recurrent layer.

##### Regularization and Training

- `l1_reg`: *Float*. L1 regularization coefficient (used for sparsity).
- `l2_reg`: *Float*. L2 regularization coefficient (used for weight decay).
- `optimizer`: *String*. Optimizer for model training (e.g., `adam`, `nadam`).
- `learning_rate`: *Float*. Learning rate used by the optimizer.
- `loss`: *String*. Loss function (e.g., `mse`, `mae`).
- `metrics`: *String*. Training evaluation metric (e.g., `mae`, `mse`).
- `batch_size`: *Int*. Number of samples per training batch.
- `epochs`: *Int*. Number of complete training passes over the dataset.

##### Parameters for ResNet Architecture (`-n resnet`)

- `resnet_trim`: *String*. Trimming strategy for input alignment before residual connections.
- `resnet_version`: *Int*. ResNet architecture version (`1` or `2`).
- `resnet_n_group`: *Int*. Number of residual groups.
- `resnet_n_block`: *Int*. Number of residual blocks per group.
- `dilation_rate_lst`: *List of Int*. Dilation rates applied per group. Length must match `resnet_n_group`.

##### Parameters for 3D Convolution (`n_dimension = 3` in `global_config`)

- `n_Conv2D_filter`: *Int*. Number of filters in each Conv2D layer.
- `n_Conv2D_kernel`: *Int*. Kernel size for Conv2D layers.
- `n_Conv2D_MaxPool_block`: *Int*. Number of Conv2D blocks, each consisting of Conv2D, activation, and pooling layers.

###  Parameters for Optimization (`inn_params_opti`)

These parameters define the hyperparameter search space for architecture and training configuration during model optimization (e.g., using Optuna). Most parameters are provided as lists or value ranges to enable sampling.


##### Convolutional Layers

- `n_Conv1D_filter`: *List[Int]* — Numbers of filters for Conv1D layers to sample from.
- `n_Conv1D_kernel`: *List[Int]* — Kernel sizes for Conv1D layers to sample from.
- `n_Conv1D_MaxPool_block`: *List[Int]* — Number of convolutional blocks (Conv1D + MaxPooling) to sample from.

##### Parameters for 3D Convolution (`n_dimension = 3` in `global_config`)

- `n_Conv2D_filter`: *List[Int]* — Numbers of filters for Conv2D layers to sample from.
- `n_Conv2D_kernel`: *List[Int]* — Kernel sizes for Conv2D layers to sample from.
- `n_Conv2D_MaxPool_block`: *List[Int]* — Number of convolutional blocks (Conv2D + MaxPooling) to sample from.

##### Dense and Recurrent Layers

- `n_dense_neuron`: *List[Int]* — Numbers of units in the dense (fully connected) layer to sample from.
- `rnn_func`: *List[String]* — Types of RNN layers to sample from (e.g., `'GRU'`, `'LSTM'`, `'BiGRU'`, `'BiLSTM'`).
- `n_rnn_units`: *List[Int]* — Numbers of hidden units in the recurrent layer to sample from.

##### ResNet Configuration

- `resnet_version`: *List[Int]* — ResNet architecture versions to sample from.
- `resnet_n_group`: *List[Int]* — Number of residual groups to sample from.
- `resnet_n_block`: *List[Int]* — Number of residual blocks per group to sample from.

##### Other Architecture and Training Parameters

- `n_pool_size`: *List[Int]* — Pooling sizes for MaxPooling1D layers to sample from.
- `activation_func`: *List[String]* — Activation functions to sample from (e.g., `'selu'`, `'leaky_relu'`, `'silu'`).
- `loss`: *List[String]* — Loss functions to sample from (e.g., `'mse'`, `'mae'`).

##### Regularization

- `l1_reg`:
  - `min`: *Float* — Lower bound of log-uniform distribution to sample L1 regularization values from.
  - `max`: *Float* — Upper bound of log-uniform distribution.
- `l2_reg`:
  - `min`: *Float* — Lower bound of log-uniform distribution to sample L2 regularization values from.
  - `max`: *Float* — Upper bound of log-uniform distribution.
- `dropout_rate`: *List[Float]* — Dropout rates to sample from.

##### Training Configuration

- `optimizer`: *List[String]* — Optimizer types to sample from (e.g., `'adam'`, `'sgd'`).
- `learning_rate`:
  - `min`: *Float* — Lower bound of log-uniform distribution for learning rate sampling.
  - `max`: *Float* — Upper bound of log-uniform distribution.
- `batch_size`: *List[Int]* — Batch sizes to sample from.
- `epochs`: *List[Int]* — Number of training epochs to sample from.


---

## Data information
A summary of provided data files can be found here:

```Data_summary.xlsx```

---

## Citation

If you use PAL-AI in your research, please cite:

Xiang & Bartel, bioRxiv, 2025 [https://doi.org/10.1101/2024.10.29.620940]
