# PAL-AI

## Prepare a virtual environment to run PAL-AI.

1.  Install `virtualenv` (if it's not already installed).

```         
pip install virtualenv
```

2.  Create a virtual environment.

```         
virtualenv PAL_AI_env --python=python3.8
```

3.  Activate the environment and install required packages listed in the `PAL_AI_environment.txt` file.

```         
source PAL_AI_env/bin/activate
pip install -r PAL_AI_environment.txt
```

Once installed, the virtual environment can be activated by

```         
source PAL_AI_env/bin/activate
```

## PAL-AI-s versus PAL-AI-m
**PAL-AI-s** is the original PAL-AI, which can be trained with a single dataset and predict poly(A) tail-length changes from mRNA sequences.

**PAL-AI-m** can be trained with multiple datasets, with dataset-specific output heads. This allows the model to learn common features of different datasets, while making dataset-specific predictions.

### Input options

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
            -o Output \
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
            -o Output \
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
            -l 2000
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
    1. A text file with predicted values from all models 
    2. A text file with ensemble-averaged predicted values (from all models)
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
            -o Optimizations \
            -l 2000 
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
            -o Optimizations \
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


