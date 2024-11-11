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

## Options for the input

-   `-p`: A text file with global and INN parameters. Global (`params_global`) and INN (`inn_params`) parameters defined in the `INN_main.py` file can be changed with this text file. One parameter can be specified in each line. The format is the name of the parameter and its value, separated by `=`. For example, `len_max=84`
-   `-d`: Input file with 3' UTR ID and tail-length difference as target values. This file should be tab-delimited. Two columns (with column names by default) must be present. The first column contains the unique identifier for each entry (3' UTR ID), and the second column contains corresponding target values (usually poly(A) tail-length changes or translational changes) for the model to train or predict. An optional third column may be included, which can be used as a metric for filtering. It usually contains the number of poly(A) tags (reflecting the abundance) for each entry.
-   `-i`: Input file with 3'-UTR ID and initial tail length (optional). During training and predicting poly(A) tail-length change, an initial tail length can be provided. The identifiers (3'-UTR ID) must be the same as those in the input file specified by `-i`. The global parameter `flag_initial_tl` will be automatically set to `True` for this application.
-   `-u`: Input (3'-UTR) sequences in FASTA format. The description line for each entry (mRNA) must contain the same unique identifiers (3'-UTR ID) as those in the input file specified by `-i`. Additional information can be present in the description line, but it must follow the 3' UTR ID and be separated by double underscores `__`. When sequences of the coding region are used (specified by `-c`), the gene ID must follow the 3' UTR ID, separated by double underscores `__`, and additional information can follow the gene ID, also separated by double underscores `__`. The format of the description line should be `3'-UTR ID__other information` when `-c` is not specified, and `3'-UTR ID__gene ID__other information` when `-c` is specified.
-   `-c`: Input (coding region) sequences in FASTA format (optional). The description line for each entry (gene) must contain unique identifiers (gene ID). Additional information can be present in the description line, but it must follow the gene ID and be separated by double underscores `__`. Note that each gene can have multiple different 3' UTRs.
-   `-f`: Input file containing base-paring probability for each 3' UTR (optional). The format of this file should be similar to the input file with 3'-UTR ID, except that for each 3' UTR, after the description line and the sequence line, a third line is included with base-pairing probability for each nucleotide in the sequence. This information can be generated with RNA structure prediction tools, such as [RNAfold](<https://www.tbi.univie.ac.at/RNA/RNAfold.1.html>).
-   `-md`: Model file for predictions in h5 format. By default, PAL-AI-trained models are saved in h5 format. In **prediction** mode, pre-trained models can be loaded via this option.
-   `-n`: Choice of neural network for the model (default: `inn`). An alternative option is a ResNet model: `resnet`.
-   `-o`: Folder name for output files (default: `INN_out`)
-   `-l`: Maximal length of sequence (from 3'-end) used as the input
-   `-t`: The function to transform target values, choice of `none` (default), `diff`, `log`, `sqrt`, `box-cox`, or `yeo-johnson`.
-   `-v`: Verbose for the output during the run of PAL-AI (default: `0`).
-   `-s`: If specified, the sequence conversion step will be skipped. Must use with `-x` and `-y`.
-   `-x`: Input file in `npy` format with converted matrices from sequences
-   `-y`: Input file in `csv` format with target values and associated labels
-   `-m`: Mode to run this script, choice of `cv` (cross-validation), `opti` (optimization for hyperparameter search), `test` (testing), or `predict` (prediction).
-   `-op`: Number of hyperparameter combinations. Only used in `opti` mode, default: `100`.

## Performing PAL-AI

### 1. Training mode

The **training** mode can be specified by the option `-m` with the value of either `cv`, when PAL-AI performs training and testing with 10-fold cross-validation (5 times for each fold), or `test`, when PAL-AI performs training and testing once. The `cv` mode is used to generate an ensemble of models for predictions, while the `test` model is useful for debugging and testing parameters.

In the training mode, an input file containing target values (specified by `-d`) and an input file containing the sequences (in fasta format, specified by `-u`) must be provided.

An example run can be performed using the provided data in the `Data` folder as below:

```         
python -u INN_main.py 
          -d Data/XL_oo_pp_7h_tail_length_change.txt \
          -u Data/XL_utr3.fa \
          -p INN_params.txt \
          -o Output \
          -l 1056 \
          -m test \
          -v 1 
```

### 2. Prediction mode

The **prediction** mode can be specified by the option `-m` with the value of `prediction`.

PAL-AI models trained on poly(A)-tail length datasets generated in *Xenopus laevis* oocytes are provided in the `Models` folder. Only 3' UTRs upto 1053 nt long were used (no coding regions or RNA folding information) when these models were trained. Note that, the final predictions should be averaged among results predicted by all 10 models as an ensemble.

An example run can be performed using the provided data in the `Data` folder and pre-trained models in the `Models` folder as below:

```         
python INN_main.py \
    -d Data/XL_oo_pp_7h_tail_length_change.txt \
    -u Data/XL_utr3.fa \
    -p INN_params.txt \
    -o Output \
    -l 1056 \
    -md Models/XL_oo_pp_7h_tag_50_lenMax_1056_model_1.h5 \
    -m predict \
    -v 1
```

### Optimization mode

The **optimization** mode can be specified by the option `-m` with the value of `opti`. The hyperparameters that can be optimized are specified in the `INN_main.py` file by the `inn_params_lst` variable. Search space for individual hyperparameters can be changed within this list. The optimization process is executed through the [`gp_minimize`](<https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html>) function. The number of searches can be specified by the `-op` option. For each search, the `R-squared` value and search parameters are written to a file ending with `_hyperparameter_search_stats.txt`.

An example run can be performed using the provided data in the `Data` folder as below:

```         
python INN_main.py \
    -d Data/XL_oo_pp_7h_tail_length_change.txt \
    -u Data/XL_utr3.fa \
    -p INN_params.txt \
    -o Output \
    -l 1056 \
    -m opti \
    -op 500 \
    -v 0
```
