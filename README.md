# ------ Instructions ------

## Options for the input
* `-d`: Input file with ID and tail-length difference as target values
* `-i`: Input file with ID and initial tail length (optional)
* `-u`: Input (3'-UTR) sequences in FASTA format 
* `-c`: Input (coding region) sequences in FASTA format (optional)
* `-f`: Input containing base-paring probability for each sequence (optional)
* `-md`: Model file for predictions in h5 format
* `-n`: Choice of neural network for the model (default: `inn`)
* `-p`: A text file with global and INN parameters (each parameter as a line and use ": " as the delimiter)
* `-o`: Folder for output files (default: `INN_out`)
* `-l`: Maximal length of sequence (from 3'-end) used as the input
* `-t`: The function to transform target values, choice of `none` (default), `diff`, `log`, `sqrt`, `box-cox`, or `yeo-johnson`
* `-v`: Verbose for the output during the run of this script (default: `0`)
* `-s`: If specified, the sequence conversion step will be skipped. Must use with `-x` and `-y`.
* `-x`: Input file in `npy` format with converted matrices from sequences
* `-y`: Input fiel in `csv` format with target values and associated labels
* `-m`: Mode to run this script, choice of `cv` (cross validation), `opti` (optimization for hyperparameter search), `test` (testing), or `predict` (prediction)
* `-op`: Number of hyperparameter combinations. Only used in `opti` mode, default: `100`.

## Examples of performing PAL-AI

### Training mode
```
python INN_main.py \
	-d Tail_length_change.txt \
	-u UTR_sequences.fa \
	-p INN_params.txt \
	-o Output \
	-l 1056 \
	-m cv \
	-v 0

```

### Prediction mode
```
python INN_main.py \
	-d Tail_length_change.txt \
	-u UTR_sequences.fa \
	-p INN_params.txt \
	-o Output \
	-l 1056 \
	-md INN_model.h5 \
	-m predict \
	-v 0

```

### Optimization mode
```
python INN_main.py \
	-d Tail_length_change.txt \
	-u UTR_sequences.fa \
	-p INN_params.txt \
	-o Output \
	-l 1056 \
	-m opti \
	-op 500 \
	-v 0

```
