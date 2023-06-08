# ECG data preprocessing

## Basic setup

1. [Download](https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip)
   and unzip files into `mit-bih` directory
2. Install requirements via `pip install -r requirements.txt`

## Train and validation data

### Generating train and validation data

```bash
cd train_val_128
python plot_train_data_gen.py
python generate_json.py
```

### Generating downsample train and validation data

```bash
cd train_val_24
python plot_train_data_gen.py
python generate_json.py
```

## Test data

### Generating test data

```bash
cd test_true_peak_128
python plot_test_data.py
python generate_test_json.py
```

### Generating downsample test data

```bash
cd test_true_peak_24
python plot_test_data.py
python generate_test_json.py
```

### Generating test data with peak detection algorithm

```bash
cd test_detect_peak_128
python plot_test_data.py
python generate_test_json.py
```

### Generating downsample test data with peak detection algorithm

```bash
cd test_detect_peak_24
python plot_test_data.py
python generate_test_json.py
```
