# Set-up

Code was tested on Ubuntu but should also run on Windows.
Tested with Python 3.11.3 and Python 3.12.3

### 1) create a conda or venv environment 

Option a: conda
```bash
conda create -n ensemble-stability python=3.11
conda activate ensemble-stability
```

Option b: python-venv
```bash
python3.11 -m venv venv-ensemble-stability

source venv-ensemble-stability/bin/activate # on ubuntu
venv-ensemble-stability\Scripts\activate # on windows
```

### 2) install required packages
```bash
pip install scikit-learn plotnine pandas tqdm
pip install https://github.com/glhr/hummingbird/archive/refs/heads/main.zip
```

# Usage

Simply run `python3 run.py` for model training, evaluation and visualisation. Folders containing trained models, results, plots etc will be created automatically.

To visualize the OP_sin dataset, run `visualize_dataset.py` which will save plots in the plots folder.