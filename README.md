This is the official source code for the paper "[Uncertainty-Aware Stability Analysis of IBR-dominated Power System with Neural Networks](https://doi.org/10.1109/TPEL.2025.3560236)" published in IEEE Transactions on Power Electronics.

This project is shared under the GNU AGPLv3 license. Please ensure that you respect and adhere to the terms and conditions of the license when using, modifying, or distributing the code. For more details, refer to the LICENSE file or [https://choosealicense.com/licenses/agpl-3.0/](https://choosealicense.com/licenses/agpl-3.0/#).

## Set-up

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

## Usage

Simply run `python3 run.py` for model training, evaluation and visualisation. Folders containing trained models, results, plots etc will be created automatically.

To visualize the OP_sin dataset, run `visualize_dataset.py` which will save plots in the plots folder.

## 📚 Citation

If you find our work useful, please cite:

```bibtex
@ARTICLE{ibr_stability_ensemble_2025,
  author={Humblot-Renaux, Galadrielle and Wu, Yang and Escalera, Sergio and Moeslund, Thomas B. and Wang, Xiongfei and Wu, Heng},
  journal={IEEE Transactions on Power Electronics}, 
  title={Uncertainty-Aware Stability Analysis of IBR-dominated Power System with Neural Networks}, 
  year={2025},
  pages={1-6},
  doi={10.1109/TPEL.2025.3560236}}
```

## ✉️ Contact

If you have have any issues or doubts about the code, please create a Github issue. Otherwise, you can contact me at gegeh@create.aau.dk
