# PINN-DeepONet

This repository contains code and experiments that implement physics-informed neural networks (PINNs) and operator-learning (DeepONet) techniques applied to electrochemical battery modeling and parameter estimation. The implementation and the onsite transfer learning workflow used in the experiments are described in the paper cited below.

## Overview

The codebase implements training and fine-tuning workflows for PINN / DeepONet style models for battery electrochemical systems. The primary entry points for running the experiments are the Python scripts under the `Main/` directory. Experiment inputs (datasets) are in `Data/`, trained model checkpoints are stored in `models/`, and evaluation plots and result tables are in `Results/`.

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training example (constant-current training):

```bash
python Main/Solid_Train_CC.py
```

3. Run the transfer-learning / fine-tuning example (uses a pretrained model and fine-tunes on a new target):

```bash
python Main/Solid_FineTune_CC.py
```

## Notes
- Scripts under `Main/` are designed to be run as standalone experiments. They typically load data from `Data/`, train a model, and save outputs (checkpoints, logs, plots) to `models/` and `Results/`. Feel free to modify the scripts to test new configurations, datasets, or model architectures.
- Training outputs (checkpoints, logs) are saved under `models/` by default. Result figures and evaluation metrics are typically written to `Results/` and `eval/`.
- The repository contains a number of evaluation and analysis utilities in `eval/` (for example `evaluation.py`, `elec_eval.ipynb`). Use these to reproduce tables and plots reported in experiments.

## Project structure
- `Main/` — primary training and fine-tuning scripts (entry points). Example files: `Solid_Train_CC.py`, `Solid_FineTune_CC.py`.
- `Data/` — input datasets used for training and evaluation (Excel files and pickles).
- `Results/` — plots and aggregated results exported by experiments.
- `eval/` — evaluation and analysis scripts / notebooks.
- `legacy/` — older scripts and experiments kept for reference.

## Running reproducible experiments

To reproduce a specific experiment, locate the script in `Main/` used to generate it (or the corresponding experiment folder inside `models/`). Many experiments were run with specific hyperparameters and random seeds — when available, these are logged inside the model experiment folder (look for config or metadata files). If you need exact reproduction of a published result, check the experiment folder name (it often encodes parameters and timestamps) in `models/`.

For the paper's main results, the training scripts used were `Solid_Train_CC.py` for the initial training and `Solid_FineTune_CC.py` for the transfer learning experiments, both of them containing the default set of parameters found in the paper. You can run these scripts with the same parameters to reproduce the results, or modify them to test new configurations. Simmilarly, the experimental part of the paper's results can be reproduced by running 'Solid_Train_Exp.py' and 'Solid_FineTune_Exp.py' with the same parameters as the ones used in the paper.

## Evaluation

Scripts under `eval/` are used generate evaluation plots and miscelaneous perform quantitative comparisons. For example, `eval/evaluation.py` and notebooks like `eval/elec_eval.ipynb` contain utilities to load saved checkpoints from `models/` and compute metrics against held-out data saved in `Data/` or `eval/data.pkl`. Overall, most of the files under `eval/` have been generated ad-hoc for specific experiments, so they may require some adaptation to run on new checkpoints or datasets. However, so be wary when running them under your setup or models.

## Citation

If you use this code or the ideas from it, please cite:

Yeregui, Josu; Lopetegi, Iker; Fernandez, Sergio; Garayalde, Erik; Iraola, Unai. "Onsite Estimation of Battery Electrochemical Parameters via Transfer Learning-Based Physics-Informed Neural Network Approach." IEEE Transactions on Industrial Informatics, 2025. DOI: 10.1109/TII.2025.3632935

BibTeX:

```bibtex
@ARTICLE{11268958,
  author={Yeregui, Josu and Lopetegi, Iker and Fernandez, Sergio and Garayalde, Erik and Iraola, Unai},
  journal={IEEE Transactions on Industrial Informatics},
  title={Onsite Estimation of Battery Electrochemical Parameters via Transfer Learning-Based Physics-Informed Neural Network Approach},
  year={2025},
  volume={},
  number={},
  pages={1-11},
  keywords={Mathematical models;Computational modeling;Aging;Electrodes;Computational efficiency;Estimation;Accuracy;Parameter estimation;Adaptation models;Real-time systems;Battery ageing;battery management system (BMS);lithium-ion battery;physical parameter estimation;physics-informed neural networks (PINNs);transfer learning (TL)},
  doi={10.1109/TII.2025.3632935}
}
```

## Contact

For questions about the code or experiments, please open an issue or contact the authors listed in the paper.