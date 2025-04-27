# Closing the Gap: Achieving Better Accuracy-Robustness Tradeoffs against Query-Based Attacks

This repository contains the Python source code for the experiments of the paper "Closing the Gap: Achieving Better Accuracy-Robustness Tradeoffs against Query-Based Attacks", published in the Proceedings of the AAAI Conference on Artificial Intelligence 2024. Paper linked [here](https://ojs.aaai.org/index.php/AAAI/article/view/30187) and cited below. 

## System and Software Requirements

The experiments in the paper were conducted on a server with the following hardware specifications:

- **CPUs**: 2 × AMD EPYC 7542
- **GPUs**: 2 × NVIDIA A40
- **Memory**: 256 GB RAM

While this hardware setup was used to achieve the reported results, it is **not strictly required** to run the code. Any modern machine with a capable GPU should be sufficient for smaller-scale experiments. It is recommended that the NVIDIA GPU have ≥6 GB VRAM and CUDA 11.7+ support.

The software environment used includes:

- **Operating System**: Ubuntu 22.04
- **Python**: 3.9
- **Anaconda**: 2023.07 or later (recommended for environment management)
- **PyTorch**: 1.13.0
- **CUDA**: 11.7

## Step by Step Guide

1. Verify that your system meets the software requirements and has a capable GPU.
2. Clone the repository.
3. Download the models, adversarial images, and parsed JSON files as described in the following steps.
4. The used models can be downloaded [here](https://ruhr-uni-bochum.sciebo.de/s/R6FGr39LZaqHRPn) and need to be unpacked into the ```models``` folder in the root directory. Follow the directory depiction below.
5. The generated adversarial images and parsed json files can be downloaded [here](https://ruhr-uni-bochum.sciebo.de/s/R6FGr39LZaqHRPn) and need to be unpacked into the root of the project. Then the main results in form of Pareto plots can be generated with the script ```evaluation.py```. Follow the directory depiction below.

## Improvements

### Adaptive Tau Thresholding

We implemented adaptive selection of the confidence threshold (τ) used for test-time defense activation.

Instead of manually specifying a fixed τ value, the threshold is automatically computed based on the model's performance on clean validation data:
- τ is set to a specified percentile (default: 5th percentile) of maximum softmax confidences on the validation set.
- This approach ensures that τ adapts to the model, dataset, and training characteristics.

**Usage:**
- Enable adaptive tau by adding `--adaptive_tau` when launching experiments.
- Control the percentile with `--adaptive_tau_percentile`, if needed (default is 5.0).

Example:
```python main.py -a popskipjump -n 50 -d cifar10 --arch densenet121 --defense rnd --epsilon 3.0 -q 15000 -m attack --adaptive_tau```



## Experiments 

### Conda environment setup

To set up the environment ```query_based_framework``` with all required packages just run:

```
conda env create -f environment.yml
```

### Structure

```
.
├── attacks // attack implementations
├── configs
│   ├── main.json
│   ├── runall.json
│   └── surfree.json
├── dataset_helper.py
├── datasets
├── defenses // defense implementations
├── environment.yml
├── main.py // main script for executing a specified experiment
├── evaluation.py // script for generting Pareto plots out of experimental results
├── model_factory.py
├── model_interface.py
├── results // output files (download link above)
├── models // contains models and special architectures (download link above)
│   ├── calibration
│   │   └── temperature_scaling.py
│   ├── cifar10
│   │   ├── densenet121.pt
│   │   ├── resnet20_pni
│   │   ├── resnet34_oat
│   │   ├── robust_training
│   │   └── vgg16_rse
│   ├── cifar100
│   │   └── resnet50.pt
│   ├── densenet.py
│   └── resnet.py
└── runall.py // wrapper that calls main.py with various parameters
```

### Launch experiments
The experiments are launched with the wrapper script ```runall.py```, which essentially constructs a list of tasks from the main.json file and then dispatches them.

It is parameterized as follows:

```
usage: runall.py [-h] [--gpus GPUS] [--attack {popskipjump,surfree}]
                 [--attacks_per_gpu ATTACKS_PER_GPU] [--seed SEED]
                 [--config_path CONFIG_PATH] [--output_folder OUTPUT_FOLDER]
                 [--output | --no-output] [--evaluate | --no-evaluate]
                 [--override | --no-override]
                 [--calibration | --no-calibration]
                 [--adaptive_tau | --no-adaptive_tau] [--adaptive_tau_percentile ADAPTIVE_TAU_PERCENTILE]

optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS, -g GPUS  Comma separated list of GPUs to use. Sorted by
                        PCI_BUS_ID.
  --attack {popskipjump,surfree}, -a {popskipjump,surfree}
                        Attack to evaluate.
  --attacks_per_gpu ATTACKS_PER_GPU, -p ATTACKS_PER_GPU
                        Number of attacks to be run on a single GPU.
  --seed SEED           Seed to randomize each attack.
  --config_path CONFIG_PATH
  --output_folder OUTPUT_FOLDER
                        Specifies the output path.
  --output, --no-output
                        Just outputs all command lines to be executed without
                        actually starting them. (default: False)
  --evaluate, --no-evaluate
                        Only performs an evaluation of experimental results.
                        (default: False)
  --override, --no-override
                        Overrides existing experiments/results. (default:
                        False)
  --calibration, --no-calibration
                        Applies calibration to the output of the model.
                        (default: True)
  --adaptive_tau, --no-adaptive_tau
                        Enables adaptive selection of the confidence threshold tau based on validation set.
                        (default: False)
  --adaptive_tau_percentile ADAPTIVE_TAU_PERCENTILE
                        Percentile of validation confidences to set tau.
                        (default: 5.0)
```

## Citation

```
@article{Zimmer_Andreina_Marson_Karame_2024, 
    title={Closing the Gap: Achieving Better Accuracy-Robustness Tradeoffs against Query-Based Attacks}, 
    volume={38}, url={https://ojs.aaai.org/index.php/AAAI/article/view/30187}, 
    DOI={10.1609/aaai.v38i19.30187}, 
    number={19}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Zimmer, Pascal and Andreina, Sébastien and Marson, Giorgia Azzurra and Karame, Ghassan}, 
    year={2024}, 
    month={Mar.}, 
    pages={21859-21868} 
}
```

## Contact

Feel free to contact the first author via the e-mail provided on the publication and author of the proposed improvement via mkrusz@uri.edu.
