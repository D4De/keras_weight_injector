# Deep Neural Network Models for Reliability Studies
Welcome to the repository containing state-of-the-art Deep Neural Network (DNN) models implemented in both PyTorch and TensorFlow for conducting reliability studies. 

## Project Collaboration

This project is a collaboration between the following institutions:

- [Politecnico di Torino](https://www.polito.it/)
- [Politecnico di Milano](https://www.polimi.it/)
- [Ecole Centrale de Lyon](https://www.ec-lyon.fr/en)

## Installation

### Injections with PyTorch

IGNORE

### Injections with TensorFlow

1. Create a virtual environment

```
python -m venv .venv
```

2. Activate the environment

```
source .venv/bin/activate
```

3. Install the dependencies from the requirements
You can find a requirements.txt from which you can install all dependencies using

```
pip install -r requirements.txt
```

4. Install PyTorch for CPU. (We need PyTorch only for dataloading and for common operations, so GPU support is not needed and may create
additional problems)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
## Pretrained models
Pretrained weights are stored inside the `benchmark_models/models/pretrained_models` folder.

### Torch weights
Given a Torch model (for instance ResNet20 trained for CIFAR10), move the weights to `benchmark_models/models/pretrained_models/CIFAR10/ResNet20.pt`.

In general, a Torch model called M for a dataset D will be seeked by the injector in the path `benchmark_models/models/pretrained_models/D/M.pt`.
Several Torch extension are supported (pt, th, ...).

### Keras weights
Given a Keras model (for instance ResNet20 for CIFAR10), place the weights inside `benchmark_models/models/converted-tf/CIFAR10/ResNet20.pt`.

In general, a Keras model called M for a dataset D will be seeked by the injector in the path `benchmark_models/models/converted_tf/D/M.keras`.

## Getting Started

The following command is an example of execution of the fault injector:
```
python tf_weight_injection.py -d CIFAR10 -n ResNet32 -f ../../fault_lists/CIFAR10/Resnet32_FL.csv
```
This command has to be executed in the folder benchmark_models/tf_injector.
In this example we execute the campaign on ResNet32 elaborating on CIFAR10 by using the specified fault list.



