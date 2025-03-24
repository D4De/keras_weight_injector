# tf_injector
This project contains a tool for injecting faults in the weights of TensorFlow models and used for the experiments supporting the benchmark suite in [dnn-benchmarks](https://gitlab.pmcs2i.ec-lyon.fr/spappala/dnn-benchmarks).

## Project Collaboration

This project has been implemented at [Politecnico di Milano](https://www.polimi.it/) as a part of a collaboration with research teams at [Politecnico di Torino](https://www.polito.it/) and [École Centrale de Lyon](https://www.ec-lyon.fr/).


## Setup
1. Ensure you have Python 3.9 installed in your working environment
2. Create a virtual environment
```
python -m venv venv_name
source venv_name/bin/activate
```
3. Install the dependencies
```
pip install -r requirements.txt
```
4. The pretrained models are available in the repository [dnn-benchmarks](https://gitlab.pmcs2i.ec-lyon.fr/spappala/dnn-benchmarks). Read the repository's `README` for more information.
5. Given a .keras file trained on a dataset, move it to the `models/dataset_name/` folder. For instance, a ResNet18.keras file trained on the CIFAR10 dataset will be placed in `models/CIFAR10/`.

## Tested Models

We have tested the fault injector on the models contained in [dnn-benchmarks](https://gitlab.pmcs2i.ec-lyon.fr/spappala/dnn-benchmarks). In particular:

- CIFAR10
    - DenseNet
    - GoogLeNet
    - Inception
    - MobileNetV2
    - ResNet
    - Vgg
- CIFAR100
    - DenseNet 
    - GoogLeNet
    - ResNet
- GTSRB
    - DenseNet
    - Resnet
    - Vgg
- PascalVOC
    - DeepLabV3  

## Dataset transformation
> [!NOTE]
> All the metrics reported in this `README` refer to the `NumPy` preprocessing displayed below. An equivalent TensorFlow preprocessing is available through the `--use-tf` flag, but it may lead to slightly different results. 

### CIFAR10
```
image = image / np.float32(255.0)
image = (image - (0.4914, 0.4822, 0.4465)) / (0.2023, 0.1994, 0.2010)
```

### CIFAR100
```
image = image / np.float32(255.0)
image = (image - (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)) / (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
```

### GTSRB
```
image = tf.image.resize(image, [50, 50]).numpy()
image = image / np.float32(255.0)
image = (image - (0.3403, 0.3121, 0.3214)) / (0.2724, 0.2608, 0.2669)
```
## Usage
Run as a Python module:
```
python -m tf_injector [ARGS, ...]
```
To display the complete usage guide, type
```
python -m tf_injector --help
```
## Input
To run an injection campaign, you will need:
- A target network saved in `.keras` format, saved in `models/<dataset>/<network>.keras`
- A fault list compatible with the target network in csv format (header needed)

| Injection | Layer  |   TensorIndex   | Bit |
|:---------:|:------:|:---------------:|:---:|
|         0 | conv2d |  "(2, 1, 0, 7)" |  15 |
|         1 | conv2d | "(2, 0, 0, 14)" |   5 |


## Outputs
By default, a summarized report of the injection campaign is stored in `reports/<dataset>/<network>/<dataset>_<network>_<datetime>.csv`
By enabling the `--save-outputs` flag, inference outputs are saved as numpy arrays in the same folder, as `<datetime>/clean.npy` for the clean run, and `<datetime>/inj_<inj_id>.npy` for the faulty runs.

### Output metrics

- Part 1: Injection info

| Injection | Layer  |   TensorIndex   | Bit |
|:---------:|:------:|:---------------:|:---:|
|         0 | conv2d |  "(2, 1, 0, 7)" |  15 |
|         1 | conv2d | "(2, 0, 0, 14)" |   5 |

- Part 2: robustness
    - `top_1_correct`: the label with the maximum score equals the test label (correct inference)
    - `top_5_correct`: the test label is inside the set of the labels which gained the five highest scores
    - `top_1_robust`: Same as top_1_correct, but compared with the predicted labels of the golden inference 
    - `top_5_robust`: Same as top_5_correct, but compared with the predicted labels of the golden inference 


| top_1_correct | top_5_correct | top_1_robust | top_5_robust |
| --------------- | --------------- | --------------- | ------------ |
| 9174 | 9977 | 10000 | 10000 |
| 9174 | 9977 | 10000 | 10000 |

- Part 3: other stats
    - `masked`: Number of dataset inferences that identified the fault as masked.
    - `non_critical`: Number of dataset inferences that identified the fault as non-critical.
    - `critical`: Number of dataset inferences that identified the fault as critical (SDC-1).

| n_injections | masked | non_critical | critical |
|:------------:|:------:|:------------:|:--------:|
|        10000 |  10000 |            0 |        0 |
|        10000 |  10000 |            3 |        0 |

## Tools
Additional tools are placed in the `tools` folder.

### `compute_mean.py`
It prints to stdout the mean of each column of a report file. The `Pandas` library is required.
Usage:
```
python tools/compute_mean.py path/to/report.csv
```

## Testing
The folder `testing` contains testing utilities to validate the result of the injector. Read the dedicated `README` for more information.

## Integrating Additional DNN Models

### Dataset
- in `loaders.py`, add a loading function that returns a TF tensor containing the data and the labels of the dataset. This function should also handle the initialisation of the dataset if not present on the device.
- Register the dataset loader editing the `loaders` variable:
```
loaders = {
    ...
    "dataset-name":loader_function,
}
```
- edit `preprocessing.py` to add a preprocessing function
- Register the preprocessors using numpy in the `np_preprocessors` variable
```
np_preprocessors = {
    ...
    "dataset-name":np_preprocessing_function,
}
```
- Register the preprocessors relying on TensorFlow in the `preprocessors` variable
```
preprocessors = {
    ...
    "dataset-name":tf_preprocessing_function,
}
```
- Register the dataset by editing the variable `SUPPORTED_DATASET` in `utils.py`
```
SUPPORTED_DATASET = [..., "dataset-name"]
```

### Models
- Place the `.keras` file in the `models/dataset-name/` folder
- Register the model by editing the variable `SUPPORTED_MODELS` in `utils.py`
```
SUPPORTED_MODELS = [..., "model-name"]
```

### Metrics
- Add a new metrics system by subclassing `Metric` in `metrics.py`. Read its docstrings for more information
- Add a new metric header in `settings.py`:
```
MY_HEADER = REPORT_HEADER + (
 'metric_name1', ... , 'metric_name_n'
)
```
- Edit lines 5 and 8 in `__main__.py` to import the new header and the new metric:
```
5. from tf_injector.utils import ..., MY_HEADER
8. from tf_injector.metrics import ..., MyMetric
```
- Edit the function `main` in `__main__.py` to implement a selection logic for the headers and the metrics, assigning them to the variables `report_header` `metric` respectively.
