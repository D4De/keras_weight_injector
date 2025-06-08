# tf_injector
This project contains a tool for injecting faults in the weights of TensorFlow models and used for the experiments supporting the benchmark suite in [dnn-benchmarks](https://github.com/ReADLBench/dnn-benchmark).

## Project Collaboration

This project has been implemented at [Politecnico di Milano](https://www.polimi.it/) as a part of a collaboration with research teams at [Politecnico di Torino](https://www.polito.it/) and [École Centrale de Lyon](https://www.ec-lyon.fr/).


## Project Structure
- **dataset_loaders**: scripts to load the dataset form a folder to the tensorflow structure `tensorflow.data.Dataset used` for inference.
- **metrics** : scripts to compute the metrics from the labels and the network output.
- **templates** : template files for implementing new datasets or metrics
- **testing** : tool to spot differencese bethween two outout of the tool. Useful to mantain consistency when updating the code.
- **tf_injector** : code of the injector
- **tools** : support tools



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
> For `CIFAR10`, `CIFAR100` and `GTSRB` both tenforflow and numpy preprocessing are avaleable. Use `--dataset DATASET` to use the preprocess with tenforflow and `--dataset DATASETnp` for numpy

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
python -m tf_injector run [ARGS, ...]
```
To display the complete usage guide, type
```
python -m tf_injector run --help
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
By default, a summarized report of the injection campaign is stored in `out/<dataset>/<network>/<dataset>_<network>_<datetime>.csv`
By enabling the `--save-outputs` flag, inference outputs are saved as numpy arrays in the same folder, as `<datetime>/clean.npy` for the clean run, and `<datetime>/inj_<inj_id>.npy` for the faulty runs.

### Output metrics

#### Part 1: Injection info

| Injection | Layer  |   TensorIndex   | Bit |
|:---------:|:------:|:---------------:|:---:|
|         0 | conv2d |  "(2, 1, 0, 7)" |  15 |
|         1 | conv2d | "(2, 0, 0, 14)" |   5 |

#### Part 2: metrics specifics columns

- `ImageClassificationMetric` for image classification

    - `top_1_correct`: the label with the maximum score equals the test label (correct inference)
    - `top_5_correct`: the test label is inside the set of the labels which gained the five highest scores
    - `top_1_robust`: Same as top_1_correct, but compared with the predicted labels of the golden inference 
    - `top_5_robust`: Same as top_5_correct, but compared with the predicted labels of the golden inference 

    - `masked`: Number of dataset inferences that identified the fault as masked.
    - `non_critical`: Number of dataset inferences that identified the fault as non-critical.
    - `critical`: Number of dataset inferences that identified the fault as critical (SDC-1).

| top_1_correct | top_5_correct | top_1_robust | top_5_robust | masked | non_critical | critical |
|:-------------:|:-------------:|:------------:|:------------:|:------:|:------------:|:--------:|
| 9174          | 9977          | 10000        | 10000        | 10000  | 0            | 0        |
| 9174          | 9977          | 10000        | 10000        | 10000  | 3            | 0        |

- `PixelAccuracy` for image segmentation

    - `pixel_accuracy_on_label`: percentage of pixels correctly classified compared to the reference label
    - `pixel_accuracy_on_golden`: percentage of pixels classified consistently with the golden inference (robustness)

| pixel_accuracy_on_label | pixel_accuracy_on_golden |
|:-----------------------:|:------------------------:|
| 0.8923                  | 0.9987                   |
| 0.8923                  | 0.9462                   |

- `ImageIntersectionOverUnion` for image segmentation

    - `IOU_on_label_i`: Intersection over Union for class `{i}` between prediction and reference label
    - `IOU_on_golden_i`: Intersection over Union for class `{i}` between prediction and golden inference
    
    Where `i` identifies the class 

Mathematically, for each class, IoU is defined as:
    
    IoU = (area of overlap) / (area of union) = (TP) / (TP + FP + FN)
    
    Where TP = true positives, FP = false positives, FN = false negatives


| IOU_on_label_0 | IOU_on_label_1 | ... | IOU_on_label_20 | IOU_on_golden_0 | IOU_on_golden_1 | ... | IOU_on_golden_20 |
|:--------------:|:--------------:|:---:|:---------------:|:---------------:|:---------------:|:---:|:----------------:|
| 0.8651         | 0.7123         | ... | 0.6987          | 0.9991          | 0.9845          | ... | 0.9912           |
| 0.8651         | 0.7123         | ... | 0.6987          | 0.8762          | 0.9124          | ... | 0.8992           |

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

### Models and Fault Lists
The `python -m tf_injector run [args]` command of the tf_injector module supports external models and fault lists. The `--model` parameter requires the absolute path to a **.keras** file, while the `--fault_list` parameter requires the absolute path to a **.csv** file formatted according to the input requirements.
```
python -m tf_injector run \
--model /PATH_TO_MODEL/model.keras
--fault_list  /PATH_TO_LIST/fl.csv
[... other ARGS]
```

### Dataset
To add a new dataset create in your local machine a python file structured as `template/Dataset_template.py`.
```
# new_DATSASET_LOADER.py
import tensorflow as tf

def load(DIRPATH : str) -> tf.data.Dataset :
    [...]
```
The purpose of this file is to create a function that traverses through the original directory structure of the dataset (since each dataset has its own unique organization with no universal pattern) and converts it into a TensorFlow object suitable for model inference. The dataset images, typically stored as .png or .jpg files, need to be transformed into tensors of the appropriate dimensions for the model input, preprocessed if required, and organized into a tensorflow.data.Dataset structure in the format of (img1, label1), (img2, label2), [...]. The code can be organized into multiple subroutines as needed, as long as you specify the main entry point function in the --function_name parameter.

Then the file can be loaded into the injector running this command:
```
python -m tf_injector lddataset \
--dataset_name new_DATASET \
--path_to_function /PATH_TO_FILE_DIR/new_DATASET_LOADER.py
--function_name load
```

When running a campaign with `python -m tf_injector run [args]`, you can reference the new dataset using the `--dataset` parameter to specify the dataset name and the `--dataset_path` parameter to indicate the root directory containing the dataset images, if necessary.

> [!NOTE]
> if a dataset with such name is already present, it gets swapped with the latter one. Check the current avaliable dataset with the `python -m tf_injector run --help` command.

### Metrics
To add a new mectric create in your local machine a python file structured as `template/Metric_template.py`. The filename must match the name of the class contained within the file.

Run the following command to load the new metric into the injector.
```
python -m tf_injector ldmetric --path /PATH_TO_PYTHON_FILE
```

To use the metric in a campaign, specify it in the `--metric` parameter when running `python -m tf_injector run [args]`, using the name of the implementing class.

## Reproduce experiments
Go to the repository on https://gitlab.pmcs2i.ec-lyon.fr/spappala/dnn-benchmarks. Download the folder `tensorflow` and copy into tf_injector. It contains models and fault lists for each dataset. 

### CIFAR10

- DenseNet121
```
python -m tf_injector run \
--dataset CIFAR10 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR10/fp32/densenet/DenseNet121_TF_FL.csv \
--model tensorflow/gpu/image_classification/CIFAR10/fp32/densenet/DenseNet121.keras \
--batch 2048
```

- DenseNet161
```
python -m tf_injector run \
--dataset CIFAR10 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR10/fp32/densenet/DenseNet161_TF_FL.csv \
--model tensorflow/gpu/image_classification/CIFAR10/fp32/densenet/DenseNet161.keras \
--batch 2048
```

- GoogleNet
```
--dataset CIFAR10 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR10/fp32/googlenet/googlenet_cifar10_TF_FL.csv\
--model tensorflow/gpu/image_classification/CIFAR10/fp32/googlenet/GoogLeNet.keras \
--batch 2048
```

- MobileNetV2
```
--dataset CIFAR10 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR10/fp32/mobilenet/mobilenetv2_cifar10_TF_FL.csv\
--model tensorflow/gpu/image_classification/CIFAR10/fp32/mobilenet/MobileNetV2.keras \
--batch  2048
```

- ResNet20
```
--dataset CIFAR10 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR10/fp32/resnet/ResNet20_TF_FL.csv\
--model tensorflow/gpu/image_classification/CIFAR10/fp32/resnet/ResNet20.keras \
--batch  2048
```

- ResNet32
```
--dataset CIFAR10 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR10/fp32/resnet/ResNet32_TF_FL.csv\
--model tensorflow/gpu/image_classification/CIFAR10/fp32/resnet/ResNet32.keras \
--batch  2048
```

- ResNet44
```
--dataset CIFAR10 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR10/fp32/resnet/ResNet44_TF_FL.csv\
--model tensorflow/gpu/image_classification/CIFAR10/fp32/resnet/ResNet44.keras \
--batch  2048
```

- Vgg11
```
--dataset CIFAR10 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR10/fp32/vgg/Vgg11_bn_TF_FL.csv\
--model tensorflow/gpu/image_classification/CIFAR10/fp32/vgg/Vgg11_bn.keras \
--batch  2048
```

- Vgg13
```
--dataset CIFAR10 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR10/fp32/vgg/Vgg13_bn_TF_FL.csv\
--model tensorflow/gpu/image_classification/CIFAR10/fp32/vgg/Vgg13_bn.keras \
--batch  2048
```

### CIFAR100

- DenseNet121
```
python -m tf_injector run \
--dataset CIFAR100 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR100/fp32/densenet/DenseNet121_TF_FL.csv \
--model tensorflow/gpu/image_classification/CIFAR100/fp32/densenet/DenseNet121.keras \
--batch 2048
```

- GoogleNet
```
--dataset CIFAR100 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR100/fp32/googlenet/googlenet_cifar100_TF_FL.csv\
--model tensorflow/gpu/image_classification/CIFAR100/fp32/googlenet/GoogLeNet.keras \
--batch 2048
```

- ResNet18
```
--dataset CIFAR100 \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/CIFAR100/fp32/resnet/ResNet18_TF_FL.csv\
--model tensorflow/gpu/image_classification/CIFAR100/fp32/resnet/ResNet18.keras \
--batch  2048
```

### GTSRB

- DenseNet121
```
python -m tf_injector run \
--dataset GTSRB \
--metrics ImageClassificationMetric \
--fault_list tensorflow/gpu/image_classification/GTSRB/fp32/densenet/DenseNet121_TF_FL.csv \
--model tensorflow/gpu/image_classification/GTSRB/fp32/densenet/DenseNet121.keras \
--batch 2048
```

### PascalVOC

1. Download the VocDataset

2. Run this prompt
```
 python -m tf_injector run \
--dataset PascalVOC \
--dataset_path PATH_TO_VOCdevkit/VOC2012 \
--metrics PixelAccuracy,ImageIntersectionOverUnion \
--fault_list tensorflow/gpu/image_segmentation/PascalVOC/fp32/DeepLabV3/DeepLabV3_ResNet50_TF_FL.csv
--model tensorflow/gpu/image_segmentation/PascalVOC/fp32/DeepLabV3/DeepLabV3_ResNet50.keras
--postprocess "lambda x : x[0]" \
--batch 2048
```

> [!NOTE]
> reports can be found in `tf_injector/out` if --out parameter is not specified.