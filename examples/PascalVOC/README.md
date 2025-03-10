# DeepLAB v3 fault injection example

The file `main.py` shows a real example of fault injection on a DeepLABv3 model. 

It is executed on the PASCALVoc Dataset (avilable [here](http://host.robots.ox.ac.uk/pascal/VOC/index.html)). 
To prepare the dataset, we provide the script `./setup.sh`. It will simply download the dataset and uncompress it. 

The model was got from [dnn-benchmarks](https://gitlab.pmcs2i.ec-lyon.fr/spappala/dnn-benchmarks) and it was generated using [nobuco](https://github.com/AlexanderLutsenko/nobuco) with the script available [here](https://github.com/D4De/dnn-benchmarks-converter.git)
