import os
from typing import List
import tensorflow as tf
import importlib.util
import json

# Definisci la directory per i plugin utente
USER_PLUGINS_DIR = os.path.join(os.path.dirname(__file__), 'user')

class Campaign:
    def __init__(
        self,
        dataset_name : str, # () -> tf.data.Dataset
        network_path : str,
        fautl_list_path : str,
        preporcessig: callable = lambda x: x,   # (tf.data.Dataset) -> tf.data.Dataset
        metrics: List[str] = None,
    ):
    
        self.dataset_name = dataset_name
        self.network_path = network_path
        self.dataset = self.__load_dataset() # tf.data.Dataset
        self.network = self.__load_network()
        self.fault_list_path = None
        self.preprocessig = preporcessig
        self.metrics = metrics if metrics is not None else []
        self.injector = None

    def __load_dataset(self) -> tf.data.Dataset :
        dataset_name = self.dataset_name


        # find python file corresponded to the dataset name
        base_path = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(base_path, "loaders", dataset_name)

        if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Dataset {dataset_name} is not loaded")
    
        files = [
            f for f in os.listdir(target_dir) 
             if os.path.isfile(os.path.join(target_dir, f))
             and f.endswith(".py")
            ]
        file_path = os.path.join(target_dir, files[0])


        # get loading function name from data.json
        config_file_path = os.path.join(target_dir, "config.json")
        with open(config_file_path, 'r') as file:
            config_data = json.load(file)
        func_name = config_data["loader_name"]
        

        # import the module
        spec = importlib.util.spec_from_file_location(f"loaders.{dataset_name}", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loader_function = getattr(module, func_name)


        # check if the function returns a tf.data.Dataset
        dataset = loader_function()
        if not isinstance(dataset, tf.data.Dataset):
            raise TypeError(f"La funzione {func_name} non ha restituito un tf.data.Dataset")
        return dataset
    
    def __load_network(self) -> tf.keras.Model :

        # check if the network path is a file .keras or .h5
        if not (self.network_path.endswith(".keras") or self.network_path.endswith(".h5")):
            raise ValueError(f"The --model must be a file .keras or .h5 : {self.network_path}")

        return tf.keras.models.load_model(self.network_path)
    
    def run(self):
        print("-------------------------------------------------------------")
        print("Running campaign...")
        print("-------------------------------------------------------------")
        return
    
    