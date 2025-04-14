import os
from typing import List
import tensorflow as tf

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
        self.dataset = self.__load_dataset()
        self.network = self.__load_network()
        self.fault_list_path = None
        self.preprocessig = preporcessig
        self.metrics = metrics if metrics is not None else []
        self.injector = None

    def __load_dataset(self):
        dataset_name = self.dataset_name
        func_name = self.dataset_loading_func_name

        # run the func_name into the loaders/dataset_name dir
    
        # check if the output is (tf.data.Dataset)


        return
    
    def __load_network(self):
        return