import os
from typing import List
import tensorflow as tf
from tensorflow import keras
import importlib.util
import json
from tf_injector.injector import Injector
from tf_injector.faultlist import FaultList
from tf_injector.writer import CampaignWriter
from tf_injector.new_metrics.metric import Metric


class Campaign:
    def __init__(
        self,
        dataset_name : str, # () -> tf.data.Dataset
        network_path : str,
        fautl_list_path : str,
        output_path : str,
        preporcessig: callable = lambda x: x,   # (tf.data.Dataset) -> tf.data.Dataset
        metrics_list: List[str] = None,
    ):
    
        self.dataset_name = dataset_name
        self.network_path = network_path
        self.dataset = self.__load_dataset() # tf.data.Dataset
        self.network = self.__load_network()

        # load fault list
        self.fault_list = FaultList()
        included_layers = self.fault_list.load_from_csv(fautl_list_path)
        self.__validate_fault_list(included_layers)

        self.output_path = output_path
        self.preprocessig = preporcessig
        self.metrics_list = metrics_list
        self.metrics = self.__load_metrics()

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
    
    def __validate_fault_list(self, included_layers : set[str]) -> None:
        
        INJECTED_LAYERS_TYPES = (keras.layers.Conv2D, keras.layers.Dense)

        target_layers: dict[str, keras.Layer] = {
            layer.name: layer
            for layer in self.network._flatten_layers(
                include_self=False, 
                recursive=True
            )  # extracts all layers
            if isinstance(layer, INJECTED_LAYERS_TYPES)
        }

        target_layers = set(target_layers.keys())

        if target_layers != included_layers:
            not_in_fault_list = target_layers - included_layers
            not_in_network = included_layers - target_layers
            if len(not_in_fault_list) > 0:
                print(f"WARNING: Some layers are not included in the fault list: {not_in_fault_list}")

            assert (
                len(not_in_network) == 0
            ), f"Fault layers and target layers didn't match: \n \
            some layers are not present in the network: {included_layers-target_layers}"

    def __load_metrics(self) -> List[type[Metric]]:

        metrics = []
        base_path = os.path.dirname(os.path.abspath(__file__))
        metrics_dir = os.path.join(base_path, "new_metrics")
        files = os.listdir(metrics_dir)
        metrics_available = [f for f in files if f.endswith('.py')]

        for metric in self.metrics_list:
            if f"{metric}.py" in metrics_available:
                path = os.path.join(metrics_dir, f"{metric}.py")

                # import module
                spec = importlib.util.spec_from_file_location(metric, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # import class
                metric_class = getattr(module, metric)

                # create instance of the class
                #metric_object = metric_class()

                # add class to the list
                metrics.append(metric_class)
            else:
                print(f"WARNING: Metric ({metric}) not found")
            
        return metrics
    


    def run(self):
        print("-------------------------------------------------------------")
        print("Running campaign...")

        injector = Injector(
            network = self.network,
            dataset = self.dataset,
            faluts = self.fault_list
        )

        network_name = os.path.basename(self.network_path)
        cw = CampaignWriter(
            dataset = self.dataset_name, 
            network = network_name, 
            report_header= (
                "col1",
                "col2",
                "col3",
                "col4",
                "col5",
                "col6",
                "col7"
            ),
            file_dir = self.output_path
            )
        
        with cw :
            injector.run_campaign(
                batch = 32,
                metrics = self.metrics,
                outputter = cw,
                save_scores = True,
                metrics_on_labels = True,
            )
        print("-------------------------------------------------------------")
        return


    def debug(self):
        for metric in self.metrics:
            metric.chiSono()
        return
    