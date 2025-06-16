from typing import List
import tensorflow as tf
from tensorflow import keras
import importlib.util
import json
from tf_injector.injector import Injector
from tf_injector.faultlist import FaultList
from tf_injector.writer import CampaignWriter


# load metrics from the metrics package
import sys
import os

original_path = sys.path.copy()
metric_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    , "metrics"
)
print(f"Loading metrics from {metric_dir}")
sys.path.insert(0, metric_dir)
from metric import Metric
sys.path = original_path


class Campaign:
    def __init__(
        self,
        dataset_name : str, # () -> tf.data.Dataset
        dataset_path : str,
        network_path : str,
        fautl_list_path : str,
        output_path : str,
        preporcessig: callable = lambda x: x,   # (tf.data.Dataset) -> tf.data.Dataset
        metrics_list: List[str] = None,
        batch_size: int = 32,
        save_scores: bool = False,
        resume_from: int = 0,
        validate_fault_list: bool = True,
        seed : int = None,
        pickle : bool = False, # if True, save the results in a pickle file
    ):
    
        self.dataset_name = dataset_name
        self.network_path = network_path
        self.dataset_path = dataset_path
        self.dataset = self.__load_dataset(dataset_path) # tf.data.Dataset
        self.network = self.__load_network()

        # load fault list
        self.fault_list = FaultList()
        included_layers = self.fault_list.load_from_csv(fautl_list_path, resume_from)
        self.__validate_fault_list(included_layers)


        self.output_path = output_path
        self.preprocessig = preporcessig
        self.metrics_list = metrics_list
        self.metrics = self.__load_metrics()

        self.num_classes = self.__get_number_of_classes()
        self.batch_size = batch_size
        self.save_scores = save_scores
        self.validate_fault_list = validate_fault_list
        self.seed = seed
        self.pickle = pickle

        # tranfrom output function
        try:
            self.transform_output = eval(self.preprocessig)
        except Exception as e:
            print(f"ERROR: lambda expression {e} impossible to tranfsorm")
            print("Set to default labda expression 'lambda x: x'")
            self.transform_output = lambda x: x


    def __load_dataset(self, path) -> tf.data.Dataset :
        dataset_name = self.dataset_name


        # find python file corresponded to the dataset name
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_dir = os.path.join(base_path, "dataset_loaders", dataset_name)

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
        dataset = loader_function(path)
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
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metrics_dir = os.path.join(base_path, "metrics")
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

    def __get_number_of_classes(self) -> int:
        """
        Get the number of classes from the last layer of the network
        """
        last_layer = self.network.layers[-1]
        last_layer_shape = last_layer.output_shape
        num_classes = last_layer_shape[-1]
        return num_classes

    def run(self):
        print("-------------------------------------------------------------")
        print("Running campaign...")

        injector = Injector(
            network = self.network,
            dataset = self.dataset,
            faults = self.fault_list,
            transform_output = self.transform_output,
            seed= self.seed
        )
        
        if (self.validate_fault_list):
            print(f"Faul list validation...")
            injector.validate()

        network_name = os.path.basename(self.network_path)

        header = (
            "inj_id",
            "target_layer",
            "layer_weigths",
            "bit_pos",
            "n_injections",
            )
        
        metrics_header = tuple()

        for metric in self.metrics:
            metrics_header += metric(None, None, self.num_classes).get_header()

        header += metrics_header

        cw = CampaignWriter(
            dataset = self.dataset_name, 
            network = network_name, 
            report_header = header,
            file_dir = self.output_path,
            one_line_per_input= False,
            pickle = self.pickle
            )
        
        with cw :
            injector.run_campaign(
                batch = self.batch_size,
                metrics = self.metrics,
                outputter = cw,
                save_scores = self.save_scores
            )
        print("-------------------------------------------------------------")
        return