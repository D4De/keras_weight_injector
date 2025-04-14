from writer import CampaignWriter
from metrics import Metric
from loader import load_network
from writer import CampaignWriter
from injector import Injector
import yaml
import importlib
import os
from typing import List, Callable, Any, Optional
import argparse
import plugin
import tensorflow as tf

# Definisci la directory per i plugin utente
USER_PLUGINS_DIR = os.path.join(os.path.dirname(__file__), 'user')

class Campaign:
    def __init__(
        self,
        dataset_name : str,
        network_path : str,
        fautl_list_path : str,
        loader: Callable = None,  # () -> tf.data.Dataset
        preporcessig: callable = lambda x: x,   # (tf.data.Dataset) -> tf.data.Dataset
        metrics: List[str] = None,
    ):
    
        self.dataset_name = dataset_name
        self.network_path = network_path
        self.fault_list_path = fault_list_path
        self.loader = loader
        self.preprocessig = preporcessig
        self.metrics = metrics if metrics is not None else []
        self.network = None
        self.dataset = None
        self.injector = None