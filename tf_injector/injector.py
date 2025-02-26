import tensorflow as tf  # type:ignore
from tensorflow import keras  # type:ignore
from tqdm.auto import tqdm  # type:ignore

import numpy as np
import csv
import shutil

from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Type, Callable, Iterable

from tf_injector.writer import CampaignWriter
from tf_injector.utils import INJECTED_LAYERS_TYPES
from tf_injector.metrics import Metric

FaultType = tuple[str, tuple[int, ...], int]


@dataclass
class FaultList:
    # [("layer", (coords,..), bitpos), ...]
    faults: list[FaultType] = field(default_factory=lambda: [])
    resume_idx: int = 0


class Injector:
    """
    class Injector
    performs fault injections on Tensorflow networks
    """

    def __init__(self, network: keras.Model, dataset: tf.data.Dataset):
        """
        Args:
            network: the target network
            dataset: the dataset on which the inferences are executed
        """
        self.network = network
        self.dataset = dataset
        self.target_layers: dict[str, keras.Layer] = {
            layer.name: layer
            for layer in network._flatten_layers(
                include_self=False, recursive=True
            )  # extracts all layers
            if isinstance(layer, INJECTED_LAYERS_TYPES)
        }
        self.faults = FaultList()
        self.faulty = False

    def load_fault_list(self, fault_path: str, resume_from: int = 0):
        """
        Loads a fault list from a csv file and runs compatibility checks with the network
        Args:
            fault_path: filename of the fault list, must be a csv file
            resume_from: only saves injections starting from its value (default=0)
        """
        self._reset_fault()
        included_layers = set()
        self.faults.resume_idx = resume_from
        with open(fault_path, "r") as f:
            reader = csv.reader(f)
            next(iter(reader))  # skip header
            for row in reader:
                if len(row) != 4:
                    raise ValueError(
                        f"invalid row format: expected 4 columns, got {len(row)}, on row {row}"
                    )
                id, layer, coords, bit = row
                included_layers.add(layer)

                # get coords as a tuple of ints
                int_coords = tuple((int(coord) for coord in coords[1:-1].split(",")))
                if int(id) >= resume_from:
                    self.faults.faults.append((layer, int_coords, int(bit)))

        # all target layers are in injection list
        target_layers = set(self.target_layers.keys())
        assert (
            target_layers == included_layers
        ), f"Fault layers and target layers didn't match: \n \
not included in the fault list: {target_layers-included_layers} \n \
not present in the network: {included_layers-target_layers}"

    @staticmethod
    def _tqdm(iterable, faulty: bool, desc: str) -> tqdm:
        return tqdm(
            iterable,
            colour="red" if faulty else "green",
            desc=desc,
            # ncols=shutil.get_terminal_size().columns,
        )
    
    def validate(self):
        """
        Tests that all the faults target a valid index inside the layer 
        """
        for  index, fault in enumerate(tqdm(self.faults.faults)):
            layer, coords, bitpos = fault
            target_shape = self.target_layers[layer].get_weights()[0].shape
            assert all(map(lambda t: t[0]<t[1], zip(coords, target_shape))), f"ERROR: index overflow at fault {index} {coords} vs {target_shape}"
            assert 0<=bitpos<32

    def _run_inference_on_batch(self, data) -> np.ndarray:
        return self.network(data).numpy()

    def run_inference(self, batch: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs an inference on the target network.
        Args:
            batch: data batch size
        Returns:
            tuple[np.array[dataset_size, n_classes], np.array[dataset_size]]: a tuple
            containing the inference results in the first element and the GT labels
            in the second.
        """
        batched = self.dataset.batch(batch)
        # pbar = self._tqdm(
        #    batched, self.faulty, "Faulty run" if self.faulty else "Clean run"
        # )
        pbar = batched
        batch_predictions: list[np.ndarray] = []
        batch_labels: list[np.ndarray] = []
        for batch in pbar:
            data, label = batch
            batch_predictions.append(self._run_inference_on_batch(data))
            batch_labels.append(label.numpy())

        predictions = np.concatenate(batch_predictions, axis=0)
        labels = np.concatenate(batch_labels, axis=0)
        return predictions, labels

    def run_campaign(
        self,
        batch: int,
        metrics: Iterable[Type[Metric]],
        outputter: CampaignWriter,
        save_scores: bool = False,
        metrics_on_labels: bool = False,
    ):
        """
        Runs a campaign with the loaded fault list
        Params:
            batch: inference batch size
            metric: A Metric object whose metrics will be passed to the outputter
            outputter: CampaignWriter instance
            save_scores: whether save scores as numpy array (default=False)
            inference_function: is the function that computes the outputs fomr the dataset
            inference_function_args: the args to be passed to inference_function
        """
        if not self.faults.faults:
            raise RuntimeError(
                "Attempting to run a campaign without a fault list loaded"
            )

        if not isinstance(metrics, Iterable):
            metrics = [metrics]

        print("running inference")
        gold_scores, labels = self.run_inference(batch)  # clean run
        print("running prediction")
        gold_labels = tf.argmax(gold_scores, axis=1) #, keepdims=True)
        gold_labels = tf.expand_dims(gold_scores, axis=1) # for compatibility with numpy's keepdims argument
        metric_instances = [
                metric(gold_scores, gold_labels, labels) for metric in metrics
        ]

        golden_values = []
        for metric in metric_instances:
            golden_values.extend(metric.clean_output())
        gold_output = (len(labels), *golden_values)
        outputter.write_gold(gold_output)
        if save_scores:
            outputter.save_scores(gold_scores)

        fault_id = self.faults.resume_idx
        pbar = self._tqdm(self.faults.faults[fault_id:], False, "Injection")
        print("Starting campaign...")
        if metrics_on_labels:
            print("NOTE: the metrics on labels are saved **AFTER** the metrics on golden")
        for fault in pbar:
            with self._apply_fault(fault):
                faulty_scores, labels = self.run_inference(batch)
                metric_values = []
                for metric in metric_instances:
                    value = metric.faulty_output(faulty_scores)
                    value = tf.cast(value, tf.double)
                    if value.shape.rank == 1:
                        value = tf.expand_dims(value, 1)
                    metric_values.append(value)

                    if metrics_on_labels:
                        value_label = metric.faulty_output(
                            faulty_scores,
                            with_respect_to_labels=True
                        )
                        value_label = tf.cast(value_label, tf.double)
                        if value_label.shape.rank == 1:
                            value_label = tf.expand_dims(value_label, 1)
                        metric_values.append(value_label)
                
                metric_values = tf.concat(metric_values, axis=1)

                outputter.write_fault(
                    fault_id,
                    len(labels),
                    fault,
                    metric_values,
                )
                if save_scores:
                    outputter.save_scores(faulty_scores, inj_id=fault_id)
            fault_id += 1

    def _reset_fault(self):
        self.faults.faults.clear()
        self.faults.resume_idx = 0

    @contextmanager
    def _apply_fault(self, fault: FaultType):
        """
        Applies a fault to the network. A fault consists in flipping a certain bit
        in a weight of a target layer.
        Usage:
            with self._apply_fault(fault):
                # In this scope the network is faulty
                ...
            # Outside the scope the network is clean
        Args:
            fault: (layer_name, weight_coords, bitpos)
        """
        target_layer_name, coords, bitpos = fault
        target_layer = self.target_layers[target_layer_name]
        bitmask = np.uint32(1) << bitpos
        layer_weights = target_layer.get_weights()
        target_weights = layer_weights[0].view(dtype=np.uint32)
        # clean_value = target_weights[coords] # for assert test
        try:
            target_weights[coords] ^= bitmask
            target_layer.set_weights(layer_weights)
            # assert (target_weights[coords] ^ clean_value) == bitmask
            self.faulty = True
            yield target_weights
        finally:
            target_weights[coords] ^= bitmask
            self.faulty = False
            # assert target_weights[coords] == clean_value
            target_layer.set_weights(layer_weights)
