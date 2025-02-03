import tensorflow as tf  # type:ignore
from tensorflow import keras  # type:ignore
from tqdm import tqdm  # type:ignore

import numpy as np
import csv
import shutil

from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Callable, Optional

from tf_injector.writer import CampaignWriter
from tf_injector.utils import INJECTED_LAYERS_TYPES

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
            dataset: the dataset on which the inferences are run
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
            ncols=shutil.get_terminal_size().columns,
        )

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
        batch: int = 64,
        save_scores: bool = False,
        gold_row_metric: Optional[Callable] = None,
        faulty_row_metric_maker: Optional[Callable[..., Callable]] = None,
        outputter: Optional[CampaignWriter] = None,
    ):
        """
        Runs a campaign with the loaded fault list
        Params:
            batch: inference batch size (default=64)
            save_scores: whether save scores as numpy array (default=False)
            gold_row_metric: a callable which computes metrics on the gold row.
            faulty_row_metric_maker: a callable which build a metric function for
                faults using gold scores and labels
            outputter: CampaignWriter instance
        """
        if not self.faults.faults:
            raise RuntimeError(
                "Attempting to run a campaign without a fault list loaded"
            )
        gold_scores, labels = self.run_inference(batch)  # clean run
        gold_labels = gold_scores.argmax(axis=1, keepdims=True)

        if gold_row_metric:
            gold_output = (len(labels), *gold_row_metric(gold_scores, labels))
            if outputter:
                outputter.write_gold(gold_output)
                if save_scores:
                    outputter.save_scores(gold_scores)

        if faulty_row_metric_maker:
            faulty_row_metric = faulty_row_metric_maker(gold_scores, gold_labels)
        else:
            faulty_row_metric = None

        fault_id = self.faults.resume_idx
        pbar = self._tqdm(self.faults.faults[fault_id:], False, "Injection")
        for fault in pbar:
            with self._apply_fault(fault):
                faulty_scores, labels = self.run_inference(batch)
                if faulty_row_metric:
                    if outputter:
                        outputter.write_fault(
                            fault_id,
                            len(labels),
                            fault,
                            faulty_row_metric(faulty_scores, labels),
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
