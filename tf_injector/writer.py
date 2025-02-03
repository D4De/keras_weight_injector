import numpy as np
import csv
import os

from datetime import datetime

from tf_injector.utils import REPORT_HEADER, DEFAULT_REPORT_DIR

from typing import Optional


class CampaignWriter:
    """
    class CampaignWriter
    writes the results of an injection campaign to a csv file
    result path: file_dir/dataset/network/
    """

    def __init__(
        self, dataset: str, network: str, file_dir: os.PathLike = DEFAULT_REPORT_DIR
    ):
        target_dir = os.path.join(file_dir, dataset, network)
        os.makedirs(target_dir, exist_ok=True)
        self.time = datetime.now().strftime("%y%m%d_%H%M")
        self.filepath = os.path.join(
            target_dir, self.get_filename(dataset, network, self.time)
        )

    def __enter__(self) -> "CampaignWriter":
        write_header = not os.path.exists(self.filepath)
        self.file = open(self.filepath, "a")
        self.writer = csv.writer(self.file)
        if write_header:
            self.writer.writerow(REPORT_HEADER)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    @staticmethod
    def get_filename(dataset: str, network: str, time: str) -> str:
        return f"{dataset}_{network}_{time}.csv"

    def get_report_folder(self) -> str:
        report_folder_p = os.path.dirname(self.filepath)
        report_folder = os.path.join(report_folder_p, self.time)
        return report_folder

    def write_gold(self, gold_row: tuple[int, int, int]):
        padding = [None]
        row = ("GOLDEN", *(padding * 3), *gold_row, *(padding * 4))
        self.writer.writerow(row)

    def write_fault(
        self,
        fault_id: int,
        num_injections: int,
        fault: tuple[int, ...],
        fault_metrics: tuple[int, ...],
    ):
        row = (fault_id, *fault, num_injections, *fault_metrics)
        self.writer.writerow(row)

    def save_scores(self, scores: np.ndarray, inj_id: Optional[int] = None):
        target_path = self.get_report_folder() + os.path.sep
        os.makedirs(target_path, exist_ok=True)
        if inj_id is None:
            target_path += "clean.npy"
        else:
            target_path += f"inj_{inj_id}.npy"
        np.save(target_path, scores)
