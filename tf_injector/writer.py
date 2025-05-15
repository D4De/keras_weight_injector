import numpy as np
import csv
import os

from datetime import datetime

from gg.utils import DEFAULT_REPORT_DIR

from typing import Optional


class CampaignWriter:
    """
    class CampaignWriter
    writes the results of an injection campaign to a csv file
    result path: file_dir/dataset/network/
    """

    def __init__(
        self,
        dataset: str, 
        network: str,
        report_header: tuple[str,...],
        file_dir: os.PathLike = DEFAULT_REPORT_DIR,
        one_line_per_input = False,
    ):
        target_dir = os.path.join(file_dir, str(dataset), network)
        os.makedirs(target_dir, exist_ok=True)
        self.time = datetime.now().strftime("%y%m%d_%H%M")
        self.filepath = os.path.join(
            target_dir, self.get_filename(dataset, network, self.time)
        )
        self.report_header = report_header
        self.one_line_per_input = one_line_per_input

    def __enter__(self) -> "CampaignWriter":
        write_header = not os.path.exists(self.filepath)
        self.file = open(self.filepath, "a")
        # TODO: Use pandas rather than csv
        # The reason behind using pandas is that some metrics
        # (like IOU, a.k.a. jaccard index) have a lot of nans
        # (not a numbers) in the output. With pandas they are
        # treated like empty bytes, which saves lots of space
        # and, apparently isn't possibile to do the same with
        # the library csv.
        # pandas has the option of appending to file.
        self.writer = csv.writer(self.file)
        if write_header:
            self.writer.writerow(self.report_header)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    @staticmethod
    def get_filename(dataset: str, network: str, time: str) -> str:
        return f"{str(dataset)}_{network}_{time}.csv"

    def get_report_folder(self) -> str:
        report_folder_p = os.path.dirname(self.filepath)
        report_folder = os.path.join(report_folder_p, self.time)
        return report_folder

    def write_gold(self, gold_row: tuple[int,...]):
        padding = [None]
        repeat = 3 if not self.one_line_per_input else 4
        row = ("GOLDEN", *(padding * repeat), *gold_row)
        self.writer.writerow(row)

    def write_fault(
        self,
        fault_id: int,
        num_injections: int,
        fault: tuple[int, ...],
        fault_metrics: tuple[int, ...],
    ):
       
        """ fault_metrics_str = []
        for row in fault_metrics:
            newRow = list( filter( 
                lambda x: None if 'nan' in x else x, # filter out nans and subsitute with empty strings
                [ str(value) for value in row.numpy().tolist() ] # convert the numpy values to str
            ))
            fault_metrics_str.append(newRow) """
        
        if not self.one_line_per_input:
            # patch: fault_metrics is a 2D tf.Tensor of float -> convert to a list of integers
            # TODO : why 32 bits?
            #fault_metrics = list(fault_metrics.numpy().T[0].astype(np.uint32))
            row = (fault_id, *fault, num_injections, *fault_metrics)
            self.writer.writerow(row) 
        else:
            rows = []
            for i in range(num_injections):
                rows.append([
                    fault_id,
                    *fault,
                    i,
                    num_injections,
                    *[ *fault_metrics[i].numpy().tolist() ], # *fault_metrics_str[i],
                ])
            self.writer.writerows(rows)


    def save_scores(self, scores: np.ndarray, inj_id: Optional[int] = None):
        target_path = self.get_report_folder() + os.path.sep
        os.makedirs(target_path, exist_ok=True)
        if inj_id is None:
            target_path += "clean.npy"
        else:
            target_path += f"inj_{inj_id}.npy"
        np.save(target_path, scores)
