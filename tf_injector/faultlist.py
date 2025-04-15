from dataclasses import dataclass, field
import csv

FaultType = tuple[str, tuple[int, ...], int]

@dataclass
class FaultList:
    # [("layer", (coords,..), bitpos), ...]
    faults: list[FaultType] = field(default_factory=lambda: [])
    resume_idx: int = 0

    def load_from_csv(self, path: str, resume_from : int = 0) -> set[str] :
        """
        Loads a fault list from a csv file and runs compatibility checks with the network
        Args:
            path: filename of the fault list, must be a csv file
            resume_from: only saves injections starting from its value (default=0)
        """
        included_layers = set()
        self.resume_idx = resume_from
        with open(path, "r") as f:
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
                    self.faults.append((layer, int_coords, int(bit)))
        return included_layers
            