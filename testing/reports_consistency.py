import glob
import os
import sys
import csv


def get_dirs(dir):
    yield from (d.__fspath__() for d in os.scandir(dir) if d.is_dir())


def get_length(file):
    with open(file, "r") as f:
        return len(list(csv.reader(f)))


def validate_reports(base_dir):
    datasets = get_dirs(base_dir)
    models = {dataset: get_dirs(dataset) for dataset in datasets}
    for dt in models:
        for model in models[dt]:
            validate_model(model)


def validate_model(model):
    csvs = sorted(glob.glob(model + "/*.csv"))
    injs = []
    all_data = []
    for file in csvs:
        with open(file, "r") as f:
            reader = csv.reader(f)
            all_data.append((file, list()))
            for row in reader:
                if row[:4] not in injs:
                    injs.append(row[:4])
                all_data[-1][1].append(row)

    for inj in injs:
        selected = []
        # selects all the files containing this injection
        for file, finjs in all_data:
            for fi in finjs:
                if inj == fi[:4]:
                    selected.append((file, fi))
                    break

        first_file, first_inj = selected.pop(0)
        for file, i in selected:
            try:
                assert i == first_inj
                print(f"{first_inj[0]}: {first_file} vs {file} OK")
            except AssertionError:
                print(
                    f"{first_inj[0]}: AssertionError {first_file} vs {file}:\
{first_inj[4:]} vs {i[4:]}"
                )


if __name__ == "__main__":
    validate_model(sys.argv[1])
