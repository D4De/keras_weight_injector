import glob
import numpy as np
from itertools import chain
import os
import sys


def get_dirs(dir):
    return (d.__fspath__() for d in os.scandir(dir) if d.is_dir())


def test_all(base_dir):
    datasets = get_dirs(base_dir)
    models = {dataset: get_dirs(dataset) for dataset in datasets}
    for dt in models:
        for model in models[dt]:
            test(model)


def get_injection_id(filename):
    if "clean" in filename:
        return -1
    return int(filename.split(".")[0].split("_")[1])


def test(model):
    output_files = chain.from_iterable(
        (glob.glob(dir + "/*.npy") for dir in get_dirs(model))
    )
    clusters = []
    ids = {}
    for file in output_files:
        basename = os.path.basename(file)
        inj_id = get_injection_id(basename)

        if not inj_id in ids:
            clusters.append((inj_id, list()))
            ids[inj_id] = len(clusters) - 1

        clusters[ids[inj_id]][1].append(file)

    for inj_id, files in sorted(clusters, key=lambda i: i[0]):
        if len(files) > 1:
            ref = np.load(files[0])
            for other in files[1:]:
                comp = np.load(other)
                try:
                    assert (
                        np.logical_or(
                            np.equal(ref, comp),
                            np.logical_and(np.isnan(ref), np.isnan(comp)),
                        )
                    ).all()
                    print(f"{inj_id}: {files[0]} vs {other} OK")
                except AssertionError:
                    distance = np.sqrt(((ref - comp) ** 2).sum())
                    differences = (ref != comp).any(axis=-1).sum() / ref.shape[0]
                    print(
                        f"{inj_id}: AssertionError: {files[0]} vs {other} ({distance:.6f}, {differences*100:.6f}%)",
                        end=" ",
                    )
                    top1_diff = ref.argmax(axis=1) != comp.argmax(axis=1)
                    if top1_diff.any():
                        print(
                            f"TOP1 differs ({top1_diff.sum()/len(top1_diff)*100:.6f}%)"
                        )
                    else:
                        print("TOP1 does not differ")


if __name__ == "__main__":
    test(sys.argv[1])
