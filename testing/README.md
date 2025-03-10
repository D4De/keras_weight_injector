# Testing
This folder contains testing scripts to validate the injector's results. Since we performed various subsequent refactorings of the code, they have been implemented to compare that multiple implementations/versions of this tool produce exactly the same result.

### `output_consistency.py`
Given in input a folder containing subfolders created by the injector and containing the inference results (the ones produced with the ``--save-scores`` flag enabled), this script validates their consistency (i.e., the same injection on the same model must generate the same data).
Multiple runs of the same model can be validated at once with a ``One`` vs. ``Rest`` strategy.
The folder is expected to have the following structure:
```
folder
    sub1
        clean.npy
        inj_0.npy
        ...
    sub2
        *.npy
        ...
    ...
```
Usage:
```
python output_consistency.py path/to/report
```

### `reports_consistency.py`
Given in input a folder containing the ``csv`` reports of multiple runs of the injector, compares row-wise the ``csv`` files and test whether the results are different across multiple runs.
Multiple versions of the same injection are evaluated with a ``One`` vs ``Rest`` strategy.
Usage:
```
python reports_consistency.py path/to/folder
```
