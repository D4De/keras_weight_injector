# Testing
This folder contains testing scripts to validate the injector's results.

### `output_consistency.py`
Given in input a folder containing subfolders created by the injector and containing the inference results (the ones produced with the --save-scores flag enabled), validates their consistency (the same injection on the same model must produce the same data).
Multiple runs of the same model can be validated at once with a One vs Rest strategy.
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
Given in input a folder containing the CSV reports of multiple runs of the injector, compares row-wise the CSV files and test whether the results are different across multiple runs.
Multiple versions of the same injection are evalued with a One vs Rest strategy.
Usage:
```
python reports_consistency.py path/to/folder
```
