# Testing
This folder contains testing scripts to validate the injector's results. Since we performed various subsequent refactorings of the code, they have been implemented to check that multiple implementations/versions of the tool produce precisely the same result.

### `output_consistency.py`
This script verifies that fault injection results produced by multiple runs of different versions of the tool are perfectly equal. The script receives in input the path of a folder containing a set of subfolders created by multiple runs of different versions of the tool and containing the inference results (the ones generated with the ``--save-scores`` flag enabled).
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
This script does the same of the previous one with the only difference that it works on ``csv`` reports. The script receives in input a folder containing the ``csv`` reports produced by multiple runs of different versions of the tool and containing the inference results.
Usage:
```
python reports_consistency.py path/to/folder
```
