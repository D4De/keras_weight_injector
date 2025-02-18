import csv
import sys
import numpy as np

def load(filepath):
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(iter(reader))[5:]
        n_inj = int(next(iter(reader))[4])
        for row in reader:
            rows.append([int(n) for n in row[5:]])
    return np.asarray(rows), header, n_inj

def calc_mean(table):
    return np.sum(table, axis=0)/table.shape[0]

if __name__ == "__main__":
    table, header, n_inj = load(sys.argv[1])
    mean = calc_mean(table)
    header = ','.join([f"{s:^16}" for s in header])
    mean = ','.join([f"{s/n_inj*100:15.3f}%" for s in mean])
    print(header)
    print(mean)
