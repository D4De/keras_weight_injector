import pandas as pd

def calculate_means(csv_file):
    # Read the CSV file into a Pandas DataFrame and skip the GOLDEN row
    df = pd.read_csv(csv_file, quotechar='"', skiprows=[1])

    # Store the "n_injections" column for later division
    div = df["n_injections"]

    # Drop unnecessary columns if they exist, avoiding errors if they are missing
    df = df.drop(columns=["inj_id", "target_layer", "layer_weigths", "bit_pos", "n_injections"], errors='ignore')

    # Convert all remaining columns to numeric values, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Normalize data by the number of injections
    df = df.div(div, axis=0)

    # Compute the mean of each column
    means = df.mean()

    print("Column means:")
    print(means)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python compute_mean.py file.csv")
    else:
        calculate_means(sys.argv[1])
