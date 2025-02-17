import os
import time
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set CPU Core Limit
def set_cpu_limit(num_cores):
    os.environ["OMP_NUM_THREADS"] = str(num_cores)

# QuickSort Algorithm
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def test_quicksort():
    arr = np.random.randint(0, 100000, 10000)  # Large random array
    start_time = time.time()
    quicksort(arr)
    end_time = time.time()
    return end_time - start_time

# Matrix Multiplication (NumPy)
def test_matrix_multiplication():
    A = np.random.rand(1000, 1000)
    B = np.random.rand(1000, 1000)
    start_time = time.time()
    C = np.dot(A, B)
    end_time = time.time()
    return end_time - start_time

# Data Processing with Pandas
def test_pandas_operations():
    df = pd.DataFrame(np.random.randint(0, 100, size=(1000000, 4)), columns=list('ABCD'))
    start_time = time.time()
    df['E'] = df['A'] + df['B'] * df['C'] - df['D']
    grouped = df.groupby('A').sum()
    end_time = time.time()
    return end_time - start_time

# Run Experiments with Different CPU Core Configurations
def run_experiments():
    total_cores = multiprocessing.cpu_count()
    core_configs = [1, total_cores // 2, total_cores]  # Test with 1, half, and all cores

    results = []

    for cores in core_configs:
        set_cpu_limit(cores)
        print(f"\nRunning tests with {cores} CPU cores...\n")

        times = {
            "CPU Cores": cores,
            "Quicksort": test_quicksort(),
            "Matrix Multiplication": test_matrix_multiplication(),
            "Pandas Operations": test_pandas_operations(),
        }
        results.append(times)

    return results

# Save and Analyze Results
if __name__ == "__main__":
    results = run_experiments()

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("cpu_performance_results.csv", index=False)

    # Plot Results
    df_results.set_index("CPU Cores").plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Execution Time (seconds)")
    plt.title("Algorithm Performance with Different CPU Core Limits")
    plt.xticks(rotation=0)
    plt.legend(title="Algorithm")
    plt.grid(True)
    plt.savefig("cpu_performance_chart.png")
    plt.show()

    print("Experiment completed. Results saved to 'cpu_performance_results.csv'.")
