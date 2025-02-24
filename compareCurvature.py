import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_curvature(file1_path: str, file2_path: str, label1: str = "QP Method", label2: str = "CasADi Method"):
    """
    Compare curvature between two raceline CSV files using the fifth column as curvature.
    """
    # Load data
    df1 = pd.read_csv(file1_path, comment='#')
    df2 = pd.read_csv(file2_path, comment='#')

    # Use the fifth column (index 4) as curvature
    column_index = 4  # Index starts from 0
    kappa1 = df1.iloc[:, column_index].values
    kappa2 = df2.iloc[:, column_index].values

    # Calculate total absolute curvature
    total_curv1 = np.sum(np.abs(kappa1))
    total_curv2 = np.sum(np.abs(kappa2))

    # Calculate average absolute curvature
    avg_curv1 = np.mean(np.abs(kappa1))
    avg_curv2 = np.mean(np.abs(kappa2))

    # Print results
    print(f"Total absolute curvature comparison:")
    print(f"{label1}: {total_curv1:.4f} rad/m")
    print(f"{label2}: {total_curv2:.4f} rad/m")
    print(f"\nAverage absolute curvature:")
    print(f"{label1}: {avg_curv1:.4f} rad/m")
    print(f"{label2}: {avg_curv2:.4f} rad/m")

    # Determine better method
    if total_curv1 < total_curv2:
        print(f"\n{label1} has lower total curvature")
    elif total_curv2 < total_curv1:
        print(f"\n{label2} has lower total curvature")
    else:
        print("\nBoth methods have identical total curvature")

# Example usage
if __name__ == "__main__":
    qp_file = "C:/Rodney/rodney/torch/F1tenthTest/outputs/traj_race_cl.csv"
    casadi_file = "C:/Rodney/rodney/torch/F1tenthTest/outputs/casadiTest.csv"
    
    compare_curvature(qp_file, casadi_file)
