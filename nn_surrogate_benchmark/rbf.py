from scipy.interpolate import RBFInterpolator
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    column_names = ["x1", "x2"]
    for function_id in range(23, 25):
        file_path = f"data/bbob_f{function_id:03d}_i01_d02_samples.csv"
        data_df = pd.read_csv(file_path)
        X = data_df[column_names].values
        y = data_df["y"].values.reshape(-1, 1)

        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        train_perc: float = 0.4
        val_perc: float = 0.3
        train_size = int(train_perc * num_samples)
        val_size = int(val_perc * num_samples)
        train_idx, val_idx, test_idx = (
            indices[:train_size],
            indices[train_size : (train_size + val_size)],
            indices[(train_size + val_size) :],
        )

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"Processing function {function_id}...")
        rbf = RBFInterpolator(X_train, y_train, kernel="linear")
        print(f"RBF model created with kernel='gaussian'")

        y_pred = rbf(X_test)

        df = pd.DataFrame(
            {
                "x1": X_test[:, 0],
                "x2": X_test[:, 1],
                "y_true": y_test.flatten(),
                "y_pred": y_pred.flatten(),
            }
        )
        print(f"Created dataframe for visualization with {len(df)} points")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x1_vals = np.linspace(df["x1"].min(), df["x1"].max(), 100)
        x2_vals = np.linspace(df["x2"].min(), df["x2"].max(), 100)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        true_contour = ax1.tricontourf(df["x1"], df["x2"], df["y_true"])
        ax1.set_title(f"{function_id}: True Values")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        plt.colorbar(true_contour, ax=ax1)

        pred_contour = ax2.tricontourf(df["x1"], df["x2"], df["y_pred"])
        ax2.set_title(f"{function_id}: Predicted Values")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        plt.colorbar(pred_contour, ax=ax2)

        plt.tight_layout()
        plt.savefig(f"rbf_f{function_id:03d}.png")
        plt.close(fig)

        print(f"Completed processing for function {function_id}\n")
