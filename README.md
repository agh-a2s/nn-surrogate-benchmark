# nn-surrogate-benchmark
Neural networks trained as surrogates for benchmark test functions.

## Installation

To install:
```{bash}
poetry install --with dev
```

## Generating Test Data

### BBOB Test Functions
The repository includes functionality to generate sample data from BBOB (Black-Box Optimization Benchmark) test functions:

```bash
poetry run python -m nn_surrogate_benchmark.bbob_sampler
```

This will:
- Generate 10,000 Latin Hypercube samples for a 2D BBOB function
- Save the samples to a CSV file named `{function_id}_samples.csv`
- Use the standard format (x1, x2, y columns) required by the surrogate model

Default settings:
- Sampling bounds: [-5.0, 5.0]
- Number of samples: 10,000
- Dimensions: 2
- Function ID: 1 (Sphere function)
- Instance ID: 1

To modify these parameters, edit the constants in `bbob_sampler.py`.

## Data Format

The input data should be in CSV format with the following structure:
- Input features should be named `x1`, `x2`, ..., `xn` where n is the dimension of the input space
- The target variable should be named `y`
- Each row represents one sample point
- All values should be numeric

Example CSV format:
```
x1,x2,y
1.234,5.678,9.012
2.345,6.789,0.123
...
```

## Running the Code

### Using Default Settings

To run with default settings:
```
poetry run python3 -m nn_surrogate_benchmark
```

### Using Custom Data

To run with your own data:
1. Prepare your CSV file following the format described above
2. Modify the following parameters in `nn_surrogate_benchmark/__main__.py`:
   - `column_names`: List of input feature names (e.g., `["x1", "x2"]`)
   - `file_path`: Path to your CSV file
   - `experiment_name`: Name for your experiment (used in TensorBoard logs)
   - `total_epochs`: Number of training epochs
   - Optional: adjust the model architecture by modifying `hidden_dims` and other parameters

### Output

The code will:
1. Train a neural network on your data
2. Generate evaluation metrics and plots in TensorBoard
3. Save CSV files with:
   - ELA (Exploratory Landscape Analysis) features comparison
   - Model predictions and errors for train/validation/test sets
4. Automatically open TensorBoard in your browser for visualization

### Accessing TensorBoard

TensorBoard will automatically:
- Start on port 6006 (default TensorBoard port)
- Open in your default web browser at `http://localhost:6006`
- Display logs from the `lightning_logs` directory

If TensorBoard doesn't open automatically or you want to access it later:
1. Ensure the training is running or has completed
2. Open your browser and navigate to `http://localhost:6006`
3. Or manually start TensorBoard with:
```
tensorboard --logdir lightning_logs --port 6006
```

### Accessing Optuna Dashboard
```
poetry run optuna-dashboard  sqlite:///optuna_results/mlp_hyperopt.db --port 8080
```

The logs contain:
- Training and validation metrics
- Model predictions visualization
- Error distribution plots
- Comparison plots between true and predicted values
