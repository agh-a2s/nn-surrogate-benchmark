# nn-surrogate-benchmark
Neural networks trained as surrogates for benchmark test functions.

To install:
```
poetry install
```

To run for `heat_inversion.csv`:
```
poetry shell
python3 nn_surrogate_benchmark/surrogate.py
```
In order to open tensorboard run:
```
tensorboard --logdir lightning_logs
```