from .surrogate import MLP, prepare_dataloaders
from .surrogate_sobol import Sobolev, prepare_sobol_dataloaders
from .ela_comparator import SurrogateELAComparator
from .model_evaluator import ModelEvaluator
from .tensorboard import ensure_tensorboard_running
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.cuda import is_available as is_cuda_available
from datetime import datetime

if __name__ == "__main__":
    input_column_names = ["x1", "x2"]
    output_column_names = ["y", "dy_dx1", "dy_dx2"]
    file_path = "data/bbob_f022_i01_d02_g_samples.csv"
    experiment_name = "bbob_f022_g"
    tensorboard_dir = "lightning_logs"
    total_epochs = 1000

    ensure_tensorboard_running(tensorboard_dir)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
    model = MLP(
        lr=1e-3,
        hidden_dims=[512],
        activation="relu",
    )
    train_dataloder, val_dataloader, test_dataloader, scaler_x, scaler_y = (
        prepare_sobol_dataloaders(
            file_path=file_path,
            input_column_names=input_column_names,
            output_column_names=output_column_names,
            batch_size=100,
            scaler_type=None,
        )
    )
    tb_logger = TensorBoardLogger(save_dir=tensorboard_dir, name=experiment_name)
    accelerator = "gpu" if is_cuda_available() else "cpu"
    trainer = Trainer(
        max_epochs=total_epochs,
        logger=tb_logger,
        log_every_n_steps=10,
        accelerator=accelerator,
        devices=1,
    )

    trainer.fit(model, train_dataloder, val_dataloader)
    trainer.test(model, test_dataloader)

    comparator = SurrogateELAComparator(model, device="cpu")
    results = comparator.compare_multiple_sets(
        train_loader=train_dataloder,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        n_points=1000,
    )
    for key, value in results.items():
        value.to_csv(f"ela_{key}_{current_datetime}.csv", index=False)

    evaluator = ModelEvaluator(model, scaler_y=scaler_y, tb_logger=tb_logger)
    results = evaluator.evaluate_multiple_sets(
        train_loader=train_dataloder,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
    )
    evaluator.log_metrics_to_tensorboard(results, total_epochs)

    for key, value in results.items():
        value.to_csv(f"value_{key}_{current_datetime}.csv", index=False)
