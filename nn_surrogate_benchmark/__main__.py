from .surrogate import MLP, prepare_dataloaders
from .ela_comparator import SurrogateELAComparator
from .model_evaluator import ModelEvaluator
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import torch

if __name__ == "__main__":
    column_names = ["x1", "x2"]
    file_path = "bbob_f001_i01_d02_samples.csv"
    experiment_name = "bbob_f001"

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
    total_epochs = 100
    model = MLP(
        input_dim=len(column_names),
        lr=1e-3,
        hidden_dims=[512],
        activation="relu",
    )
    train_dataloder, val_dataloader, test_dataloader, scaler_x, scaler_y = (
        prepare_dataloaders(
            file_path=file_path,
            scaler_type=None,
            column_names=column_names,
        )
    )
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=experiment_name)
    trainer = Trainer(
        max_epochs=total_epochs,
        logger=tb_logger,
        log_every_n_steps=10,
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(model, train_dataloder, val_dataloader)
    trainer.test(model, test_dataloader)

    model_path = "surrogate_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # model.load_state_dict(torch.load("surrogate_model.pt"))
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
