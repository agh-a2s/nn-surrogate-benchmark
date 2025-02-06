from .surrogate import MLP, prepare_dataloaders
from .ela_comparator import SurrogateELAComparator
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch

if __name__ == "__main__":
    total_epochs = 100
    model = MLP(
        input_dim=3,
        lr=1e-3,
        hidden_dims=[512],
        activation="relu",
    )
    train_dataloder, val_dataloader, test_dataloader = prepare_dataloaders(
        file_path="heat_inversion_lhs.csv",
        scaler_type="minmax",
    )
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs", name="surrogate_experiment"
    )
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
    train_dataloder, val_dataloader, test_dataloader = prepare_dataloaders(
        file_path="heat_inversion_lhs.csv",
        scaler_type="minmax",
    )
    comparator = SurrogateELAComparator(model, device="cpu")
    results = comparator.compare_multiple_sets(
        train_loader=train_dataloder,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        n_points=1000,
    )
    print(results)
