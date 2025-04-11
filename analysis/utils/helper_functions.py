from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import lightning as L

def create_classifier_trainer(name, max_epochs=20, log_every_n_steps=5, enable_pbar=True):

    csv_logger = CSVLogger("logs", name=name)

    checkpoint = ModelCheckpoint(
        monitor="val_mse",
        mode="min",
        save_top_k=1,
        filename=f"best_{name}",
    )
    early_stop = EarlyStopping(
        monitor="val_mse",  
        patience=3,  
        mode="min",  

    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        logger=[csv_logger],
        callbacks=[checkpoint, early_stop],
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_pbar,
    )
    return trainer, csv_logger, checkpoint