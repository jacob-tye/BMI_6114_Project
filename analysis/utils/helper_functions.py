from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import lightning as L
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch

def create_classifier_trainer(name, max_epochs=20, log_every_n_steps=5, enable_pbar=True, allow_early_stop=True):

    csv_logger = CSVLogger("logs", name=name)

    callbacks = []

    checkpoint = ModelCheckpoint(
        monitor="val_mse",
        mode="min",
        save_top_k=1,
        filename=f"best_{name}",
    )
    callbacks.append(checkpoint)
    if allow_early_stop:
        early_stop = EarlyStopping(
            monitor="val_mse",  
            patience=3,  
            mode="min",  

        )
        callbacks.append(early_stop)

    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        logger=[csv_logger],
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_pbar,
    )
    return trainer, csv_logger, checkpoint

def split_data(df, random_state=42):

    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)
    return train_df, val_df, test_df


def plot_regression_results_torch(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            y_hat = model(x)
            y_true.append(y)
            y_pred.append(y_hat)

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Regression Results')
    plt.show()

def get_features_and_target(data):
    columns = data.columns.tolist()
    features = data.iloc[:, :-1].copy()
    target = data[columns[-1]].values
    return features, target


def plot_regression_results_sklearn(model, test_data):
    x, y = get_features_and_target(test_data)
    y_hat = model.predict(x)

    r2 = r2_score(y, y_hat)
    mse = mean_squared_error(y, y_hat)

    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_hat, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Regression Results')

    # Add R² and MSE text in upper left corner
    plt.text(
        0.05, 0.95,
        f'R² = {r2:.3f}\nMSE = {mse:.3f}',
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.tight_layout()
    plt.show()

def make_plots(history, title):
    plt.figure(figsize=(12, 5))
    plt.suptitle(title)
    train_history = history.dropna(subset=["train_loss"])
    val_history = history.dropna(subset=["val_loss"])
    # Loss Plot
    plt.plot(train_history["step"], train_history["train_loss"], label="Train Loss", marker="o")
    plt.plot(
        val_history["step"],
        val_history["val_loss"],
        label="Validation Loss",
        marker="o",
        linestyle="-",
    )
    plt.xlabel("step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()