import pandas as pd
from ray import train
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from .dataset import MovingWindowDataset
from .hypermodel import LSTMModel


def build_trainer(data: pd.DataFrame, target: str, tolerance=5, min_delta=0.0001):
    """Builds a trainer function that takes `config` hyperparameters and train a pytorch LSTM model"""

    def trainer(config: dict):
        """
        Takes a `config` dictionary with hyper-paramenters to be trained
        """
        window_size = config.get("window_size")
        dataset = MovingWindowDataset(data, target, window_size)
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

        input_size = data.shape[1]  # 0:rows 1:columns
        hidden_size = config.get("hidden_size", 1)
        num_layers = config.get("num_layers", 1)

        model = LSTMModel(input_size, hidden_size, num_layers, 1)
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=config["lr"])

        for epoch in range(config["num_epochs"]):
            epoch_loss = 0
            for inputs, targets in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            train.report(dict(loss=epoch_loss, avg_epoch_loss=avg_loss))
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")

    return trainer
