import logging
import shutil
from pathlib import Path

import torch
from data.data import EarthquakeData
from ray import tune
from ray.air import Result
from lstm.base import BaseTrainable
from lstm.plot import plot_scatter, plot_timeseries

logger = logging.getLogger(__name__)


class RegressionTrainable(BaseTrainable):
    def forecast(self):
        self.model.eval()
        with torch.no_grad():
            train_output = [self.model(x.to(self.device)) for x, _ in self.train_loader]
            test_output = [self.model(x.to(self.device)) for x, _ in self.test_loader]
            train_output = torch.cat(train_output, dim=0).detach().cpu().numpy()
            test_output = torch.cat(test_output, dim=0).detach().cpu().numpy()

        return self.y_train[:, -1].numpy(), train_output, self.y_test[:, -1].numpy(), test_output

    def test_result(self, result: Result, metric, mode):
        logger.info("Loading testing from config")
        best_checkpoint = result.get_best_checkpoint(metric, mode)
        self.load_checkpoint(best_checkpoint)

        print(result.path)
        print(result.metrics_dataframe)

        train_y, train_pred, test_y, test_pred = self.forecast()

        def target_idx(y, pred, idx):
            return y[:, idx, None], pred[:, idx, None]

        save_to = Path.home() / "plots" / self.qdata.hash / Path(result.path).stem
        shutil.copytree(result.path, save_to, dirs_exist_ok=True)
        for idx, target in enumerate(self.qdata.targets):
            plot_scatter(*target_idx(train_y, train_pred, idx), save_to / f"{target}_train_scatter.png")
            plot_scatter(*target_idx(test_y, test_pred, idx), save_to / f"{target}_test_scatter.png")
            plot_timeseries(*target_idx(train_y, train_pred, idx), target, save_to / f"{target}_train_timeseries.png")
            plot_timeseries(*target_idx(test_y, test_pred, idx), target, save_to / f"{target}_test_timeseries.png")

        result.metrics_dataframe[["loss", "test_loss"]].plot(legend=True).get_figure().savefig(save_to / "loss.png")
