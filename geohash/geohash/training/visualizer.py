"""Real-time terminal training visualizer using matplotlib with mpl_ascii backend."""

import matplotlib as mpl

mpl.use("module://mpl_ascii")

import matplotlib.pyplot as plt  # noqa: E402  (must come after backend selection)


_CLEAR = "\033[2J\033[H"  # ANSI: clear screen + move cursor to top-left


class TrainingVisualizer:
    """Live terminal line graph visualizer for epoch-by-epoch training metrics."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def update(
        self,
        history: dict[str, list[float]],
        epoch: int,
        total_epochs: int,
    ) -> None:
        """Redraw the terminal plots with the latest history after each epoch."""
        if not self.enabled:
            return

        print(_CLEAR, end="", flush=True)

        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))

        # Top-left: Train + Test loss
        axes[0, 0].plot(epochs, history["train_loss"], label="Train")
        axes[0, 0].plot(epochs, history["test_loss"], label="Test")
        axes[0, 0].set_title(f"Loss  [epoch {epoch}/{total_epochs}]")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("MSE")
        axes[0, 0].legend()

        # Top-right: RMSE
        axes[0, 1].plot(epochs, history["rmse"])
        axes[0, 1].set_title("Test RMSE")
        axes[0, 1].set_xlabel("Epoch")

        # Bottom-left: MAE
        axes[1, 0].plot(epochs, history["mae"])
        axes[1, 0].set_title("Test MAE")
        axes[1, 0].set_xlabel("Epoch")

        # Bottom-right: Test loss alone (for scale clarity)
        axes[1, 1].plot(epochs, history["test_loss"])
        axes[1, 1].set_title("Test Loss")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("MSE")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def close(self) -> None:
        """Print a blank line to separate the live display from subsequent output."""
        if self.enabled:
            print()
