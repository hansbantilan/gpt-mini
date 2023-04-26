import os

import matplotlib.pyplot as plt
import tensorflow as tf

from gpt_mini.utility import logger

log = logger.init("plot_layer")


class PlotLayer:
    def __init__(
        self, history: tf.keras.callbacks.History, scalar: str, model_output_dir: str
    ):
        self._history = history
        self._scalar = scalar
        self._model_output_dir = model_output_dir

    def plot_learning_curves(self) -> None:
        loss = self._history.history[self._scalar]
        val_loss = self._history.history["val_" + self._scalar]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, color="navy", lw=2, label="training " + self._scalar)
        plt.plot(
            epochs,
            val_loss,
            color="darkorange",
            lw=2,
            label="validation " + self._scalar,
        )
        plt.xlabel("epoch")
        plt.ylabel(self._scalar)
        plt.title(f"{self._scalar} per epoch")
        plt.legend(loc="upper right")
        log.info(f"Saving {self._scalar} learning curve...")
        fig_fpath = os.path.join(
            self._model_output_dir, f"learning_curve_{self._scalar}.png"
        )
        plt.savefig(fig_fpath)
        plt.show()
        plt.clf()
        plt.close()
