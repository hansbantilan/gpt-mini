import os
from abc import ABC, abstractmethod

import tensorflow as tf

from gpt_mini.utility import logger, well_known_paths
from gpt_mini.utility.utility import load_params

log = logger.init("gpt")


class Gpt(ABC):
    """abstract class for GPT models"""

    def __init__(
        self,
        model_type: str,
        model_version: str,
        model_config: str,
        data_source: str,
    ) -> None:
        self._model_type = model_type
        self._model_version = model_version
        self._model_config = model_config
        self._data_source = data_source

        self._params = load_params(
            os.path.join(
                well_known_paths["PARAMS_DIR"],
                model_type,
                f"{self._model_config}.yaml",
            )
        )

        self._model_output_dir = os.path.join(
            well_known_paths["ROOT"],
            self._params["model_output_dir"],
            f"model_{self._model_version}",
        )
        if not os.path.exists(self._model_output_dir):
            os.makedirs(self._model_output_dir)

    @abstractmethod
    def train(self):
        """
        implements a specific GPT model's training
        :returns: None
        """

    @abstractmethod
    def score(self):
        """
        implements a specific GPT model's inference
        :returns: None
        """

    def _save_model(self, model: tf.keras.Model) -> None:
        log.info("Saving model...")
        model.save(self._model_output_dir)

    def _load_model(
        self,
        local_model_dir: str = None,
        compile: bool = False,
    ) -> tf.keras.Model:
        """
        compile: True = when you want to retrain the model
        """
        if local_model_dir is None:
            local_model_dir = self._model_output_dir
        log.info("Loading model...")
        model = tf.keras.models.load_model(local_model_dir)
        return model
