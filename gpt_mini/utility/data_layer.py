import os

import tensorflow_datasets as tfds

from gpt_mini.utility import logger

log = logger.init("data_layer")


class DataLayer:
    def __init__(self, data_source: str):
        self._data_source = data_source

    def _get_data(self) -> dict:
        text_dict = dict()
        if self._data_source == "local":
            for split in ["train", "validation", "test"]:
                local_path = os.path.join(
                    well_known_paths["DATASETS_DIR"],
                    f"local-{split}.txt",
                )
                try:
                    with open(local_path, "r") as f:
                        text_dict[split] = f.read()
                except:
                    raise RuntimeError(f"missing {local_path}")
        elif self._data_source == "shakespeare":
            dataset = tfds.load("tiny_shakespeare")
            for split in ["train", "validation", "test"]:
                for element in dataset.get(split).as_numpy_iterator():
                    # Extract text from dataset using get()
                    text_dict[split] = element.get("text").decode("utf-8")
        else:
            raise RuntimeError("data_source must be one of {'shakespeare', ...}.")
        return text_dict
