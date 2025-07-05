import importlib

from gpt_mini.modeling.gpt import Gpt


class GptFactory:
    """object factory class for GPT models"""

    def __init__(
        self,
        model_type: str,
        model_version: str,
        model_config: str,
        data_source: str,
        disable_gpu: bool,
    ):
        self._model_type = model_type
        self._model_version = model_version
        self._model_config = model_config
        self._data_source = data_source
        self._disable_gpu = disable_gpu

    def _gpt_factory(self) -> Gpt:
        """
        non-public member function that instantiates the correct GPT model
        :returns: an object instantiated from one of {TensorflowCharGpt, ...}
        """
        if self._model_type == "tensorflow_char":
            module = importlib.import_module("gpt_mini.modeling.tensorflow_char_gpt")
            implementation = getattr(module, "TensorflowCharGpt")
            return implementation(
                self._model_version,
                self._model_config,
                self._data_source,
                self._disable_gpu,
            )
        elif self._model_type == "torch_char":
            module = importlib.import_module("gpt_mini.modeling.torch_char_gpt")
            implementation = getattr(module, "TorchCharGpt")
            return implementation(
                self._model_version,
                self._model_config,
                self._data_source,
                self._disable_gpu,
            )
        else:
            raise RuntimeError(
                "model_type must be one of {'tensorflow_char', 'torch_char'}."
            )

    def train(self) -> None:
        """
        public member function that is overridden by model-specific training
        :returns: None
        """
        return self._gpt_factory().train()

    def score(self) -> None:
        """
        public member function that is overridden by model-specific scoring
        :returns: None
        """
        return self._gpt_factory().score()
