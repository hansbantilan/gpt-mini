import os
from typing import Generator, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from gpt_mini.modeling.gpt import Gpt
from gpt_mini.utility import logger, well_known_paths
from gpt_mini.utility.plot_layer import PlotLayer

log = logger.init("tensorflow_char")

# TF optimization flags
os.environ["CUDA_CACHE_DISABLE"] = "0"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "1"
os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
os.environ["TF_SYNC_ON_FINISH"] = "0"

# TF logging flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


class Tensorflow_Char_Gpt(Gpt):
    """implementation of a character-level GPT model"""

    def __init__(
        self,
        model_version: str = "hb_20230411",
        model_config: str = "default",
        data_source: str = "local",
        disable_gpu: bool = False,
    ):
        super().__init__(
            "tensorflow_char",
            model_version,
            model_config,
            data_source,
        )
        if disable_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self._params["dim_head"] = (
            self._params["embedding_dim"] // self._params["num_heads"]
        )

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

    def _tokenize(self, text_dict: dict) -> dict:
        # Construct a character-level vocabulary based on the training set
        vocabulary = sorted(list(set(text_dict.get("train"))))
        self._vocab_size = len(vocabulary)

        # Construct mapping and inverse mapping between characters and integers
        char_to_int = dict()
        int_to_char = dict()
        for i, char in enumerate(vocabulary):
            # add the character and its corresponding integer to the dictionary
            char_to_int[char] = i
            int_to_char[i] = char

        # Construct tokenizer encoder/decoder
        self._encode = lambda string: [char_to_int[char] for char in string]
        self._decode = lambda integer_list: "".join(
            [int_to_char[i] for i in integer_list]
        )

        # Tokenize each dataset split and cast each to a TensorFlow tensor
        data_dict = dict()
        for split in ["train", "validation", "test"]:
            data_dict[split] = tf.constant(self._encode(text_dict.get(split)))

        return data_dict

    def _generate_batch(
        self, data_dict: dict, split: str
    ) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
        data = data_dict.get(split)

        # Extract from data at random indices
        shape = (self._params.get("batch_size"),)
        minval = 0
        maxval = len(data) - self._params.get("context_length")
        dtype = tf.dtypes.int32
        random_index_list = tf.random.uniform(
            shape, minval=minval, maxval=maxval, dtype=dtype
        )
        context = tf.stack(
            [
                data[index : index + self._params.get("context_length")]
                for index in random_index_list
            ]
        )
        target = tf.stack(
            [
                data[index + 1 : index + self._params.get("context_length") + 1]
                for index in random_index_list
            ]
        )

        while True:
            yield context, target

    class _embedding_layer(tf.keras.layers.Layer):
        def __init__(self, params: dict, vocab_size: int):
            super().__init__()
            self.context_length = params["context_length"]
            self.token_embedding_layer = tf.keras.layers.Embedding(
                input_dim=vocab_size, output_dim=params["embedding_dim"]
            )
            self.position_embedding_layer = tf.keras.layers.Embedding(
                input_dim=params["context_length"],
                output_dim=params["embedding_dim"],
            )

        def call(self, inputs):
            """
            inputs here is batch_size x context_length
            returns: batch_size x context_length x embedding_dim
            """
            token_embedding = self.token_embedding_layer(inputs)
            position_embedding = self.position_embedding_layer(
                tf.range(self.context_length)
            )
            return token_embedding + position_embedding

    class _multi_headed_attention_layer(tf.keras.layers.Layer):
        def __init__(self, params: dict):
            super().__init__()
            self.multi_attention_layer_list = (
                [  # instantiate num_heads different self-attention heads
                    self._single_attention_layer(params)
                    for _ in range(params["num_heads"])
                ]
            )
            self.skip_projection = tf.keras.layers.Dense(
                units=params["embedding_dim"], use_bias=False
            )
            self.dropout_layer = tf.keras.layers.Dropout(rate=params["dropout"])

        def call(self, inputs):
            """
            inputs here is batch_size x context_length x embedding_dim
            concatenates num_heads representations each of batch_size x context_length x dim_head
            returns: batch_size x context_length x embedding_dim
            """
            multi_attention_list = [
                layer(inputs) for layer in self.multi_attention_layer_list
            ]
            x = tf.concat(multi_attention_list, axis=-1)
            x = self.skip_projection(x)
            x = self.dropout_layer(x)
            return x

        class _single_attention_layer(tf.keras.layers.Layer):
            def __init__(self, params: dict):
                super().__init__()
                self.embedding_dim = params["embedding_dim"]
                self.query_layer = tf.keras.layers.Dense(
                    units=params["dim_head"], use_bias=False
                )
                self.key_layer = tf.keras.layers.Dense(
                    units=params["dim_head"], use_bias=False
                )
                self.value_layer = tf.keras.layers.Dense(
                    units=params["dim_head"], use_bias=False
                )
                self.dropout_layer = tf.keras.layers.Dropout(rate=params["dropout"])

            def call(self, inputs):
                """
                inputs here is batch_size x context_length x embedding_dim
                returns: batch_size x context_length x dim_head
                """
                query = self.query_layer(inputs)
                key = self.key_layer(inputs)
                value = self.value_layer(inputs)
                weights = (
                    query
                    @ tf.transpose(key, perm=[0, 2, 1])
                    * self.embedding_dim**-0.5
                )
                weights = tf.keras.layers.Softmax()(
                    weights, mask=tf.cast(tf.linalg.band_part(weights, -1, 0), tf.bool)
                )
                weights = self.dropout_layer(weights)
                return weights @ value

    class _feed_forward_layer(tf.keras.layers.Layer):
        def __init__(self, params: dict):
            super().__init__()
            self.feed_forward = tf.keras.layers.Dense(
                units=4 * params["embedding_dim"], activation="relu"
            )
            self.skip_projection = tf.keras.layers.Dense(
                units=params["embedding_dim"], use_bias=False
            )
            self.dropout_layer = tf.keras.layers.Dropout(rate=params["dropout"])

        def call(self, inputs):
            """
            inputs here is batch_size x context_length x embedding_dim
            returns: batch_size x context_length x embedding_dim
            """
            x = self.feed_forward(inputs)
            x = self.skip_projection(x)
            x = self.dropout_layer(x)
            return x

    def _create_model_architecture(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=self._params["context_length"])
        x = self._embedding_layer(self._params, vocab_size=self._vocab_size)(inputs)
        for _ in range(self._params["layer_depth"]):  # loop over decoder blocks
            x = x + self._multi_headed_attention_layer(self._params)(
                tf.keras.layers.LayerNormalization()(x)
            )
            x = x + self._feed_forward_layer(self._params)(
                tf.keras.layers.LayerNormalization()(x)
            )
        x = tf.keras.layers.LayerNormalization()(x)
        outputs = tf.keras.layers.Dense(units=self._vocab_size)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def _respond(
        self, model: tf.keras.Model, context: tf.Tensor, max_next_tokens: int
    ) -> int:
        for _ in range(max_next_tokens):
            y_pred = model.predict(context[:, -self._params["context_length"] :])
            logits = y_pred[:, -1, :]
            next_index = tf.random.categorical(
                logits=logits, num_samples=1, dtype=tf.int32
            )
            context = tf.concat([context, next_index], axis=1)
        return context

    def train(self) -> None:
        text_dict = self._get_data()
        data_dict = self._tokenize(text_dict)
        training_generator = self._generate_batch(data_dict, split="train")
        validation_generator = self._generate_batch(data_dict, split="validation")

        model = self._create_model_architecture()
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(self._params["learning_rate"]),
            run_eagerly=False,
        )
        history = model.fit(
            training_generator,
            epochs=self._params["epochs"],
            validation_data=validation_generator,
            steps_per_epoch=self._params["steps_per_epoch"],
            validation_steps=self._params["validation_steps"],
        )

        plotlayer = PlotLayer(
            history=history, scalar="loss", model_output_dir=self._model_output_dir
        )
        plotlayer.plot_learning_curves()
        self._save_model(model)

    def score(self) -> None:
        text_dict = self._get_data()
        data_dict = self._tokenize(text_dict)
        test_generator = self._generate_batch(data_dict, split="test")

        context, _ = next(test_generator)
        prompt = self._decode(context.numpy()[0].tolist())

        model = self._load_model()
        response = self._decode(
            self._respond(
                model=model,
                context=context,
                max_next_tokens=self._params["max_next_tokens"],
            )[:, self._params["context_length"] :]
            .numpy()[0]
            .tolist()
        )
        print(f"--PROMPT--\n{prompt}\n")
        print(f"--RESPONSE--\n{response}\n")
