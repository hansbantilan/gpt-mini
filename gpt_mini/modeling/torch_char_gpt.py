import os
from typing import Generator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_mini.modeling.gpt import Gpt
from gpt_mini.utility import logger
from gpt_mini.utility.data_layer import DataLayer
from gpt_mini.utility.plot_layer import PlotLayer

log = logger.init("torch_char")


class TorchCharGpt(Gpt):
    """implementation of a character-level GPT model using PyTorch"""

    def __init__(
        self,
        model_version: str = "hb_20230411",
        model_config: str = "default",
        data_source: str = "local",
        disable_gpu: bool = False,
    ):
        super().__init__(
            "torch_char",
            model_version,
            model_config,
            data_source,
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"
        )
        self._params["dim_head"] = (
            self._params["embedding_dim"] // self._params["num_heads"]
        )

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

        # Tokenize each dataset split and cast each to a PyTorch tensor
        data_dict = dict()
        for split in ["train", "validation", "test"]:
            data_dict[split] = torch.tensor(
                self._encode(text_dict.get(split)), device=self.device
            )

        return data_dict

    def _generate_batch(
        self, data_dict: dict, split: str
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        data = data_dict.get(split)

        # Extract from data at random indices
        while True:
            random_indices = torch.randint(
                0,
                len(data) - self._params.get("context_length"),
                (self._params.get("batch_size"),),
                device=self.device,
            )
            context = torch.stack(
                [
                    data[idx : idx + self._params.get("context_length")]
                    for idx in random_indices
                ]
            )
            target = torch.stack(
                [
                    data[idx + 1 : idx + self._params.get("context_length") + 1]
                    for idx in random_indices
                ]
            )
            yield context, target

    class _EmbeddingLayer(nn.Module):
        def __init__(self, params: dict, vocab_size: int):
            super().__init__()
            self.context_length = params["context_length"]
            self.token_embedding_layer = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=params["embedding_dim"]
            )
            self.position_embedding_layer = nn.Embedding(
                num_embeddings=params["context_length"],
                embedding_dim=params["embedding_dim"],
            )

        def forward(self, inputs):
            """
            inputs here is batch_size x context_length
            returns: batch_size x context_length x embedding_dim
            """
            token_embedding = self.token_embedding_layer(inputs)
            position_embedding = self.position_embedding_layer(
                torch.arange(self.context_length, device=inputs.device)
            )
            return token_embedding + position_embedding

    class _MultiHeadedAttentionLayer(nn.Module):
        def __init__(self, params: dict):
            super().__init__()
            self.multi_attention_layer_list = nn.ModuleList(
                [self._SingleAttentionLayer(params) for _ in range(params["num_heads"])]
            )
            self.skip_projection = nn.Linear(
                in_features=params["embedding_dim"],
                out_features=params["embedding_dim"],
                bias=False,
            )
            self.dropout_layer = nn.Dropout(p=params["dropout"])

        def forward(self, inputs):
            """
            inputs here is batch_size x context_length x embedding_dim
            concatenates num_heads representations each of batch_size x context_length x dim_head
            returns: batch_size x context_length x embedding_dim
            """
            multi_attention_list = [
                layer(inputs) for layer in self.multi_attention_layer_list
            ]
            x = torch.cat(multi_attention_list, dim=-1)
            x = self.skip_projection(x)
            x = self.dropout_layer(x)
            return x

        class _SingleAttentionLayer(nn.Module):
            def __init__(self, params: dict):
                super().__init__()
                self.embedding_dim = params["embedding_dim"]
                self.query_layer = nn.Linear(
                    in_features=params["embedding_dim"],
                    out_features=params["dim_head"],
                    bias=False,
                )
                self.key_layer = nn.Linear(
                    in_features=params["embedding_dim"],
                    out_features=params["dim_head"],
                    bias=False,
                )
                self.value_layer = nn.Linear(
                    in_features=params["embedding_dim"],
                    out_features=params["dim_head"],
                    bias=False,
                )
                self.dropout_layer = nn.Dropout(p=params["dropout"])

            def forward(self, inputs):
                """
                inputs here is batch_size x context_length x embedding_dim
                returns: batch_size x context_length x dim_head
                """
                query = self.query_layer(inputs)
                key = self.key_layer(inputs)
                value = self.value_layer(inputs)
                weights = query @ key.transpose(-2, -1) * self.embedding_dim**-0.5
                # Create causal mask
                seq_len = inputs.size(1)
                mask = torch.tril(
                    torch.ones(seq_len, seq_len, device=inputs.device)
                ).bool()
                weights = weights.masked_fill(~mask, float("-inf"))
                weights = F.softmax(weights, dim=-1)
                weights = self.dropout_layer(weights)
                return weights @ value

    class _FeedForwardLayer(nn.Module):
        def __init__(self, params: dict):
            super().__init__()
            self.feed_forward = nn.Linear(
                in_features=params["embedding_dim"],
                out_features=4 * params["embedding_dim"],
            )
            self.skip_projection = nn.Linear(
                in_features=4 * params["embedding_dim"],
                out_features=params["embedding_dim"],
                bias=False,
            )
            self.dropout_layer = nn.Dropout(p=params["dropout"])

        def forward(self, inputs):
            """
            inputs here is batch_size x context_length x embedding_dim
            returns: batch_size x context_length x embedding_dim
            """
            x = F.relu(self.feed_forward(inputs))
            x = self.skip_projection(x)
            x = self.dropout_layer(x)
            return x

    def _create_model_architecture(self) -> nn.Module:
        class GPTModel(nn.Module):
            def __init__(
                self,
                params,
                vocab_size,
                embedding_layer,
                multi_headed_attention_layer,
                feed_forward_layer,
            ):
                super().__init__()
                self.params = params
                self.embedding_layer = embedding_layer(params, vocab_size)
                self.layers = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "attention": multi_headed_attention_layer(params),
                                "feed_forward": feed_forward_layer(params),
                                "norm1": nn.LayerNorm(params["embedding_dim"]),
                                "norm2": nn.LayerNorm(params["embedding_dim"]),
                            }
                        )
                        for _ in range(params["layer_depth"])
                    ]
                )
                self.final_norm = nn.LayerNorm(params["embedding_dim"])
                self.output_layer = nn.Linear(
                    in_features=params["embedding_dim"], out_features=vocab_size
                )

            def forward(self, inputs):
                x = self.embedding_layer(inputs)
                for layer in self.layers:
                    x = x + layer["attention"](layer["norm1"](x))
                    x = x + layer["feed_forward"](layer["norm2"](x))
                x = self.final_norm(x)
                outputs = self.output_layer(x)
                return outputs

        model = GPTModel(
            self._params,
            self._vocab_size,
            self._EmbeddingLayer,
            self._MultiHeadedAttentionLayer,
            self._FeedForwardLayer,
        )
        return model.to(self.device)

    def _respond(
        self, model: nn.Module, context: torch.Tensor, max_next_tokens: int
    ) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            for _ in range(max_next_tokens):
                context_crop = context[:, -self._params["context_length"] :]
                y_pred = model(context_crop)
                logits = y_pred[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat([context, next_token], dim=1)
        return context

    def _prepare_data(self) -> dict:
        """
        Prepare data outside of train/score for easier debugging
        """
        datalayer = DataLayer(data_source=self._data_source)
        text_dict = datalayer._get_data()
        data_dict = self._tokenize(text_dict)
        return data_dict

    def _save_model(self, model: nn.Module) -> None:
        log.info("Saving model...")
        torch.save(
            model.state_dict(), os.path.join(self._model_output_dir, "model.pth")
        )
        # Also save model parameters for reconstruction
        torch.save(
            {"vocab_size": self._vocab_size, "params": self._params},
            os.path.join(self._model_output_dir, "model_config.pth"),
        )

    def _load_model(
        self,
        local_model_dir: str = None,
        compile: bool = False,
    ) -> nn.Module:
        """
        compile: True = when you want to retrain the model
        """
        if local_model_dir is None:
            local_model_dir = self._model_output_dir
        log.info("Loading model...")

        # Load model configuration
        config_path = os.path.join(local_model_dir, "model_config.pth")
        config = torch.load(config_path, map_location=self.device)
        self._vocab_size = config["vocab_size"]

        # Create model architecture
        model = self._create_model_architecture()

        # Load model weights
        model_path = os.path.join(local_model_dir, "model.pth")
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        return model

    def train(self) -> None:
        data_dict = self._prepare_data()
        training_generator = self._generate_batch(data_dict, split="train")
        validation_generator = self._generate_batch(data_dict, split="validation")

        model = self._create_model_architecture()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self._params["learning_rate"]
        )
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_losses = []

        model.train()
        for epoch in range(self._params["epochs"]):
            epoch_train_loss = 0.0
            for step in range(self._params["steps_per_epoch"]):
                context, target = next(training_generator)

                optimizer.zero_grad()
                outputs = model(context)
                loss = criterion(outputs.view(-1, self._vocab_size), target.view(-1))
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / self._params["steps_per_epoch"]
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for step in range(self._params["validation_steps"]):
                    context, target = next(validation_generator)
                    outputs = model(context)
                    loss = criterion(
                        outputs.view(-1, self._vocab_size), target.view(-1)
                    )
                    epoch_val_loss += loss.item()

            avg_val_loss = epoch_val_loss / self._params["validation_steps"]
            val_losses.append(avg_val_loss)
            model.train()

            log.info(
                f"Epoch {epoch+1}/{self._params['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

        # Create history object compatible with PlotLayer
        class History:
            def __init__(self, train_losses, val_losses):
                self.history = {"loss": train_losses, "val_loss": val_losses}

        history = History(train_losses, val_losses)
        plotlayer = PlotLayer(
            history=history, scalar="loss", model_output_dir=self._model_output_dir
        )
        plotlayer.plot_learning_curves()
        self._save_model(model)

    def score(self) -> None:
        data_dict = self._prepare_data()
        test_generator = self._generate_batch(data_dict, split="test")

        context, _ = next(test_generator)
        prompt = self._decode(context[0].cpu().numpy().tolist())

        model = self._load_model()
        response = self._decode(
            self._respond(
                model=model,
                context=context,
                max_next_tokens=self._params["max_next_tokens"],
            )[:, self._params["context_length"] :][0]
            .cpu()
            .numpy()
            .tolist()
        )
        print(f"--PROMPT--\n{prompt}\n")
        print(f"--RESPONSE--\n{response}\n")
