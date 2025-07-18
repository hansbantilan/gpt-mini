import tensorflow as tf
import tensorflow_datasets as tfds

# Create a dictionary called _params
_params = dict()
_params["epochs"] = 1  # 5000
_params["steps_per_epoch"] = 100
_params["validation_steps"] = 100
_params["learning_rate"] = 0.00001  # 0.0003
_params["batch_size"] = 32  # 64
_params["context_length"] = 256  # 256
_params[
    "embedding_dim"
] = 128  # 384 #note: dimension of each head is embedding_dim//num_heads
_params["num_heads"] = 2  # 6
_params["layer_depth"] = 6  # 6
_params["dropout"] = 0.0
_params["max_next_tokens"] = 200

# Load the tiny-shakespeare dataset
dataset = tfds.load("tiny_shakespeare")

# Iterate over the dataset
text_dict = dict()
for split in ["train", "validation", "test"]:
    for element in dataset.get(split).as_numpy_iterator():
        # Extract text from dataset using get()
        text_dict[split] = element.get("text").decode("utf-8")

# Construct vocabulary by extracting unique characters in the training set
vocabulary = sorted(list(set(text_dict.get("train"))))
vocab_size = len(vocabulary)

# Create an empty dictionary to hold the character<>integer mappings
char_to_int = {}
int_to_char = {}

# Loop through each character in the list of characters
for i, char in enumerate(vocabulary):
    # add the character and its corresponding integer to the dictionary
    char_to_int[char] = i
    int_to_char[i] = char

# Define how our tokenizer encodes from a string to a list of integers
encode = lambda string: [char_to_int[char] for char in string]

# Define how our tokenizer decodes from a list of integers to a string
decode = lambda integer_list: "".join([int_to_char[i] for i in integer_list])

# Tokenize each dataset split and cast to a TensorFlow tensor
data_dict = dict()
for split in ["train", "validation", "test"]:
    data_dict[split] = tf.constant(encode(text_dict.get(split)))

# uncomment to test output of _generate_batch()
# context, target = next(_generate_batch("train"))
# for batch in range(_params.get("batch_size")):
#    for time in range(_params.get("context_length")):
#        print(f"when input is {context[batch,:time+1].numpy()} the target {target[batch,time]}")

# uncomment for an example of batch_size x context_length
# inputs = tf.constant([[1,2,-1,-2], [1,2,-1,-2]])


class _EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.token_embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=_params["embedding_dim"]
        )
        self.position_embedding_layer = tf.keras.layers.Embedding(
            input_dim=_params["context_length"], output_dim=_params["embedding_dim"]
        )

    def call(self, inputs):
        """
        inputs here is batch_size x context_length
        returns: batch_size x context_length x embedding_dim
        """
        token_embedding = self.token_embedding_layer(inputs)
        position_embedding = self.position_embedding_layer(
            tf.range(_params["context_length"])
        )
        return token_embedding + position_embedding


class _SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, dim_head: int):
        super().__init__()
        self.query_layer = tf.keras.layers.Dense(units=dim_head, use_bias=False)
        self.key_layer = tf.keras.layers.Dense(units=dim_head, use_bias=False)
        self.value_layer = tf.keras.layers.Dense(units=dim_head, use_bias=False)
        self.dropout_layer = tf.keras.layers.Dropout(rate=_params["dropout"])

    def call(self, inputs):
        """
        inputs here is batch_size x context_length x embedding_dim
        returns: batch_size x context_length x dim_head
        """
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        weights = (
            query @ tf.transpose(key, perm=[0, 2, 1]) * _params["embedding_dim"] ** -0.5
        )
        weights = tf.keras.layers.Softmax()(
            weights, mask=tf.cast(tf.linalg.band_part(weights, -1, 0), tf.bool)
        )
        weights = self.dropout_layer(weights)
        return weights @ value


class _MultiHeadedAttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dim_head = _params["embedding_dim"] // _params["num_heads"]
        self.attention_layer_list = [
            _SelfAttentionLayer(dim_head=self.dim_head)
            for _ in range(_params["num_heads"])
        ]
        self.skip_projection = tf.keras.layers.Dense(
            units=_params["embedding_dim"], use_bias=False
        )
        self.dropout_layer = tf.keras.layers.Dropout(rate=_params["dropout"])

    def call(self, inputs):
        """
        inputs here is batch_size x context_length x embedding_dim
        concatenates num_heads representations each of batch_size x context_length x dim_head
        returns: batch_size x context_length x embedding_dim
        """
        attention_list = [layer(inputs) for layer in self.attention_layer_list]
        x = tf.concat(attention_list, axis=-1)
        x = self.skip_projection(x)
        x = self.dropout_layer(x)
        return x


class _FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.feed_forward = tf.keras.layers.Dense(
            units=4 * _params["embedding_dim"], activation="relu"
        )
        self.skip_projection = tf.keras.layers.Dense(
            units=_params["embedding_dim"], use_bias=False
        )
        self.dropout_layer = tf.keras.layers.Dropout(rate=_params["dropout"])

    def call(self, inputs):
        """
        inputs here is batch_size x context_length x embedding_dim
        returns: batch_size x context_length x embedding_dim
        """
        x = self.feed_forward(inputs)
        x = self.skip_projection(x)
        x = self.dropout_layer(x)
        return x


class _DecoderBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.multi_headed_attention_layer = _MultiHeadedAttentionLayer()
        self.feed_forward_layer = _FeedForwardLayer()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        """
        inputs here is batch_size x context_length x embedding_dim
        returns: batch_size x context_length x embedding_dim
        """
        x = inputs + self.multi_headed_attention_layer(self.layer_norm_1(inputs))
        x = x + self.feed_forward_layer(self.layer_norm_2(x))
        return x


def create_model_architecture() -> tf.keras.Model:
    """
    returns: tf.keras.Model
    """
    inputs = tf.keras.Input(shape=_params["context_length"])
    x = _EmbeddingLayer()(inputs)
    for _ in range(_params["layer_depth"]):
        x = _DecoderBlock()(x)
    x = tf.keras.layers.LayerNormalization()(x)
    outputs = tf.keras.layers.Dense(units=vocab_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Define how we get each batch
def generate_batch(split):
    # Define data as data_dict.get(split)
    data = data_dict.get(split)

    # Extract random indices
    shape = (_params.get("batch_size"),)
    minval = 0
    maxval = len(data) - _params.get("context_length")
    dtype = tf.dtypes.int32
    random_index_list = tf.random.uniform(
        shape, minval=minval, maxval=maxval, dtype=dtype
    )

    context = tf.stack(
        [
            data[index : index + _params.get("context_length")]
            for index in random_index_list
        ]
    )
    target = tf.stack(
        [
            data[index + 1 : index + _params.get("context_length") + 1]
            for index in random_index_list
        ]
    )

    while True:
        yield context, target


model = create_model_architecture()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(_params["learning_rate"]),
    run_eagerly=False,
)
history = model.fit(
    generate_batch("train"),
    epochs=_params["epochs"],
    validation_data=generate_batch("validation"),
    steps_per_epoch=_params["steps_per_epoch"],
    validation_steps=_params["validation_steps"],
)


def generate(context: tf.Tensor, max_next_tokens: int) -> int:
    for _ in range(max_next_tokens):
        y_pred = model.predict(context[:, -_params["context_length"] :])
        logits = y_pred[:, -1, :]
        next_index = tf.random.categorical(logits=logits, num_samples=1, dtype=tf.int32)
        context = tf.concat([context, next_index], axis=1)
    return context


context, _ = next(generate_batch("test"))
prompt = decode(context.numpy()[0].tolist())
response = decode(
    generate(context, max_next_tokens=_params["max_next_tokens"])[
        :, _params["context_length"] :
    ]
    .numpy()[0]
    .tolist()
)
print(f"--PROMPT--\n{prompt}\n")
print(f"--RESPONSE--\n{response}")
