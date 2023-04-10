import tensorflow as tf
import tensorflow_datasets as tfds

# Load the tiny-shakespeare dataset
dataset = tfds.load("tiny_shakespeare")

# Iterate over the dataset
text_dict = dict()
for split in ["train", "validation", "test"]:
    for element in dataset.get(split).as_numpy_iterator():
        # Extract text from dataset using get()
        text_dict[split] = element.get("text").decode("utf-8")

# Extract unique characters as a sorted list
chars = sorted(list(set(text_dict.get("train"))))
vocab_size = len(chars)

# Create an empty dictionary to hold the character<>integer mappings
char_to_int = {}
int_to_char = {}

# Loop through each character in the list of characters
for i, char in enumerate(chars):
    # add the character and its corresponding integer to the dictionary
    char_to_int[char] = i
    int_to_char[i] = char

# Define how our tokenizer encodes from a string to a list of integers
encode = lambda string: [char_to_int[char] for char in string]

# Define how our tokenizer decodes from a list of integers to a string
decode = lambda integer_list: "".join([int_to_char[i] for i in integer_list])

# Cast tokenized text to a TensorFlow constant
data_dict = dict()
for split in ["train", "validation", "test"]:
    data_dict[split] = tf.constant(encode(text_dict.get(split)))

# Create a dictionary called _params
_params = dict()
_params["epochs"] = 10 #5000
_params["steps_per_epoch"] = 100
_params["validation_steps"] = 100
_params["learning_rate"] = 0.0001 # 0.0003
_params["batch_size"] = 2 #64
_params["context_length"] = 4 #256
_params["embedding_dim"] = 8 #384
_params["head_dim"] = 3 #6
_params["layer_depth"] = 1 #6
_params["dropout"] = 0 #0.2

# uncomment to test output of _generate_batch()
#context, target = _generate_batch("train")
#for b in range(_params.get("batch_size")):
#    for t in range(_params.get("context_length")):
#        print(f"when input is {context[b, :t+1].numpy()} the target {target[b,t]}")

#uncomment for an example of batch_size x context_length
#inputs = tf.constant([[1,2,-1,-2], [1,2,-1,-2]])

class _embedding_layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.token_embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=_params["embedding_dim"])
        self.position_embedding_layer = tf.keras.layers.Embedding(input_dim=_params["context_length"], output_dim=_params["embedding_dim"])
        
    def call(self, inputs):
        '''
        inputs here is batch_size x context_length
        returns: batch_size x context_length x embedding_dim
        '''
        token_embedding = self.token_embedding_layer(inputs)
        position_embedding = self.position_embedding_layer(tf.range(inputs.shape[1]))
        return token_embedding + position_embedding

class _self_attention_layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.query_layer = tf.keras.layers.Dense(units=_params["head_dim"], use_bias=False)
        self.key_layer = tf.keras.layers.Dense(units=_params["head_dim"], use_bias=False)
        self.value_layer = tf.keras.layers.Dense(units=_params["head_dim"], use_bias=False)
        
    def call(self, inputs):
        '''
        inputs here is batch_size x context_length x embedding_dim
        returns: batch_size x head_dim x head_dim
        '''
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        weights = query @ tf.transpose(key, perm=[0, 2, 1]) *_params["embedding_dim"]**-0.5
        weights = tf.keras.layers.Softmax()(weights, mask=tf.cast(tf.linalg.band_part(weights, -1, 0), tf.bool))
        return weights @ value

def _create_model_architecture() -> tf.keras.Model:
    '''
    returns: tf.keras.Model 
    '''
    inputs = tf.keras.Input(_params["context_length"])
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = _embedding_layer()(x)
    x = _self_attention_layer()(x)
    outputs = tf.keras.layers.Dense(units=vocab_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Define how we get each batch
def _generate_batch(split):
    
    # Define data as data_dict.get(split)
    data = data_dict.get(split)
 
    # Extract random indices
    shape = (_params.get("batch_size"),)
    minval = 0
    maxval = len(data) - _params.get("context_length")
    dtype = tf.dtypes.int64
    random_index_list = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    
    context = tf.stack([data[index:index+_params.get("context_length")] for index in random_index_list])
    target = tf.stack([data[index+1:index+_params.get("context_length")+1] for index in random_index_list])

    while True:
        yield context, target

model = _create_model_architecture()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(_params["learning_rate"]),
    run_eagerly=False,
)
history = model.fit(
    _generate_batch("train"),
    epochs=_params["epochs"],
    validation_data=_generate_batch("validation"),
    steps_per_epoch=_params["steps_per_epoch"],
    validation_steps=_params["validation_steps"]
)


def _generate(context: tf.Tensor, max_next_tokens: int) -> int:
    for _ in range(max_next_tokens):
        model.predict(context[:,-_params["context_length"]:])
        logits = y_pred[:,-1,:]
        next_index = tf.random.categorical(logits=logits, num_samples=1)
        context = tf.concat([context, next_index], axis=1)
    return context

context = tf.constant([[0,32,53,1],[32,53,1,40]], dtype=tf.int64)
decode(_generate(context, max_next_tokens=50).numpy()[0].tolist())

#class _Mini_Language_Model(tf.keras.Model):
#    def __init__(
#        self,
#        
#    ):
#        super().__init__()
