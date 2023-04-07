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
_params["batch_size"] = 4
_params["context_length"] = 8

# Define how we get each batch
def get_batch(split):
    
    # Define data as data_dict.get(split)
    data = data_dict.get(split)
 
    # Extract random indices
    shape = (_params.get("batch_size"),)
    minval = 0
    maxval = len(data) - _params.get("context_length")
    dtype = tf.dtypes.int32
    random_index_list = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    
    context = tf.stack([data[index:index+_params.get("context_length")] for index in random_index_list])
    target = tf.stack([data[index+1:index+_params.get("context_length")+1] for index in random_index_list])

    # Return context and target
    return context, target


