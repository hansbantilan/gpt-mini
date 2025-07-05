Create a class called TorchCharGpt that inherits from the Gpt base class and that is intended to be an implementation of a character-level GPT model using the pytorch framework.

Implement this in torch_char_gpt.py and place it in the same directory as the currently existing tensorflow_char_gpt.py.

Make sure that the GptFactory is updated with this new torch_char_gpt implementation.

Closely mirror the implementation in the currently existing TensorflowCharGpt class, using the same member function and member class names wherever possible.

This ports the current tensorflow implementation in TensorflowCharGpt to a pytorch implementation in TorchCharGpt.

Use torch==2.0.0
