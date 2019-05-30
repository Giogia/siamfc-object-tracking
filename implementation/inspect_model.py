from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf

# Use this line to print all the tensor in the saved model
print_tensors_in_checkpoint_file(file_name='model/model', tensor_name='', all_tensors=False, all_tensor_names=True)
