import tensorflow as tf

"""This script defines the network.
"""

class MLP(object):

	def __init__(self, num_hid_layers, num_hid_units, num_classes):
		"""Define hyperparameters.

		Args:
			num_hid_layers: A positive integer.
				Define the number of hidden layers.
			num_hid_units: A positive integers. 
				Define the number of hidden units in hidden layers.
			num_classes: A positive integer.
		"""
		self.num_hid_layers = num_hid_layers
		self.num_hid_units = num_hid_units
		self.num_classes = num_classes

	def __call__(self, inputs, training):
		"""Add operations to classify a batch of input images.

		Args:
			inputs: A Tensor representing a batch of input images.
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			A logits Tensor with shape [<batch_size>, self.num_classes].
		"""
		outputs = self._hidden_layers(inputs, training)

		return self._output_layer(outputs, training)

	################################################################################
	# Blocks building the network
	################################################################################
	def _hidden_layers(self, inputs, training):
		"""Implement the hidden layers according to self.num_hid_layers
		and self.num_hid_units.

		Args:
			inputs: A Tensor with shape [<batch_size>, 784].
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			outputs: A Tensor with shape [<batch_size>, self.num_hid_units].
		"""
		
		### YOUR CODE HERE

		# Note: for tensorflow APIs, only those in tf.layers and tf.nn
		# are allowed to use.
		current_layer = inputs
		for layer in range(self.num_hid_layers):
			current_layer = tf.layers.dense(inputs=current_layer, units=self.num_hid_units, activation=tf.nn.relu,
				use_bias=True, trainable = training, name="Hidden_Layer"+str(layer+1), reuse=tf.AUTO_REUSE)
			#current_layer = tf.layers.dropout(inputs=current_layer, rate=0.2, name="Dropout_Layer"+str(layer+1))

		outputs = current_layer
		### END CODE HERE
		return outputs

	def _output_layer(self, inputs, training):
		"""Implement the output layer.

		Args:
			inputs: A Tensor with shape [<batch_size>, self.num_hid_units].
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			outputs: A logits Tensor with shape [<batch_size>, self.num_classes].
		"""

		### YOUR CODE HERE
		
		# Note: for tensorflow APIs, only those in tf.layers and tf.nn
		# are allowed to use.
		outputs = tf.layers.dense(inputs = inputs, units = self.num_classes,
			activation=tf.nn.softmax, use_bias=True, trainable = training, name="OutputLayer", reuse=tf.AUTO_REUSE)
		### END CODE HERE

		return outputs

















