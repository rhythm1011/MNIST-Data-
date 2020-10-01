import os
import sys
import argparse
import tensorflow as tf
from Model import MNIST
from DataReader import load_data, train_valid_split
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def configure():
	flags = tf.app.flags

	flags.DEFINE_integer('num_hid_layers', 3, 'the number of hidden layers')
	flags.DEFINE_integer('num_hid_units', 256, 'the number of hidden units in hidden layers')
	flags.DEFINE_integer('batch_size', 128, 'training batch size')
	flags.DEFINE_integer('num_classes', 10, 'number of classes')
	flags.DEFINE_string('modeldir', 'model', 'model directory')
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    
	#sess = tf.Session()
	print('---Prepare data...')
	x_train, y_train, x_test, y_test = load_data()
	x_train_new, y_train_new, x_valid, y_valid \
				= train_valid_split(x_train, y_train)

	
	# model = MNIST(sess, conf())
	### YOUR CODE HERE
	conf = configure()

	# First run: use the train_new set and the valid set to choose
	# hyperparameters, like num_hid_layers, num_hid_units, stopping epoch, etc.
	# Report chosen hyperparameters in your hard-copy report.
	params = {
	'num_hid_layers': [0, 1, 2, 3, 4, 5],
	'num_hid_units': [64, 128, 256, 512],
	'max_epoch': [50, 100, 125, 150, 175, 200],
	'batch_size': [128, 256, 512, 1024, 2048]
	}

	#lowest supported by python
	best_accuracy = -sys.maxsize -1
	best_params = None

	for batch_size in params['batch_size']:
		conf.batch_size = batch_size
		for num_hid_units in params['num_hid_units']:
			conf.num_hid_units = num_hid_units
			for num_hid_layers in params['num_hid_layers']:
				conf.num_hid_layers = num_hid_layers
				max_epoch = max(params['max_epoch'])
				sess = tf.Session(graph=tf.get_default_graph())
				model = MNIST(sess, conf)
				model.train(x_train_new, y_train_new, x_valid, y_valid, max_epoch, validation=False)
				for epoch in params['max_epoch']:
					accuracy = model.test(x_valid, y_valid, epoch)
					print ("Accuracy with", num_hid_units, "hidden Units,", batch_size, "batch_size,", 
						num_hid_layers, "hidden Layers and", epoch, "epoch:", accuracy)
					if accuracy > best_accuracy:
						best_params = (batch_size, num_hid_units, num_hid_layers, epoch)
						best_accuracy = accuracy
				sess.close()
				tf.reset_default_graph()

	print ("Best Accuracy with", best_params[1], "hidden Units,", best_params[0], "batch_size,", 
						best_params[2], "hidden Layers and", best_params[3], "epoch:", best_accuracy)


	# Second run: with hyperparameters determined in the first run, re-train
	# your model on the original train set.
	sess = tf.Session(graph=tf.get_default_graph())
	conf.batch_size, conf.num_hid_units, conf.num_hid_layers, max_epoch = best_params
	model = MNIST(sess, conf)
	model.train(x_train, x_train, x_valid, y_valid, max_epoch, validation=False)

	# Third run: after re-training, test your model on the test set.
	# Report testing accuracy in your hard-copy report.
	accuracy = model.test(x_test, y_test, max_epoch)
	sess.close()
	
	### END CODE HERE

if __name__ == '__main__':
	tf.app.run()
