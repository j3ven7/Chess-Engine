#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import chess.pgn as cp
import parse_pgn as p
#TODO: Write functions to fit and predict 

class Evaluator:
	def __init__(self):
		"""
		Args
			X_placeholder -- moves of a game -- input layer
			Y_placeholder -- labels / results of a game -- input layer
			predict_op    -- result of prediction
			optimizer     -- Optimizer we use for training against cross_entropy loss function
			accuracy      -- how we determine accuracy of our predictions against our labesl
	
		"""
	
		self.X_placeholder = tf.placeholder(tf.float32, [None, 64], name="X") #64 is board size
		self.Y_placeholder = tf.placeholder(tf.float32, [None, 3], name="Y") #3 is possible outcomes
		self.learning_rate = .1
		self.keep_prob_placeholder = tf.placeholder(tf.float32)
		self.predict_op = None
		self.optimizer = None
		self.accuracy = None
		self.make_model()
	
	def make_model(self):
		X = tf.reshape(self.X_placeholder, [-1, 8, 8, 1])
	
		#convolutional layer1
		conv_weights = {1 : self.weight_variable([4,4,1,64]), 2 : self.weight_variable([2,2,64,64])}
		conv_biases = {1 : self.bias_variable([64]), 2 : self.bias_variable([64])}
		
		model = self.conv_layer(X, conv_weights[1], conv_biases[1], name='conv1') 	

		#convolutional layer2 -- takes input from conv1
		model = self.conv_layer(model, conv_weights[2], conv_biases[2], name='conv2')		
		model = tf.reshape(model, [-1, 2*2*64]) #64 is board size

		#fully connected layer
		fc_weight = self.weight_variable([2*2*64, 1024])
		fc_bias = self.bias_variable([1024])
		#Not using sigmoid to avoid density issues
		model = tf.nn.relu(tf.matmul(model, fc_weight) + fc_bias)

		#dropout layer
		model = tf.nn.dropout(model, self.keep_prob_placeholder)

		w_out = self.weight_variable([1024,3])
		b_out = self.bias_variable([3])
	
		y_predicted = tf.matmul(model, w_out) + b_out
		self.predict_op = y_predicted

		#Setting loss function
		with tf.name_scope("cross_entropy"):
			cross_entropy = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_placeholder, logits=y_predicted))
	
		#Setting optimizer -- Adam optimizer using cross entropy as loss function
		with tf.name_scope("train"):
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
	
		#Setting accuracy rules for reading labeled examples
		with tf.name_scope("accuracy"):
			correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(self.Y_placeholder, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			
	def weight_variable(self, shape):
		""" Just used to account for issue found here https://github.com/tensorflow/tensorflow/issues/9243 """
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name='W')

	def bias_variable(self, shape):
		""" Just used to account for issue found here https://github.com/tensorflow/tensorflow/issues/9243 """
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, name='B')
	
	def conv_layer(self,X, W, b, name='conv'):
		""" Builds a convolutional layer
		
		Args:
			X - input to the neural net -- in this case a 1D representation of the board
			W - weight vector
			b - bias term
		
		Return:
			tf.nn.max_pool using relu activation function """
		with tf.name_scope(name):
			convolution = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
			activation = tf.nn.relu(convolution + b)
	
			tf.summary.histogram('weights', W)
			tf.summary.histogram('biases', b)
			tf.summary.histogram('activation', activation)
	
			return tf.nn.max_pool(activation, ksize=[1, 2, 2, 1],
								  strides=[1, 2, 2, 1], padding='SAME')		
		
	def fit(self,
            training_data,
            testing_data,
            learning_rate=0.5,
            train_keep_prob=0.5,
            test_keep_prob=1.0):
		"""
        Trains on data in the form of a tuple of np arrays stored as (X, y),
        or the path to a folder containing 2 npy files 
        wih X named ``features.npy`` and y named ``labels.npy``.
		
        :param training_data: data used to train the network
        :type: Tuple[np.array, np.array]
        :param testing_data: data used to test the neural network
        :param epochs: Number of epochs to run when training. Defaults to 300.
        :param batch_size: Size of individual batches to 
        :param learning_rate: 
        :param train_keep_prob: 
        :param test_keep_prob: 
        """
		positions, advantages = training_data
		test_positions, test_advantages = testing_data
		saver = tf.train.Saver()
		    
		tf.global_variables_initializer()
		with tf.Session() as sess:
			
			print("Session starting")

            # Initialization
			print("Length of training data: {} Length of testing data: {}".format(len(positions), len(advantages)))			
			# Dict fed to train model
			train_dict = {self.X_placeholder:         positions,
                          self.Y_placeholder:         advantages,
                          self.keep_prob_placeholder: train_keep_prob,
                          self.learning_rate:         learning_rate}

			sess.run(self.optimizer, feed_dict=train_dict)
				
			#Test Accuracy
			accuracy_dict = {self.X_placeholder:         test_positions,
                             self.y_placeholder:         test_advantages,
                             self.keep_prob_placeholder: test_keep_prob}

			print("Test accuracy {}".format(self.accuracy.eval(accuracy_dict)))

            # tf.add_to_collection('evaluate', self.evaluate)
			saver.save(sess, self.save_path)

	def predict(self, positions):
		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess, self.save_path)
            # self.evaluate = tf.get_collection('evaluate')[0]
			advantages = self.predict_op.eval(feed_dict={self.X_placeholder: positions,
														 self.keep_prob_placeholder: 1.0})
			print("Position evaluation is {}".format(advantages))
		return advantages


	

if __name__ == '__main__':
	train_features, train_labels = p.read_all_games()
	test_features, test_labels = train_features, train_labels
	#test_features, test_labels = p.read_all_games() #REPLACE THIS WITH NON-DEFAULT FILE ARG
	evaluator = Evaluator()
	evaluator.fit(training_data=(train_features, train_labels),
				  testing_data=(test_features, test_labels),
				  learning_rate=.1)
	x = evaluator.predict(test_features)
	print(x)
