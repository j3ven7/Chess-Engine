#!/bin/usr/env python
import tensorflow as tf
import numpy as np
import parse_pgn as p
import time
import random 

def cnn_model(features, labels, mode):
	
	
	#Reshape input to 8x8 matrix
	input_layer = tf.reshape(features["x"], [-1, 8, 8, 1])
	#Convolutional layer 1 - Tune kernel_size if training fails\
	conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=64,
			kernel_size=[2,2],
			padding="same",
			activation=tf.nn.relu,
			name="conv1")

	#Convolutional layer 2 
	conv2 = tf.layers.conv2d(
			inputs=conv1,
			filters=64,
			kernel_size=[4,4],
			padding="same",
			activation=tf.nn.relu,
			name="conv2")

	#Flatten our feature map so our tensor only has two dimensions -- fed into dense layer
	conv2_flat = tf.reshape(conv2, [-1, 8*8*64])

	#Builds our fully connected dense layer
	dense = tf.layers.dense(inputs=conv2_flat, units=1024, activation=tf.nn.relu, name="dense")

	#Dropout layer to prevent over-regularization to our dense layer -- change rate for training
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN, name="dropout")

	#Logits layer -- return the raw values (probabilities) of our predictions -- 3 representing who wins (may change to material advantage)
	logits = tf.layers.dense(inputs=dropout, units=3, name="logits")

	"""Predictions -- "classes" labels our inputs - takes max from "probabilities" which gives probability that a feature vector is a
	win, tie, or loss -- probabilities derived using softmax activation"""
	
	predictions = { 
		"classes" : tf.argmax(input=logits, axis=1),
		"probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
	}

	print("MODE IS : " + mode)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	#Convert labels into one hot vectors -- NOW WE DO THIS AHEAD OF TIME
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)

	#Calculating loss - cross entropy used for multiclass classification problems
	loss = tf.losses.softmax_cross_entropy(
	   onehot_labels=onehot_labels, logits=logits)

	#<------ Configuring Training Operations Starts Here ---------->
	
	#Optimize loss value -- stochastic gradient descent for optimization
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss= loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	eval_metric_ops = {
		"accuracy" : tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(placeholder):
	#Get time stamp	
	#NOTE THESE NEED TO BE CHANGED FOR OPTIMAL RESULTS
	train_data, train_labels = p.read_all_games(file="ficsgamesdb_201401_chess2000_nomovetimes_1549357.pgn")
	eval_data, eval_labels = p.read_all_games(file="ficsgamesdb_201602_standard2000_nomovetimes_1546990.pgn")
	
	#Just testing to see if prediction is viable option for training our alpha beta pruning
	test_prediction = train_data[random.randint(0,len(train_data) - 1)]
	test_label 		= train_labels[random.randint(0,len(train_data) - 1)]
	
	# Create the Estimator
	classifier = tf.estimator.Estimator(
		model_fn=cnn_model, model_dir="/Users/joshua/Documents/Cornell_Documents/CS_4700/chess_stuff/tmp")
	
	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=500)
	
	# Train the model -- MAY WANT TO TURN SHUFFLING OFF
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=100,
		num_epochs=None,
		shuffle=True)
	
	# Builds function for making predictions -- might want to use this for alpha-beta priuning
	
	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		num_epochs=1,
		shuffle=False)
	
	classifier.train(
		input_fn=train_input_fn,
		steps=100, #20,000 For training to start - lower if it's overfitting
		 hooks=[logging_hook])
	
	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False)
	
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)
	
	#THIS ALLOWS YOU TO MAKE PREDICTIONS
	predict_results = list(classifier.predict(input_fn=predict_input_fn))
	print(predict_results)
	#saver = tf.train.Saver()
	
	#with tf.Session as sess:
		#saver.restore(sess, INSERT SAVE PATH)
		
	
if __name__ == '__main__':
	start_time = time.time()
	tf.app.run()
	print("--- %s seconds ---" % (time.time() - start_time))



