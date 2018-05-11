#/usr/bin/env python
import chess
import chess.pgn
import parse_pgn as p
import numpy as np
import fenparsermaster.fenparser as f
import train
import sys
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
import ai
import time
from rl.src.fast_predict import FastPredict
"""
TODO: Figure out a way to upload a saved model so that predictions can be made about
 a given board
"""

class Game(object):
	
	def __init__ (self, player1, player2, name=None):
		"""
		Initializes a new game
		
		Args:
			player1 - white 
			
			player2 - black
			
			name - string - name of the Game (optional)
		"""
		self.player1 = player1
		self.player2 = player2
		
	def startGame(self):
		"""
		Starts a game of chess between player1 and player2 of the Game
		
		"""
		#saver = tf.train.Saver()
		# Creates a new board
		board = chess.Board()
		optimal_move = None
		#For testing
		start_time = time.time()

		moves = []
		while (not board.is_checkmate()) or (not board.is_stalemate()):
			"""
			This is where the game is played 
				
			TODO: Include mechanism for moves to be made
					- Should be easy given that the board keeps track of whose turn it is so we can probably
					  just use the move function in ai.py
			
			TODO: Figure out how to quantify the one-hot vectors
					- Maybe use whose turn it is i.e. if black is going we check the value that black wins
						while if white is going we check probability that black wins and try to maximize that
					- May also want to include a tiebreaker that is the probability it results in a tie since
						if the probability of a tie is higher probability of a loss is inherently lower
			"""
			moves.append(board)
			if board.turn == True:
				optimal_move = self.player1.getBestMove(board)
			else:
				optimal_move = self.player2.getBestMove(board)
			
			# Play whatever the best move is
			board.push(optimal_move)
			print("Updated Board: \n " , board)
			print("--- %s seconds ---" % (time.time() - start_time))
			
		print(moves)
			
			
if __name__ == '__main__':
	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph('/Users/joshua/Documents/Cornell_Documents/CS_4700/chess_stuff/tmp/model.ckpt-100.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('/Users/joshua/Documents/Cornell_Documents/CS_4700/chess_stuff/tmp'))
			
		# Restore the tensors you want
		graph = tf.get_default_graph()
			
		chkp.print_tensors_in_checkpoint_file("/Users/joshua/Documents/Cornell_Documents/CS_4700/chess_stuff/tmp/model.ckpt-100", tensor_name='', all_tensors=True, all_tensor_names=True)
			
		# Loading the estimator to feed to the AI
		estimator = tf.estimator.Estimator(model_fn=train.cnn_model,model_dir="/Users/joshua/Documents/Cornell_Documents/CS_4700/chess_stuff/tmp")

	# Instantiate players
	player1 = ai.AI(estimator, depth=3, name="White")
	player2 = ai.AI(estimator, depth=3, name="Black")

	game = Game(player1, player2)
	#print(estimator.get_variable_value())
	game.startGame()
		