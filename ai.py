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
import time
from rl.src import fast_predict

"""
TODO: Figure out a way to upload a saved model so that predictions can be made about
 a given board
"""

class AI(object):
	
	def __init__ (self, evaluationMethod, depth, name=None):
		"""
		Initializes a new AI player
		
		Args:
			depth - int - specifies the depth that trees need to reach, default is 3
			
			evaluationMethod - which method of evaluation (static, or trained)
							   where static is handcrafted and trained uses our classifier
			
			name - string - name of the AI (optional)
		"""
		self.depth = depth
		#TODO: Add a non-estimator based static eval method (maybe material advantage or something else simple)
		self.evaluationMethod = evaluationMethod
		self.name=name
		
		

	def getBestMove(self, board):
		"""
		Uses alphaBetaNegamax to determine the best move
		
		Args:
			board - current chess.Board() object
		
		Returns:
			optimal move
		"""
		optimal_move = None
		
		high_score = -10
		
		for move in board.legal_moves:
			copy = chess.Board(board.fen())
			copy.push(move)
			score = -self.alphaBetaNegamax(copy, self.depth, -(sys.maxsize) + 1, sys.maxsize - 1, board.turn);
			if score > high_score:
				high_score = score
				optimal_move = move

		return optimal_move
	
	def alphaBetaNegamax(self, board, depth, alpha, beta, turn):
		"""
		Determines optimal move using negamax version of alpha beta pruning
		
		Takes advantage of the fact that max(a,b) = -min(-b,-a)
		
		Args:
			board - chess.Board() object -- used to generate legal moves and for
					evaluation of a given move
			
			depth - how deep we want to explore in the tree
			
			alpha, beta - values to record max/min for a given leaf in alpha_beta pruning
			
			turn - True if white's turn ; False if black's turn
		
		Returns:
			value of optiomal move
		"""
		#Check if we are at max depth  -- evaluate the board using evaluationMethod
		if depth <= 0:
			# TODO: Provide mechanism for evaluation method to be static
			start = time.time()
			result_dict = list(self.evaluationMethod.predict(self.generatePredictFn(board), yield_single_examples=True))
			# result_dict = self.evaluationMethod.predict(p.fen_to_2d(board))
			# print(fast_predict.FastPredict(self.evaluationMethod.predict(input_fn=self.generatePredictFn(board))))
			probabilities = result_dict[0]['probabilities']			
			# Basically return probability white wins if it is white's turn otherwise return black's probability to win
			return probabilities[0] if turn == True else probabilities[2]		
 
		#starting value is minimum possible value + 1 since minimum is default		
		score = -(sys.maxsize) + 1
		
		#Iterating over move tree
		for move in board.legal_moves:
			print("Turn: ", turn)
			#We don't want to modify our current board so we simply create a child copy
			child = chess.Board(board.fen())
			#Now we update the child with the current move
			child.push(move)
			#NOTE -- alpha and beta args are swapped to reflect back passing
			score = -(self.alphaBetaNegamax(child, depth - 1, -beta, -alpha, turn))
			
			print("score: ", score)
			
			#<------ Return Conditions -------->
			if score >= beta:
				print("pruned NOW")
				return score
			if score > alpha:
				print("pruned NOW")
				alpha = score
		
		#Negamax allows us to return alpha since this will just be flipped each time
		return alpha
	
	def generatePredictFn(self,board):
		"""
		Generates a predict function for our estimator
		
		Args:
			board - chess.Board() object that we need to make a prediction on
		
		Returns:
			tf.estimator.inputs.numpy_input_fn -- to be used as input_fn for estimator.predict
		"""
		return tf.estimator.inputs.numpy_input_fn(
								x={"x" : np.array(p.fen_to_2d(board))},
								num_epochs=1,
								shuffle=False
								)
		