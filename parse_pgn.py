#/usr/bin/env python
import chess
import chess.pgn
import numpy as np
import fenparsermaster.fenparser as f
import time

center_board = { 2 : [3,4], 3 : [2,3,4,5],
				 5 : [3,4], 4 : [2,3,4,5]}

"""
TODO:

Once we run our model using our basic evaluation methods we should really be
testing on two of the following

1. Just using who one (3 choices)
2. Using who won in conjunction with current method using material advantage and
   central control
   
"""


#Big pgn = ficsgamesdb_201602_standard2000_nomovetimes_1546990.pgn
def read_all_games(file="kasparov-deep-blue-1997.pgn", mode=0):
	""" Reads and parses all games in a given pgn file, storing boards and results in a tuple
	
	TODO:
		Come up with meaningful data structure to return for training on cnn
		
	Args:
		file - .pgn file to be parsed -- Deaults to set of games from 2016 for now
		
	Returns:
		Tuple:
		2-D array of all the training data ; 1-D array of who has advantage for a given board
		
	"""
	pgn = open(file)
	
	game = chess.pgn.read_game(pgn) #reads in one game of pgn file
	win_count_dict = {"W" : 0, "B" : 0, "T" : 0} #Winner results
	training_data = []
	training_labels = []
	
	while game != None:
		#representation of the board
		board = game.board() 
		#Iterates over moves of a game
		tmp = [] #tmp data used for storing current move set
		for move in game.main_line():
			board_2d = np.array(fen_to_2d(board))
			#Adds board to tmp array -- to be appended to training data
			training_data.append(board_2d)
			training_labels.append(calc_advantage(board_2d, game, mode))
			#Game is a stack of moves so "pushes" next move onto stack
			board.push(move)
		
		 #Game is over let's check the next game
		game = chess.pgn.read_game(pgn)
		tmp = []
	print("Length of training_data {}".format(len(training_data)))
	print("Read them all!")
	return (np.asarray(training_data), np.asarray(training_labels))

def calc_advantage(board, game=None, mode=0):
	""" Calculates advantage based on 3 basic metrics -- material advantage, who controls the middle, and
		who won the game
		
		TODO: Maybe add some advantage if the person won the game or not when weighing moves so if
		the winner made a move that seems bad it should not be penalized as heavily (?)
	
	Args:
		board -- 2d representation of board -- used to calculate material advantage and who controls middle
		game  -- game object from chess -- used to get "Result" header
		mode  -- default is zero -- if switched to 1 just takes into account the winner of the game 
		
	Returns:
		One hot vector indicative of who has the "advantage" """
	
	start = time.time()
	
	if mode == 0:
		#Indicates who has the 'advantage' based on a combination of # and qualoity of pieces
		material_advantage = sum(map(sum, board))
		
		#Evaluate the middle of the board
		center_advantage = calc_center_control(board)
		
		#Result is summation of material advantage and who controls the center
		rst = material_advantage if material_advantage != 0 else material_advantage + center_advantage
		print(time.time() - start)
		if rst > 0.:
			return 0
		elif rst < 0.:
			return 2
		else:
			return 1
	
	elif mode == 1:
		return result_to_int(game.headers["Result"])
		

def calc_center_control(board, w=.1):
	""" Calculates sum of center squares of board to see has control over the center
	
	Args:
		board -- 2d representation of board
		w 	  -- weight to be applied to result -- deault is .1
		
	Returns:
		float indicative of who is winning -- Positive = White, Negative = Black, 0 = Tie """
	
	rst = 0	
	#Iterate over central board locations
	for key in center_board:
		for val in center_board[key]:
			rst = rst + board[key][val]
		
	return w * (rst)

def result_to_int(s):
	""" Gives float representation of win result given a string of the form "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (tie)
	
	Args:
		s - string to parse
	Retunrs:
		float: 0 - White wins ; 2 - Black wins ; 1 - Tie
	
	"""
	if s == "1-0":
		return 0.
	elif s == "0-1":
		return 2.
	else:
		return 1.
	
def result_to_one_hot(s):
	""" Gives one hot representation of win result given a string of the form "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (tie)
	
	Args:
		s - string to parse
	Retunrs:
		One hot vector: [1,0,0] - White wins ; [0,0,1] - Black wins ; [0,1,0] - Tie
	
	"""
	if s == "1-0":
		return [1,0,0] 
	elif s == "0-1":
		return [0,0,1]
	else:
		return [0,1,0]

		
def fen_to_2d(board):
	""" Converts FEN board representation to 2d board representation compliant with python chess structure
		Positive values for white pieces ; Negative values for corresponding black pieces
		
	Args:
		board - chess.Board object
	
	Returns:
		result - 8x8 2-d array representing the board
	"""
	
	fen_parser = f.FenParser(board.fen())
	#Gives 2d string representation
	result = fen_parser.parse()
	
	piece_map =  {'p' : np.float32(-1.), 'n' : np.float32(-5.), 'b' : np.float32(-7.), 'r' : np.float32(-8.), 'q' : np.float32(-20.), 'k' : np.float32(-999.),
				  'P' : np.float32(1.),  'N' : np.float32(5.),  'B' : np.float32(7.),  'R' : np.float32(8.),  'Q' : np.float32(20.),  'K' :  np.float32(999.), ' ' : np.float32(0.) }
	
	for i in range(len(result)):
		for j in range(len(result[i])):
			key = result[i][j]
			result[i][j] = piece_map[key]
			
	return result
