# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:12:15 2021

@author: Teddy
"""

import chess
import chess.svg
import chess.engine
import numpy
import random

import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks

########################
#        Chess
########################
squares_index = {
  'a': 0,
  'b': 1,
  'c': 2,
  'd': 3,
  'e': 4,
  'f': 5,
  'g': 6,
  'h': 7
}
# example: h3 -> 17
def square_to_index(square):
  letter = chess.square_name(square)
  return 8 - int(letter[1]), squares_index[letter[0]]


# create 14x8x8 board to contain the location of each type of piece
# two extra 8x8 matrices for possible attacks each side can do
def board_to_matrices(board):
    board3d = numpy.zeros((14, 8, 8), dtype=numpy.int8)
    
    for piece in chess.PIECE_TYPES:#iterate through each piece type
        for square in board.pieces(piece, chess.WHITE):#iterate through the position of each piece
            idx = numpy.unravel_index(square, (8, 8))#convert numerical board location to 8x8 index
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1#-1 for 0 index, 7 - to have white at bottom
    
    #repeat for black pieces        
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1
    # add attacks and valid moves too
    # so the network knows what is being attacked
    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux
    return board3d
    
#randomly select n amount of legal moves and evaluates them. Return highest eval move
def get_best_rand_move(board, model, n):
    if len(list(board.legal_moves)) < n:
        n = len(list(board.legal_moves))
    moves = random.sample(list(board.legal_moves),n)
    evals = [0]*n
    i = 0
    for m in moves:
        boardt = board.copy()
        boardt.push(m)
        inp = board_to_matrices(boardt)
        inp = inp[None]
        evals[i] = model.predict(inp)
        i = i+1
        del(boardt)
    mx = max(evals)
    ind = evals.index(mx)
    return moves[ind]

#plays randomly selected move
def playGameNN(board):
    model = tf.keras.models.load_model(r"AI/supervised_chessAI_nov_4th.h5")
    n = 30
    move = get_best_rand_move(board, model, n)
    move_str = str(move)
    return move_str   

