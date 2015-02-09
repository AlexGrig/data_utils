# -*- coding: utf-8 -*-
"""
Author: Alexander Grigorevskiy. Aalto University, 2013.

"""

import numpy as np
from . import data_utils as du

class Inverse_Update(object):
    """    
    Instance of this class provide the fuctions which updates the inverse
    of a Gram matrix. Even more general than inverse because diagonal matrix
    can be added. 
    
    Function returns update to this inverse
    ( X.T * X  + Diag )^(-1)    
    
    when new columns to matrix X is added.    
    
    Details are given in:
    Mark JL Orr. Introduction to Radial Basis Function Networks.
    Centre for Cognitive Science, University of Edinburgh, Technical
    Report, April 1996. 
    """
    
    def  __init__(self):   
        """
        Constructor
        """
        
        self.prev_inv = None # inverse from previous iteration
        self.prev_size = None #  size of the previous inverse (symmetric)
    
    def new_column(self, new_col, new_lambda=0, X=None, prev_inverse=None):     
        """
        Returns the updated inverse matrix. Original parameters are not modified. 
        
        Input: 
            new_col - new colum of matrix X
            new_lambda - (optional, float) new diagonal value
            X - old matrix X (without new column)
            prev_inverse - if we want to start not from the first iteration.
                            We provide already computed inverse matrix
        Output:
            New inverse matrix
        """
        
        if not isinstance(new_col, np.ndarray):
            raise AssertionError("Class Inverse_Update, function new_column: new_col is not ndarray derivative, instead %s", type(new_col) )
            
        new_col = np.matrix( new_col.reshape(du.vector_len(new_col),1  ) ) # copy of new_col and make it column matrix
            
        if new_col.shape[0] == 1: 
            new_col = new_col.T # make sure this is column vector
            
        if prev_inverse:
            self.prev_inv = np.matrix( prev_inverse ) # copy of new_col and make it a matrix
            self.prev_size = self.prev_inv.shape[0] # prev_inv is a square matrix

                
        if self.prev_inv is None: # first iteration
            
            self.prev_inv = 1./ ( new_col.T * new_col + new_lambda)
            self.prev_size = 1
            
            return self.prev_inv
            
        else: # not the first iteration
            n = self.prev_size
            
            
            # Part 1 of the formula: add one more zero row and zero column
            P1 = np.hstack(  ( self.prev_inv, np.matrix( (0,)* n ,dtype='d').T ) )
            P1 = np.vstack( (P1, np.matrix( (0,)* (n+1),dtype='d')) )
            
            T = self.prev_inv * X.T # temporary expression
            
            delta = new_lambda + new_col.T * ( new_col - X * ( T * new_col) )
            delta = delta[0,0] # to make constant
            
            v = np.vstack(( T * new_col, -1)) # vector for part 2        
            
            # Part 2
            P2 = np.outer(v,v)
            
            self.prev_inv = P1 + 1/delta * P2
            self.prev_size = self.prev_inv.shape[0]
            
            return self.prev_inv            

def sqr_euclid(v):
    """
    Square of euclidean norm for a vector.
    
    Introduced because sometimes squared norm is enough and square root 
    computation is an extrta cost.
    
    Input:
        v - input vector
        
    Output:
        Squared euclidean norm.
    """
    
    return ((v.conj()*v).real).sum()
        
