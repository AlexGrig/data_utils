# -*- coding: utf-8 -*-

"""
In this module collection of useful data manipulation utils are provided.

Author: Alexander Grigorievskiy, Aalto University, 2013.
"""

import numpy as np
try: # if bottle neck is not imported, use numpy functions
    import bottleneck as bn
except ImportError:
    bn = np 


def normalize(D, ntype=0, means=None, stds=None):
    """
    Function which mormalizes the data. Rows of D - data points.
    
    Original varible D is not modified, new variables are allocated from scratch.
    
    When external means and stds are not provided, means and stds
    from the data are calculated.    
    
    Input:
        D - data
        ntype - normalization type
            0 - zero mean unit variance
            1 - zero mean, [-1,1]
            2 - zero mean, unit length (equclidean norm)
            3 - only zero mean, do not scale the variance
            
        means - external means. For instance means of part of the dataset
        stds - external stds. For instance means of part of the dataset
    Output:
        result - normalized vector
        means - vector of column means
        stds - change of scale. In case ntype=0 these are standard deviations
    """
    
    if (not isinstance(D,np.ndarray)) or (len(D.shape) > 2):
        raise AssertionError("Input D must be derivative of numpy.ndarray and have less than 3 dimensions.")
    
    (D,initial_shape) = ensure_column(D)
            
    n_rows = D.shape[0]    
    
    if means is None:
        means = bn.nanmean(D, axis= 0)    
    
    tmp = D - np.tile( means, (n_rows,1) ) # temporary result. Data with 
       # substracted mean                                                                         
  
    if stds is None:
        if (ntype == 0):    
            stds = bn.nanstd(tmp,axis=0, ddof=1 ) # one degree of freadom as matlab default
            
        elif (ntype == 1):
            stds = bn.nanmax(np.abs(tmp), axis=0)
            
        elif (ntype == 2):    
            stds = np.sqrt( bn.nansum( np.power(tmp,2) , axis = 0) )                
        
        elif (ntype == 3):    
            stds = np.ones( (D.shape[1],) )
            
        else:
            raise ValueError("Normalization type %s is unknown" % ntype)
    
    # result = np.dot(  tmp ,  np.diagflat( 1./stds  ) )
    result = np.divide( tmp, stds ) 
    
    result = rev_ensure_column(result,initial_shape)
    D = rev_ensure_column(D,initial_shape)    
    
    return (result,means,stds)
    
    
def denormalize(D, means, stds=None):
    """
    Denormalizes the data using means and stds.
    
    Original varible D is not modified, new variables are allocated from scratch.
    
    Output:
        result - denormalized vector
    """    
    
    (D,initial_shape) = ensure_column(D)    
    
    n_rows =  D.shape[0]    
    
    if stds is not None:
        result = np.multiply(  D, stds  ) + np.tile( means, (n_rows,1) )
    else:
        result = D + np.tile( means, (n_rows,1) )
    
    result = rev_ensure_column(result,initial_shape)
    D = rev_ensure_column(D,initial_shape)    
    
    return result    


def ensure_column(v):
    """
    Function affects only one dimensioanl arrays ( including (1,n) and (n,1) dimensional)
    It then represent the output as a (n,1) vector strictly. It also returns
    initial shape by using which original dimensions of the vector can be reconstructed.
        
    For more than one dimensional vector function do nothing.
        
    Inputs:
        v - array
        
    Output:
        col - the same array with (n,1) dimensions - column
        params - params by which original shape can be restored
    """
    
    initial_shape = v.shape
    if len(initial_shape) == 1: # one dimensional array
        v.shape = (initial_shape[0],1)
    else:
        if (len(initial_shape) == 2)  and (initial_shape[0] == 1): # row vector        
            v.shape = (initial_shape[1],1)     

    return v,initial_shape


def rev_ensure_column(v,initial_shape):
    """
        This function is reverse with respect to ensure_coulmn
        It restores the original dimensions of the vector
        
    """
    if initial_shape: # check that the tuple is nonempty
        v.shape = initial_shape    
    
    return v
    
    
def vector_len( vector ):
    """
    Function determines dimension of a vector.
    E. g. if vector has a shape (1,len) or (len,1) or (len,)
    it returns len.
    
    Input:
        vector - vector of type ndarray or matrix. If both dimensions are larger 
                 than 1 error is returned.
                 
    Output:
        len - length of a vector.
    """
        
    if not isinstance(vector, np.ndarray ):
        return len(vector)
    else:
        shape = vector.shape # shape is a tuple
        
        sl = len(shape)
        if sl == 0:
            return 0
        elif sl == 1:
            return shape[0]
        else: 
            non_one_dims = [ s for s in shape if s > 1 ]
            non_one_dims_len = len(non_one_dims)
            if non_one_dims_len > 1:
                raise ValueError("Function vector_len: Not a vector provided, shape : %s", shape)
            elif non_one_dims_len == 0:
                return 1
            else:
                return non_one_dims[0]
