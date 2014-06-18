## Recursive Covariance Estimation using Fixed Forgetting Factors (FFFs) or Adaptive Filters ##
#
#
# fixed forgetting factor & adaptive filtering implemented as described in:
# "Online Linear and Quadratic Discriminant Analysis with Adaptive Forgetting for Streaming Classification", Anagnostopoulos et al 2012.
#
# TODO:
#  - adaptive filtering code

import numpy

class CovEstFF():
    """Class for recursive covariance estimation
    It is assumed data arrive at equally spaced points in time
    
    The FF mean is stored in mu as a numpy array
    The FF sample covariance is stored in S as a list of numpy arrays (each entry an estimate of the sample covariance)
    
    NOTE: we may wish to change the way mu and S are stored so that only the most recent estimates are kept (as opposed to all of them as is the case now)
    
    """
    
    def __init__(self, data, l):
    	self.w = 1. # normalising constant
	self.l = float(l) # fixed forgetting factor
	self.Pi = numpy.zeros((data.shape[1], data.shape[1]))
	if len(data.shape)==1:
	    # only one data point provided, initialise mean to 
	    self.mu = data
	    self.S = [numpy.outer(data, data)]
	else:
	    self.mu = data[0,:].reshape((1, data.shape[1]))
	    self.S = [numpy.outer(data[0,:], data[0,:])]
	    for i in range(1, data.shape[0]):
		#print self.w
		self.w = self.l*self.w + 1.
		self.mu = numpy.vstack((self.mu, (1.- (1./self.w))*self.mu[-1,:] + (1./self.w)*data[i,:] ))
		self.Pi = (1.- (1./self.w))*self.Pi + (1./self.w)*numpy.outer(data[i,:], data[i,:] )
		self.S.append( self.Pi -  (1./self.w)*numpy.outer(self.mu[-1], self.mu[-1]))
		#self.S.append( (1.- (1./self.w))*self.S[-1] + (1./self.w)*numpy.outer(data[i,:]-self.mu[-1,:], data[i,:]-self.mu[-1,:] ))
		    
	
    def __repr__(self):
	mes = " ### Fixed forgetting factor estimation ###\n"
	mes += " # Forgetting factor: "+ str(self.l) + '\n'
	mes += " # Mean and sample covariance estimated for " + str(len(self.S)) + " observations\n"
	return mes
	
    def updateS(self, newX):
	"""Function  to update estimates of mean and covariance given new observations
	
	newX is a new data point which we use to update mean, mu, and sample covariance, S.
	
	"""
	# update normalising constant:
	self.w = self.l*self.w + 1
	# update mu:
	self.mu = numpy.vstack((self.mu, (1.- (1./self.w))*self.mu[-1,:] + (1./self.w)*newX ))
	# update covariance:
	self.Pi = (1.- (1./self.w))*self.Pi + (1./self.w)*numpy.outer(newX, newX )
	self.S.append( self.Pi -  (1./self.w)*numpy.outer(self.mu[-1], self.mu[-1]))
	#self.S.append((1.- (1./self.w))*self.S[-1] + (1./self.w)*numpy.outer(newX-self.mu[-1,:], newX-self.mu[-1,:] ))
	
	
	
	
	
	
	
	
	
	
	
    


