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

def lambdaFix(x):
    return max(0.7, min(x,1))

class CovEstAF():
    """Class for recursive covariance estimation
    using ADAPTIVE forgetting factors as opposed to fixed forgetting factors.
    """
    
    def __init__(self, data, alpha):
	"""
	data is numpy array of observations (possibly multivariate)
	alpha is stepsize parameter for tuning forgetting factor
	"""
	
	self.alpha = alpha
	self.lamMin = lamMin
	self.lam = 1. # initial estimate of varying forgetting factor
	self.lamTrack = [self.lam] # used to keep track
	self.n = 1 # effective sample size
	self.dn = 0 # derivative of ESS wrt forgetting factor
	if len(data.shape)==1:
	    # only one observation provided, 
	    self.p = len(data) # number of nodes
	else:
	    self.p = data.shape[1]
	
	self.mu = numpy.zeros((self.p, )) # numpy.zeros((1, self.p))
	self.dmu = numpy.zeros((self.p, )) # numpy.zeros((1, self.p)) # derivative of mean wrt forgetting factor
	self.Pi = numpy.identity(self.p) # auxiliary parameter (estimate of outer product of uncentered x)
	self.S = [numpy.identity(self.p)] # estimate of covariance
	self.dPi = numpy.zeros((self.p, self.p)) # derivative of Pi wrt forgetting factor
	self.dS =  numpy.zeros((self.p, self.p)) # derivative of covariance wrt forgetting factor
	self.invS = numpy.identity(self.p) # initial estimate of precision
	self.dinvS = numpy.zeros((self.p, self.p)) # derivative of precision wrt forgetting factor
	self.logDetS = numpy.log(numpy.linalg.det(self.S[-1]))
	self.dlogDetS = 0 # derivative of log det S wrt forgetting factor
	
	print "Running adative estiamtion..."
	for x in range(data.shape[0]):
	    self.updateSAF(newX = data[x,:])

    def __repr__(self):
	mes = " ### Adaptive forgetting factor estimation ###\n"
	mes += " # Stepsize parameter: "+ str(self.alpha) + '\n'
	mes += " # Mean and sample covariance estimated for " + str(len(self.S)) + " observations\n"
	return mes
	
    def updateSAF(self, newX):
	"""Update estimate of covariance using adaptive filtering
	
	newX is the newest observation
	"""
	# get NLL derivative
	dJ = 0.5*self.dlogDetS + 0.5*numpy.dot(newX - self.mu, numpy.dot( -2*self.invS, self.dmu) + numpy.dot(self.dinvS, newX - self.mu) )
	# update forgetting factor
	self.lam = lambdaFix(self.lam - self.alpha * dJ)
	self.lamTrack.append(self.lam) # store
	
	# update effective parameters:
	#oldn = numpy.copy(self.n)
	self.n = self.lam * self.n + 1
	self.dn = self.lam * self.dn + self.n
	
	# update mean and covariance parameters:
	oldmu = numpy.copy(self.mu)
	oldPi = numpy.copy(self.Pi)
	oldS = numpy.copy(self.S[-1])
	self.mu = (1. - 1./self.n)*self.mu + (1./self.n)*newX
	self.Pi = (1. - 1./self.n)*self.Pi + (1./self.n)*numpy.outer(newX, newX)
	self.S.append( self.Pi - numpy.outer(self.mu, self.mu) )
	
	# update parameter derivatives:
	self.dmu = (1. - 1./self.n)*self.dmu - (self.dn/(self.n * self.n))*(newX - oldmu)
	self.dPi = (1. - 1./self.n)*self.dPi - (self.dn/(self.n * self.n))*(numpy.outer(newX, newX) - oldPi)
	self.dS = self.dPi - numpy.outer(self.dmu, self.mu ) - numpy.outer(self.mu, self.dmu)
	
	# update auxiliary parameters:
	gamma = (self.n-1)*(self.n-1)/(self.n) + numpy.dot( numpy.dot(newX - self.mu, self.invS), newX - self.mu)
	dgamma = self.dn*(self.n*self.n-1)/(self.n*self.n) + numpy.dot( numpy.dot(newX - self.mu, self.dinvS), newX - self.mu) - 2*numpy.dot( numpy.dot(newX - self.mu, self.invS), self.dmu)
	
	h = numpy.outer(numpy.dot(self.invS, newX - self.mu), numpy.dot(newX - self.mu, self.invS) )
	dh_pos = numpy.outer( numpy.dot( self.dinvS, newX - self.mu), numpy.dot(newX - self.mu, self.invS)) + numpy.outer( numpy.dot(self.invS, newX-self.mu), numpy.dot(newX - self.mu, self.dinvS))
	dh_neg = -1*numpy.outer( numpy.dot(self.invS, self.dmu), numpy.dot(newX - self.mu, self.invS)) - numpy.outer(numpy.dot(self.invS, newX - self.mu), numpy.dot(self.dmu, self.invS))
	dh = dh_pos + dh_neg
	
	# update invS, dinvS, logDetS & dlogDetS efficiently*
	self.dinvS = (-1*self.dn)/((self.n-1)*(self.n-1)) * (self.invS - (1./gamma)*h) + (self.n/(self.n-1))*(self.dinvS + (dgamma/(gamma*gamma))*h - 1./gamma * dh)
	self.invS = self.n/(self.n-1) * ( self.invS - h/gamma )
	
	self.logDetS = (self.p-2) * numpy.log(self.n-1) +(1-self.p)*numpy.log(self.n ) + numpy.log(gamma) + self.dlogDetS
	self.dlogDetS = (self.p-2) * self.dn/(self.n-1) + (1-self.p)*self.dn/self.n + dgamma/gamma + self.dlogDetS
	
	
