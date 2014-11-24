## online implementation of SINGLE algorithm ##
#
#
# here at each t a new observation X_t arrives
# this is first used to update the covariance S_t (and mean mu_t).
# S_t is then used to estimate an up-to-date precision matrix \Theta_t.
#
# When estimating \Theta_t we enforce 2 constraints:
#    1) \Theta_t should be sparse (graphical lasso penalty)
#    2) the difference between \Theta_t and \Theta_{t-1} should also be sparse (similar to a fused lasso penalty, but \Theta_{t-1} remains fixed)
#
# For now a fixed forgetting factor is used to estimate covariance matrices - this will be extended in future
#

import pandas
import math
import numpy
import os
from scipy.linalg import solveh_banded
import multiprocessing
from operator import add, sub
from BurnIn import *
import pylab as plt

os.chdir('/media/1401-1FFE/Documents/RETNE/Code/')
from CovEstimation import CovEstFF, CovEstAF


class onlineSINGLE(CovEstFF, CovEstAF):
    """
    Online implementation of the SINGLE algorithm.
    """
    
    def __init__(self, data, l1, l2, ff=None, alpha=None, epsilon=None, max_iter=500, tol=0.0001):
	"""
	
	INPUT:
	- data: training dataset with which to estimate precision matrices (offline - this can be just 1 observation, but will be a "poor" initial approximation)
	- l1, l2: sparsity and temporal homogeneity parameters
	- ff: fixed forgetting factor, must be a real number between 0 and 1.
	- epsilon: optional parameter if we wish to have adaptive l1 parameter. At each step, we do a grid search over l1-epsilon, l1, l1+epsilon and the 
	best value is chosen. Best here is defined as the maximum look ahead likelihood (since observation is unseen we don't need to penalise)
	"""
	
	# sparsity & temporal homogeneity parameters
	self.l1 = l1
	self.l2 = l2
	self.iterTrack = [] # list to track number of iterations at each update
	self.max_iter = max_iter
	self.tol = tol
	
	if ff!=None:
	    print "Using a fixed forgetting factor: "+str(ff)
	    self.UpdateMethod = 'FF'
	    self.w = 1.
	    self.l = float(ff)
	    self.burnIn = numpy.floor(1/(1-self.l))/2.
	    self.Pi = numpy.zeros((data.shape[1], data.shape[1]))
	    if len(data.shape)==1:
		# only one data point provided, initialise mean to 
		self.mu = data
		self.S = [numpy.outer(data, data)]
	    else:
		# multiple datapoints recieved - calculate covariances as if it were online and then burn in for the SINGLE algorithm
		self.mu = data[0,:].reshape((1, data.shape[1]))
		self.S = [numpy.outer(data[0,:], data[0,:])]
		for i in range(1, data.shape[0]):
		    #print self.w
		    self.w = self.l*self.w + 1.
		    self.mu = numpy.vstack((self.mu, (1.- (1./self.w))*self.mu[-1,:] + (1./self.w)*data[i,:] ))
		    self.Pi = (1.- (1./self.w))*self.Pi + (1./self.w)*numpy.outer(data[i,:], data[i,:] )
		    
		    self.S.append( self.Pi -  (1./self.w)*numpy.outer(self.mu[-1], self.mu[-1]))
		    #self.S.append((1.- (1./self.w))*self.S[-1] + (1./self.w)*numpy.outer(data[i,:]-self.mu[-1,:], data[i,:]-self.mu[-1,:] ))
		    
		# convert to array (needed to run Burnin):
		Sarray = numpy.zeros((len(self.S), self.Pi.shape[1], self.Pi.shape[1]))
		for i in range(len(self.S)):
		    Sarray[i,:,:] = self.S[i]
		
		# run burn in:
		print "Running Burn in Calculation"
		self.Z = BurnInSINGLE(Sarray, l1=self.l1, l2=self.l2, tol=.001)
		#print "BURN IN DONE"

		if epsilon == None:
		    self.epsilon = None
		else:
		    self.epsilon = epsilon
		    self.Zlower = self.Z[-1] # this will be estimated precision with lower end of grid (i.e., l1-epsilon)
		    self.Zupper = self.Z[-1] # this will be estimated precision with upper end of grid (i.e., l1+epsilon)
	else:
	    print "Using a adaptive forgetting with stepsize: "+str(alpha)
	    self.UpdateMethod = 'AF'
	    self.burnIn = 10
	    self.alpha = alpha
	    self.lam = 1
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
	    
	    print "Running adative estiamtion of covariances..."
	    for x in range(data.shape[0]):
		self.updateSAF(newX = data[x,:])
		
	    Sarray = numpy.zeros((len(self.S), self.Pi.shape[1], self.Pi.shape[1]))
	    for i in range(len(self.S)):
		Sarray[i,:,:] = self.S[i]
	    
	    # run burn in:
	    print "Running Burn in Calculation"
	    self.Z = BurnInSINGLE(Sarray, l1=self.l1, l2=self.l2, tol=.001)
	    
    def updateTheta(self, newX):
	"""
	New X_t arrives. We perform the following steps:
	    1) choose l1 penalty parameter (if appropriate)
	    2) update covariance S
	    3) update precision \Theta (for various l1 values, if appropriate)
	"""
	
	if self.UpdateMethod == 'FF':
	    if self.epsilon==None:
		pass
	    else:
		# we must first choose which l1 value to use at the previous step
		ii = self.choosel1Val(newX)
		if ii==0:
		    self.Z[-1] = numpy.copy(self.Zlower)
		    self.l1 -= self.epsilon
		elif ii==2:
		    self.Z[-1] = numpy.copy(self.Zupper)
		    self.l1 += self.epsilon

	    # get new estimate of sample covariance
	    self.updateS(newX) 
	    
	    # update precision:
	    if self.w < self.burnIn:
		newTheta, conv, iter_ = getNewTheta(St=self.S[-1]+ numpy.identity(self.mu.shape[1]), oldTheta=self.Z[-1], l1=self.l1, l2=self.l2, rho=1, max_iter=self.max_iter, tol=self.tol ) # load sample covariance evalues to avoid complex solutions
		if self.epsilon != None:
		    # also calculate lower & upper estimates:
		    self.Zlower, conv = getNewTheta(St=self.S[-1]+ numpy.identity(self.mu.shape[1]), oldTheta=self.Z[-1], l1=max(0,self.l1-self.epsilon), l2=self.l2, rho=1, max_iter=self.max_iter, tol=self.tol ) 
		    self.Zupper, conv = getNewTheta(St=self.S[-1]+ numpy.identity(self.mu.shape[1]), oldTheta=self.Z[-1], l1=self.l1+self.epsilon, l2=self.l2, rho=1, max_iter=self.max_iter, tol=self.tol )
	    else:
		newTheta, conv, iter_ = getNewTheta(St=self.S[-1], oldTheta=self.Z[-1], l1=self.l1, l2=self.l2, rho=1, max_iter=self.max_iter, tol=self.tol )
		if self.epsilon != None:
		    # also calculate lower & upper estimates:
		    self.Zlower, conv = getNewTheta(St=self.S[-1]+ numpy.identity(self.mu.shape[1]), oldTheta=self.Z[-1], l1=max(0, self.l1-self.epsilon), l2=self.l2, rho=1, max_iter=self.max_iter, tol=self.tol ) 
		    self.Zupper, conv = getNewTheta(St=self.S[-1]+ numpy.identity(self.mu.shape[1]), oldTheta=self.Z[-1], l1=self.l1+self.epsilon, l2=self.l2, rho=1, max_iter=self.max_iter, tol=self.tol )
	    self.Z.append( numpy.real(newTheta)) # throw away imaginary parts (should be minimal)
	    self.iterTrack.append(iter_)
	else:
	    self.updateSAF(newX)
	    newTheta, conv, iter_ = getNewTheta(St=self.S[-1], oldTheta=self.Z[-1], l1=self.l1, l2=self.l2, rho=1, max_iter=self.max_iter, tol=self.tol )
	    self.Z.append( numpy.real(newTheta) )
	    self.iterTrack.append(iter_)

    def choosel1Val(self, newX):
	"""
	Choose regularisation penalty (l1) based on look-ahead likelihood. 
	Since this observation is unseen we don't need to penalise (i.e., do AIC/BIC type penalties)
	
	Given new observation newX, we choose from a grid search on l1 value. We search: l1-epsilon,l1, l1+epsilon
	
	"""
	
	centeredX = newX -  self.mu[ self.mu.shape[0]-1, : ]
	
	LL = numpy.array([0.]*3) # store loglikelihood (LL) for each of the 3 potential models
	
	# lower penalisation:
	LL[0] = 0.5*numpy.log( numpy.linalg.det(self.Zlower) ) - 0.5*numpy.dot( centeredX.transpose(), numpy.dot(self.Zlower, centeredX ))
	
	# current penalisation:
	LL[1] = 0.5*numpy.log( numpy.linalg.det(self.Z[-1]) ) - 0.5*numpy.dot( centeredX.transpose(), numpy.dot(self.Z[-1], centeredX ))

	# higher penalisation:
	LL[2] = 0.5*numpy.log( numpy.linalg.det(self.Zupper) ) - 0.5*numpy.dot( centeredX.transpose(), numpy.dot(self.Zupper, centeredX ))
	
	#print LL
	
	if len(numpy.unique(LL))<3:
	    ii = 1 # there is a draw, stay on same penalisation value
	else:
	    ii = LL.argmax()
	
	return ii
	
	
    def plot(self, index, ncol_=None):
	"""Plot output from rt-SINGLE algorithm"""
	
	Zarray = numpy.array(self.Z).reshape(len(self.Z), self.Z[0].shape[0], self.Z[0].shape[0])
	    
	if ncol_==None: 
	    ncol_=4
	nrow_ = numpy.ceil((len(index)*(len(index)-1)/2.) / ncol_)
	
	#fig, axes = plt.subplots(ncols=int(ncol_), nrows=int(nrow_), sharex=True, sharey=False)
	fig, axarr = plt.subplots(int(ncol_), int(nrow_), sharex=True, sharey=False)
	
	if ncol_==nrow_==1:
	    plt.plot(Zarray[:, index[0], index[1] ])
	
	else:
	    axarr = axarr.reshape((nrow_, ncol_))
	    counter_col = 0
	    counter_row = 0
	    for i in range(len(index)):
		for j in range(i+1, len(index)):
		    axarr[counter_row,counter_col].plot(Zarray[:, index[i], index[j]])
		    axarr[counter_row, counter_col].set_title("PC nodes "+str(index[i])+" and  "+str(index[j]))
		    counter_col = (counter_col+1) % ncol_
		    counter_row += int(counter_col==0)
		
	plt.show()


def getNewTheta(St, oldTheta, l1, l2, rho=1., max_iter=500, tol=.0001):
    """
    Function to estimate \Theta_t 
    
    INPUT:
	  - St: estimate of covariance at time t
	  - oldTheta: estimate of \Theta_{t-1}. Will enforce sparse differences in order to encourage temporal homogeneity
	  - l1, l2: sparity and temporal homogeneity parameters
	  - rho: stepsize parameter, can be set to 1 (usually!)
	  - max_iter: maximum number of ADMM iterations
	  - tol: convergence criterion
    
    """

    iter_ = 0 
    convergence = False
    
    # initialise theta, Z and U:
    #theta = numpy.identity(oldTheta.shape[0])
    #Z = numpy.zeros((St.shape[0], St.shape[0]))
    theta = numpy.array(oldTheta, copy=True)
    Z = numpy.array(oldTheta, copy=True)
    Zold = numpy.array(Z, copy=True)
    U = numpy.array(Z, copy=True)
    
    # ADMM iters:
    while ((convergence==False) & (iter_ < max_iter)):
	# theta step:
	theta = minimize_theta(S_ = St - rho*Z + rho * U)
	# Z step:
	Z = minimize_Z(A = theta + U, oldTheta=oldTheta, l1=l1, l2=l2)
	# U step:
	U += (theta - Z)
	
	# check convergence:
	convergence = check_conv1D(theta=theta, Z=Z, Zold=Zold, tol=tol)
	iter_ += 1
	Zold = numpy.array(Z, copy=True)
    #print iter_
    return Z, convergence, iter_


def minimize_theta(S_, rho=1, obs=1):
    """1st step: Minimize theta step of the ADMM algorithm for solving SIGL
    input:
	- S_ = S_i - rho/obs * Z_i + rho/obs * U_i where S_i is ith entry of S (our list of covariance estimates)
    output:
	- new update of theta_i"""
    D, V = numpy.linalg.eig(S_)
    
    D_ = numpy.identity(len(D)) * [obs/(2. * rho) * (-x + math.sqrt(x*x + 4.*rho/obs)) for x in D]

    return numpy.dot(numpy.dot(V, D_), V.T)


def minimize_Z(A, oldTheta, l1, l2, rho=1.):
    """2nd step: Minimize Z step of the ADMM algorithm for solving 
    input:
	- A is a numpr array such that A = theta + U. This is essentially my y is in my fused lasso-ish problem
	- oldTheta: estimate of \Theta_{t-1}, new estimate will be constrained by this
    outout:
	- new update of Z (numpy array)"""
    
    sudoZ = numpy.array(A, copy=True)
    
    # go through upper triangular part of the matrix:
    for i in range(A.shape[0]):
	for j in range(i, A.shape[0]):
	    y = A[i,j]
	    alpha = oldTheta[i,j] # value which will constain new estimate \Theta_t
	    
	    searchRange = numpy.linspace( min(0, y, alpha), max(0, y, alpha), num=100)
	    scoreEval = 0.5*(y - searchRange)*(y-searchRange) + l1*abs(searchRange) + l2*abs(searchRange - alpha)
	    
	    newB = searchRange[numpy.argmin(scoreEval)] # new entry in \Theta_t[i,j]
	    sudoZ[i,j] = newB
	    sudoZ[j,i] = newB
	    
    return sudoZ	

    
def check_conv1D(theta, Z, Zold, tol):
    """Check convergence of the ADMM algorithm"""
    cond1 = True
    cond2 = True
    if ( ((abs(theta-Z))**2).sum() >= tol):
	cond1 = False
    if ( ((abs(Z - Zold))**2).sum() >= tol):
	cond2 = False
    return cond1 & cond2




