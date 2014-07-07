## Burn in for the Online version of the SINGLE algorithm ##
#
# This is essentially a watered down implementation of the SINGLE algorithm 
#
# Fused Lasso step is implemented using an ADMM algorithm
#
#
#
#

import numpy
import math

def BurnInSINGLE(Sarray, l1, l2, tol=.01, max_iter=50):
    """Function to estimate covariance matrices during burn in
    
    INPUT:
	 - Sarray: input array of estimated covariance matrices
	 - l1, l2: sparsity & smoothness parameters
	 - tol: convergence criterion
	 - max_iter: maximum number of iterations
	 
	 
	 
    
    """
    
    theta = numpy.dstack(( [numpy.identity(Sarray.shape[1]) for i in range(Sarray.shape[0])] )).transpose() # initialise to array of identity matrices
    
    #theta = numpy.zeros((Sarray.shape[0], Sarray.shape[1], Sarray.shape[1]))
    Z = numpy.zeros((Sarray.shape[0], Sarray.shape[1], Sarray.shape[1]))
    U = numpy.copy(Z)
    Zold = numpy.copy(Z) # used to confirm convergence
    convergence = False
    iter = 0
    
    while (convergence==False) & (iter < max_iter):
	
	# theta step:
	for i in range(theta.shape[0]):
	    theta[i,:,:] = minimize_theta(theta[i,:,:] - Z[i,:,:] + U[i,:,:])
	    
	# Z step:
	Z = minimize_Z_fused(A = theta+U, l1=l1, l2=l2)
	
	# U step:
	U = U + theta - Z
	
	# check convergence:
	if ( ((theta-Z)**2).sum() < tol ) & ( ((Z-Zold)**2).sum()<tol ):
	    convergence = True
	else:
	    iter += 1
	    print iter
	    Zold = numpy.copy(Z)

    # return Z as a list as thats what will be used later:
    Z_ = [Z[i,:,:] for i in range(Z.shape[0])]
    return Z_#, iter


def minimize_theta(S_, rho=1, obs=1):
    """1st step: Minimize theta step of the ADMM algorithm for solving SIGL
    input:
	- S_ = S_i - rho/obs * Z_i + rho/obs * U_i where S_i is ith entry of S (our list of covariance estimates)
    output:
	- new update of theta_i"""
    D, V = numpy.linalg.eig(S_)
    
    D_ = numpy.identity(len(D)) * [obs/(2. * rho) * (-x + math.sqrt(x*x + 4.*rho/obs)) for x in D]

    return numpy.dot(numpy.dot(V, D_), V.T)

    
def minimize_Z_fused(A, l1, l2, rho=1):
    """2nd step: Minimize Z step of the ADMM algorithm for solving SIGL
    input:
	- A is a list such that A[i] = theta[i] + U[i]
    outout:
	- new update of Z (ie a list)"""
    

    # convert A into an array:
    sudoZ = numpy.copy(A)
    
    for i in range(A.shape[1]):
	for j in range(i, A.shape[1]):
	    resp = A[:,i,j]
	    
	    beta_hat = ADMMFused(resp, l1=l1, l2=l2)
	    
	    sudoZ[:,i,j] = beta_hat
	    sudoZ[:,j,i] = beta_hat

    return sudoZ
    

def ADMMFused(resp, l1, l2, tol=.001, max_iter = 500):
    """ADMM implementation of the Fused Lasso
    
    INPUT:
	 - resp: response vector
	 - l1, l2: sparsity and fusion parameters
	 - tol: convergence criterion
	 - max_iter: maximum number of iterations
    
    """
    
    # define D matrix:
    n = len(resp)
    D = numpy.hstack(( numpy.identity(n-1), numpy.zeros((n-1,1)) )) - numpy.hstack(( numpy.zeros((n-1,1)), numpy.identity(n-1)  ))
    D = numpy.vstack((numpy.identity(n), float(l2)/l1 * D))
    
    # store (I + D^T D)^-1
    invD = numpy.linalg.inv( numpy.dot(numpy.transpose(D), D) + numpy.identity(n))
    
    # initial values for beta, z and u:
    p = D.shape[0]
    beta = numpy.ones(n)
    z = numpy.zeros(p)
    z_old = numpy.zeros(p) # to check for convergence
    u = numpy.zeros(p)
    
    # set up convergence criteria
    conv = False
    iter = 0
    
    while (conv==False) & (iter < max_iter):
	
	# beta step:
	beta = numpy.dot(invD, resp + numpy.dot(D.transpose(), z-u))
	
	# z step:
	z = VecST(numpy.dot(D, beta) + u, l1)
	
	# u step:
	u += numpy.dot(D, beta) - z
	
	# check convergence
	if (  ((numpy.dot(D, beta) - z)**2).sum() < tol ) & ( ((z-z_old)**2).sum() < tol):
	    conv = True
	else:
	    z_old = numpy.copy(z)
	    iter += 1
	    #print iter
	    
    # do some cleaning on beta (this wont be sparse because sparsity is imposed on z thus indirectly on beta)
    beta = VecClean(beta, tol)
    return beta
	

def SoftThres(x, lam):
    """Softthresholding function"""
    return int(abs(x)>lam)*(abs(x)-lam)*math.copysign(1,x)
    

def cleanFunc(x, tol):
    """Function to clean beta"""
    return int(abs(x)>tol)*x
    
    
VecST = numpy.vectorize(SoftThres)
VecClean = numpy.vectorize(cleanFunc)


## small sim
#from pylab import *
#resp = numpy.concatenate((numpy.random.randn(50), numpy.random.randn(50)+2, numpy.random.randn(50)))
#x = ADMMFused(resp, 1, 5, tol=.001, max_iter=500)

#plot(resp)
#plot(x)
#show()
    
    
    