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
	    print iter
	    
    # do some cleaning on beta (this wont be sparse because sparsity is imposed on z thus indirectly on beta)
    
    beta = VecClean(beta, tol)
    return beta
	

def SoftThres(x, lam):
    """Softthresholding function"""
    return int(abs(x)>lam)*(abs(x)-lam)*copysign(1,x)
    

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
    
    
    