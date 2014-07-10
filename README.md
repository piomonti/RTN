Real Time Network Estimation
==================
This is a "real-time" implementation of SINGLE algorithm.
Allows for the estimation of networks on-the-fly
##### Example:
We provide simulated data in the `Sample Data` folder. 
```
import numpy, pandas
from ThetaEstimation import onlineSINGLE

# read in data:
data = numpy.array(pandas.read_csv('Sample Data/biggerdata.csv'))
# This dataset contains 300 observations over 10 nodes.

# initialise - note that here the first 15 observations are provided as a burn-in. 
# Alternatively only 1 observation can be provided (i.e. no burn-in)

Theta = onlineSINGLE(data=data[0:15,:], l1=1, l2=.5, ff=.9)

# For each of the remaining 285 observations we can update our 
# estimate of the precision as follows:
for i in range(15, 300):
    Theta.updateTheta(newX = data[i,:])
    
# results for precision matrices are stored in Theta.Z (a list)    

```

