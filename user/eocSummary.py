import numpy as np
import sys

if __name__ == '__main__':
    data = np.loadtxt(sys.argv[1])

    # Count number of grid res and error cols
    nRes,nErr = data.shape;
    nErr = nErr - 1

    # Log spacings in resolution
    ldx = np.zeros((nRes - 1,1))
    ldx[:,0] = np.log10(data[1:,0]) - np.log10(data[:-1,0])
    ldx = np.repeat(ldx,nErr,axis=1)

    # Log differences in errors
    lde = np.log10(data[:-1,1:]) - np.log10(data[1:,1:])

    # experimental order of convergence
    eoc = np.zeros((nRes - 1,nErr + 1))
    eoc[:,0] = data[1:,0]
    eoc[:,1:] = lde/ldx
    print(eoc)
