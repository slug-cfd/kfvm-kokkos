import numpy as np
import h5py
import sys
import glob

def findError(fSol,fRef):
    # list out ignored keys
    ignoredKeys = ['weno','psi',
                   'xcoord','ycoord','zcoord',
                   'time','time_step','gamma']
    
    # Get number of cells
    nx = len(fRef['xcoord']) - 1
    ny = len(fRef['ycoord']) - 1
    nz = len(fRef['zcoord']) - 1
    N = nx*ny*nz
    
    # Add up error
    l1err = {}
    for k in fRef.keys():
        if k in ignoredKeys:
            continue
        l1err[k] = np.sum(np.abs(fSol[k][()] - fRef[k][()]),axis=None)/N

    return nx,l1err

def printErrorsHydro(N,l1err):
    # get momentum error as norm
    momErr = np.sqrt(l1err['momx']**2 + l1err['momy']**2 + l1err['momz']**2)
    print(f"{N} {l1err['dens']} {momErr} {l1err['etot']}")

if __name__ == '__main__':
    baseDir = sys.argv[1]
    rad = sys.argv[2]

    dataDirs = [baseDir + '/R' + rad + '_NX32_NY32/ckpt/',
                baseDir + '/R' + rad + '_NX64_NY64/ckpt/',
                baseDir + '/R' + rad + '_NX128_NY128/ckpt/',
                baseDir + '/R' + rad + '_NX256_NY256/ckpt/',
                baseDir + '/R' + rad + '_NX512_NY512/ckpt/']

    for ddir in dataDirs:
        sols = glob.glob(ddir + '*.h5')
        
        fSol = h5py.File(sols[0],'r')
        fRef = h5py.File(sols[1],'r')
        
        N,l1err = findError(fSol,fRef)

        printErrorsHydro(N,l1err)
