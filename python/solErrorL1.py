import numpy as np
import h5py
import sys

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

def printErrorsMHD(N,l1err):
    # get momentum and mag field errors as norms
    momErr = np.sqrt(l1err['momx']**2 + l1err['momy']**2 + l1err['momz']**2)
    magErr = np.sqrt(l1err['magx']**2 + l1err['magy']**2 + l1err['magz']**2)
    print(f"{N} {l1err['dens']} {momErr} {l1err['etot']} {magErr}")

if __name__ == '__main__':
    # Display help message and exit
    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print('Usage: python solError.py <sol.h5> <ref.h5>')
        print('<sol.h5>: Solution file to get error of')
        print('<ref.h5>: Reference file with exact solution')
        exit(0)
    
    # open data files
    fSol = h5py.File(sys.argv[1],'r')
    fRef = h5py.File(sys.argv[2],'r')

    # get all l1 errors
    N,l1err = findError(fSol,fRef)

    # print out errors for appropriate system
    if 'magx' in fRef.keys():
        printErrorsMHD(N,l1err)
    else:
        # Also handles srhydro
        printErrorsHydro(N,l1err)

    # close files and exit
    fRef.close()
    fSol.close()
    
