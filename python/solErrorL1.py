import numpy as np
import h5py
import sys

def avgDown(full,nx,ny,nz):
    # number of cells to average into
    xr = full.shape[2]//nx
    yr = full.shape[1]//ny
    zr = full.shape[0]//nz
    # output array
    out = np.zeros((nz,ny,nx))
    for i in np.arange(nx):
        il,iu = i*xr,(i+1)*xr
        for j in np.arange(ny):
            jl,ju = j*yr,(j+1)*yr
            for k in np.arange(nz):
                kl,ku = k*zr,(k+1)*zr
                out[k,j,i] = np.mean(full[kl:ku,jl:ju,il:iu])
    return out

def findError(fSol,fRef):
    # list out ignored keys
    ignoredKeys = ['weno','psi','fomx',
                   'xcoord','ycoord','zcoord',
                   'time','time_step','time_step_size','gamma']
    
    # Get number of cells in reference
    nxr = len(fRef['xcoord']) - 1
    nyr = len(fRef['ycoord']) - 1
    nzr = len(fRef['zcoord']) - 1
    Nr = nxr*nyr*nzr
    # and in solution
    nxs = len(fSol['xcoord']) - 1
    nys = len(fSol['ycoord']) - 1
    nzs = len(fSol['zcoord']) - 1
    Ns = nxs*nys*nzs
    
    # Add up error
    l1err = {}
    for k in fRef.keys():
        if k in ignoredKeys:
            continue
        if nxr == nxs and nyr == nys and nzr == nzs:
            l1err[k] = np.sum(np.abs(fSol[k][()] - fRef[k][()]),axis=None)/Ns
        else:
            ref = avgDown(fRef[k][()],nxs,nys,nzs)
            l1err[k] = np.sum(np.abs(fSol[k][()] - ref),axis=None)/Ns

    return nxs,l1err

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
    
