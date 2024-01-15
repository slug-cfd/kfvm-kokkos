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
    pdiff = 0.0
    fld = 'dens'
    if nxr == nxs and nyr == nys and nzr == nzs:
        pdiff = 100.0*np.max(np.abs(fSol[fld][()] - fRef[fld][()])/fRef[fld][()],axis=None)
    else:
        ref = avgDown(fRef[fld][()],nxs,nys,nzs)
        pdiff = 100.0*np.max(np.abs(fSol[fld][()] - ref)/ref,axis=None)

    return nxs,pdiff

if __name__ == '__main__':
    # Display help message and exit
    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print('Usage: python pctDiff.py <sol.h5> <ref.h5>')
        print('<sol.h5>: Solution file to get error of')
        print('<ref.h5>: Reference file with exact solution')
        exit(0)
    
    # open data files
    fSol = h5py.File(sys.argv[1],'r')
    fRef = h5py.File(sys.argv[2],'r')

    # get all l1 errors
    N,pdiff = findError(fSol,fRef)

    print(f'Sol. res. {N}, pdiff: {pdiff}%')

    # close files and exit
    fRef.close()
    fSol.close()
    
