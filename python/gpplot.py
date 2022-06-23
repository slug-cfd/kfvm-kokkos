# Author: Ian May
# File: gpplot.py
# Purpose: Make plots and movies from solution files generated
#          by my GP-WENO code

import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
import sys

# Return hdf5 dictionary corresponding to a single file
def dataFromFile(fStr):
    print('Opening ', fStr)
    f = h5py.File(fStr,'r')
    data = {}
    for k in f.keys():
        data[k] = f[k][()]
    data['ext'] = tuple(data['ext'])
    if 'time' in data:
        data['time'] = data['time'][0]
    data['filename'] = f.filename
    f.close()
    return data

# Return a list of hdf5 dictionaries for all file matching a glob
def dataFromGlob(gStr):
    dset = []
    for fStr in glob.iglob(gStr):
        dset.append(dataFromFile(fStr))
    return dset

def summarizeField(d,fld):
    print('Field: %s' % fld)
    print('        max/min: %e || %e' % (np.max(d[fld]),np.min(d[fld])))
    print('   Abs. max/min: %e || %e' % (np.max(np.abs(d[fld])),np.min(np.abs(d[fld]))))
    print(' mean/abs. mean: %e || %e' % (np.mean(d[fld]),np.mean(np.abs(d[fld]))))

# Plot a single field from one dataset
# Optionally supply fname to override showing the plot window
#   saving directly to a file
def plotField(d,fld,nCont=0,cmap='jet',fname='',vr=[]):
    plt.figure(figsize=(9.6,5.4),dpi=200)
    if len(vr)==2:
        plt.imshow(d[fld].T,origin='lower',extent=d['ext'],interpolation='gaussian',cmap=cmap,vmin=vr[0],vmax=vr[1])
    else:
        plt.imshow(d[fld].T,origin='lower',extent=d['ext'],interpolation='gaussian',cmap=cmap)
    plt.colorbar()
    if nCont != 0:
        if len(vr)==2:
            plt.contour(d['x'],d['y'],d[fld].T,nCont,colors='black',linewidths=0.5,vmin=vr[0],vmax=vr[1])
        else:
            plt.contour(d['x'],d['y'],d[fld].T,nCont,colors='black',linewidths=0.5)
    if 'time' in d:
        tstr = ('Time: %1.3e' % d['time'])
        plt.title(tstr,loc='left')
    plt.tight_layout()
    if fname != '':
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

# Save many plots for a given field split into frames through time
def saveFrames(gStr,fld):
    fldMin = sys.float_info.max
    fldMax = -sys.float_info.max
    for fStr in glob.iglob(gStr):
        d = dataFromFile(fStr)
        fldMin = min(np.min(d[fld]),fldMin)
        fldMax = max(np.max(d[fld]),fldMax)
    for fStr in glob.iglob(gStr):
        d = dataFromFile(fStr)
        ln = d['filename'].split('.')
        del ln[-1]
        fname = '.'.join(ln)+'_'+fld+'.png'
        plotField(d,fld,fname=fname,vr=[fldMin,fldMax])

# Calling convention: python3 gpplot.py <path/to/data.hdf5> <field name>
if __name__ == "__main__":
    # Open dataset
    print(sys.argv)
    d = dataFromFile(sys.argv[1])
    summarizeField(d,sys.argv[2])
    if len(sys.argv)>3:
        plotField(d,sys.argv[2],nCont=int(sys.argv[3]))
    else:
        plotField(d,sys.argv[2])
