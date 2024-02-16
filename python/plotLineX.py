import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os

# Return hdf5 dictionary corresponding to a single file
def dataFromFile(fStr):
    print('Opening ', fStr)
    f = h5py.File(fStr,'r')
    data = {}
    for k in f.keys():
        data[k] = f[k][()]
    # Convert from face locations to cell centers in x
    data['x'] = 0.5*(data['xcoord'][:-1] + data['xcoord'][1:])

    # add in extra fields as useful
    data['vel'] = np.sqrt(data['velx']**2 + data['vely']**2 + data['velz']**2)
    if 'pres' in f.keys():
        data['mach'] = data['vel']*np.sqrt(data['vel']/(data['gamma']*data['pres']))
    elif 'prsg' in f.keys():
        data['mach'] = data['vel']*np.sqrt(data['vel']/(data['gamma']*data['prsg']))
    f.close()
    return data

if __name__ == '__main__':
    # parse as many data files as possible
    data = {}
    for fn in range(1,len(sys.argv)):
        fStr = sys.argv[fn]
        if os.path.exists(fStr):
            name = f'File {fn}'
            data[name] = dataFromFile(fStr)
        else:
            break

    # last arg is the field to plot
    fld = sys.argv[-1]

    yidx = 0
    zidx = 0

    # Plot it.
    names = []
    for n,d in data.items():
        names.append(n)
        plt.plot(d['x'],d[fld][zidx,yidx,:])
    plt.legend(names)
    plt.show()
