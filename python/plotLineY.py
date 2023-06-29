import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

# Return hdf5 dictionary corresponding to a single file
def dataFromFile(fStr):
    print('Opening ', fStr)
    f = h5py.File(fStr,'r')
    data = {}
    for k in f.keys():
        data[k] = f[k][()]
    # Convert from face locations to cell centers in y
    data['y'] = 0.5*(data['ycoord'][:-1] + data['ycoord'][1:])
    f.close()
    return data

if __name__ == '__main__':
    # First argument is the data file
    fStr = sys.argv[1]
    data = dataFromFile(fStr)

    # second arg is the field to plot
    fld = sys.argv[2]

    xidx = 0
    zidx = 0

    # third argument is optionally the y index to cut through
    if len(sys.argv) > 3:
        xidx = int(sys.argv[3])

    # fourth argument is optionally the z index to cut through
    if len(sys.argv) > 4:
        zidx = int(sys.argv[4])

    # Plot it.
    plt.plot(data['y'],data[fld][zidx,:,xidx])
    plt.show()
