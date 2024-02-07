import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

if __name__ == '__main__':
    f = h5py.File(sys.argv[1],'r')
    data = f['stats_data'][()]
    fn = np.argmax(data[:,0])
    plt.plot(data[:fn,0],data[:fn,1])
    plt.show()
