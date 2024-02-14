import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

# See Torrilhon in doi:10.1016/S0021-9991(03)00347-4 for exact solutions
# Rarefactions are just blanked out for now

# Return hdf5 dictionary corresponding to a single file
def dataFromFile(fStr):
    print('Opening ', fStr)
    f = h5py.File(fStr,'r')
    data = {}
    for k in f.keys():
        data[k] = f[k][()]
    # Convert from face locations to cell centers in x
    data['x'] = 0.5*(data['xcoord'][:-1] + data['xcoord'][1:])
    f.close()
    return data

def exact0(x):
    rdex = 0.2*np.ones(x.size)
    rbex = -1.0*np.ones(x.size)
    cdex = 0.2*np.ones(x.size)
    cbex = -1.0*np.ones(x.size)
    
    # regular solution
    rdex[x < 1.3207457] = np.nan
    rbex[x < 1.3207457] = np.nan
    
    rdex[x < 1.0413879] = 0.1725845
    rbex[x < 1.0413879] = -0.7274937
    
    rdex[x < 0.5517935] = 0.3645335
    rbex[x < 0.5517935] = -0.4137439
    
    rdex[x < 0.2015924] = 0.7693119
    rbex[x < 0.2015924] = -0.4137439
    
    rdex[x < -0.1516145] = 0.6922069
    rbex[x < -0.1516145] = -0.4966783
    
    rdex[x < -0.239839] = 0.6922069
    rbex[x < -0.239839] = 0.4966783
    
    rdex[x < -0.3628142] = np.nan
    rbex[x < -0.3628142] = np.nan
    
    rdex[x < -0.7082423] = 1.0
    rbex[x < -0.7082423] = 1.0
    
    # compound solution
    cdex[x < 1.3207457] = np.nan
    cbex[x < 1.3207457] = np.nan
    
    cdex[x < 1.0438319] = 0.1728151
    cbex[x < 1.0438319] = -0.7300059

    cdex[x < 0.5537568] = 0.3653074
    cbex[x < 0.5537568] = -0.4140819
    
    cdex[x < 0.2036784] = 0.7538870
    cbex[x < 0.2036784] = -0.4140819
    
    cdex[x < -0.1800155] = np.nan
    cbex[x < -0.1800155] = np.nan
    
    cdex[x < -0.2412073] = 0.6452294
    cbex[x < -0.2412073] = 0.3986243
    
    cdex[x < -0.30228] = np.nan
    cbex[x < -0.30228] = np.nan
    
    cdex[x < -0.7082423] = 1.0
    cbex[x < -0.7082423] = 1.0

    # insert attached shock
    idx = np.argmin(np.fabs(x + 0.2412073))
    cdex[idx] = 0.8410989
    cbex[idx] = -0.2825269

    return rdex,rbex,cdex,cbex

def exact1(x):
    rdex = 0.2*np.ones(x.size)
    rbex = -0.9899924*np.ones(x.size)
    cdex = 0.2*np.ones(x.size)
    cbex = -1.0*np.ones(x.size)
    
    # regular solution
    rdex[x < 1.3207457] = np.nan
    rbex[x < 1.3207457] = np.nan
    
    rdex[x < 1.0453886] = 0.1729621
    rbex[x < 1.0453886] = -0.7242827
    
    rdex[x < 0.7765843] = 0.1729621
    rbex[x < 0.7765843] = -0.7087891
    
    rdex[x < 0.5511855] = 0.3644655
    rbex[x < 0.5511855] = -0.406322
    
    rdex[x < 0.2017165] = 0.7682366
    rbex[x < 0.2017165] = -0.406322
    
    rdex[x < -0.1513512] = 0.6936407
    rbex[x < -0.1513512] = -0.4839091
    
    rdex[x < -0.2405916] = 0.6936407
    rbex[x < -0.2405916] = 0.4994856
    
    rdex[x < -0.3628142] = np.nan
    rbex[x < -0.3628142] = np.nan
    
    rdex[x < -0.7082423] = 1.0
    rbex[x < -0.7082423] = 1.0
    
    # compound solution
    cdex[x < 1.3207457] = np.nan
    cbex[x < 1.3207457] = np.nan
    
    cdex[x < 1.0438319] = 0.1728151
    cbex[x < 1.0438319] = -0.7300059

    cdex[x < 0.5537568] = 0.3653074
    cbex[x < 0.5537568] = -0.4140819
    
    cdex[x < 0.2036784] = 0.7538870
    cbex[x < 0.2036784] = -0.4140819
    
    cdex[x < -0.1800155] = np.nan
    cbex[x < -0.1800155] = np.nan
    
    cdex[x < -0.2412073] = 0.6452294
    cbex[x < -0.2412073] = 0.3986243
    
    cdex[x < -0.30228] = np.nan
    cbex[x < -0.30228] = np.nan
    
    cdex[x < -0.7082423] = 1.0
    cbex[x < -0.7082423] = 1.0

    # insert attached shock
    idx = np.argmin(np.fabs(x + 0.2412073))
    cdex[idx] = 0.8410989
    cbex[idx] = -0.2825269

    return rdex,rbex,cdex,cbex

def exact2(x):
    rdex = 0.2*np.ones(x.size)
    rbex = -0.7373937*np.ones(x.size)
    cdex = 0.2*np.ones(x.size)
    cbex = -0.7373937*np.ones(x.size)
    
    # regular solution
    rdex[x < 1.3779391] = np.nan
    rbex[x < 1.3779391] = np.nan
    
    rdex[x < 1.0669255] = 0.1708649
    rbex[x < 1.0669255] = -0.4947266
    
    rdex[x < 0.8556428] = 0.1708649
    rbex[x < 0.8556428] = -0.6709124
    
    rdex[x < 0.6544735] = 0.3901321
    rbex[x < 0.6544735] = -0.3225398
    
    rdex[x < 0.2763844] = 1.0604288
    rbex[x < 0.2763844] = -0.3225398
    
    rdex[x < -0.0899376] = 1.0224315
    rbex[x < -0.0899376] = -0.3596119
    
    rdex[x < -0.1451486] = 1.0224315
    rbex[x < -0.1451486] = 0.3596119
    
    rdex[x < -0.2224215] = np.nan
    rbex[x < -0.2224215] = np.nan
    
    rdex[x < -0.6312291] = 1.7
    rbex[x < -0.6312291] = 1.0
    
    # compound solution
    cdex[x < 1.3779391] = np.nan
    cbex[x < 1.3779391] = np.nan
    
    cdex[x < 1.0676406] = 0.1709292
    cbex[x < 1.0676406] = -0.4953384

    cdex[x < 0.8559222] = 0.1709292
    cbex[x < 0.8559222] = -0.6717421
    
    cdex[x < 0.6549893] = 0.3903221
    cbex[x < 0.6549893] = -0.3226993
    
    cdex[x < 0.2769261] = 1.0549708
    cbex[x < 0.2769261] = -0.3226993
    
    cdex[x < -0.0998446] = np.nan
    cbex[x < -0.0998446] = np.nan
    
    cdex[x < -0.1465035] = 0.9796115
    cbex[x < -0.1465035] = 0.3026538
    
    cdex[x < -0.1912812] = np.nan
    cbex[x < -0.1912812] = np.nan
    
    cdex[x < -0.6312291] = 1.7
    cbex[x < -0.6312291] = 1.0

    # insert attached shock
    idx = np.argmin(np.fabs(x + 0.1465035))
    cdex[idx] = 1.1485298
    cbex[idx] = -0.2013996
    

    return rdex,rbex,cdex,cbex

if __name__ == '__main__':
    # First argument is the data file
    fStr = sys.argv[1]
    data = dataFromFile(fStr)

    # second arg is the problem type
    pt = int(sys.argv[2])

    yidx = 0
    zidx = 0

    # pull out x coords and solution
    x = data['x']
    dens = data['dens'][zidx,yidx,:]
    magy = data['magy'][zidx,yidx,:]

    if pt == 0:
        rdex,rbex,cdex,cbex = exact0(x)
    elif pt == 1:
        rdex,rbex,cdex,cbex = exact1(x)
    elif pt == 2:
        rdex,rbex,cdex,cbex = exact2(x)
    elif pt == 3:
        rdex,rbex = exact3(x)

    # Plot it.
    plt.subplot(1,2,1)
    plt.plot(x,dens,'-g',x,rdex,'-k',x,cdex,'--k')
    plt.title('Density')
    
    plt.subplot(1,2,2)
    plt.plot(x,magy,'-g',x,rbex,'-k',x,cbex,'--k')
    plt.title('By')
    
    plt.show()
