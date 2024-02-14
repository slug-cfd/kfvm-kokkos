import numpy as np
import matplotlib.pyplot as plt
import h5py

def dataFromFile(fStr):
    f = h5py.File(fStr,'r')
    data = {}
    for k in f.keys():
        data[k] = f[k][()]
    f.close()
    return data

# Get nth frequency and its amplitude
def modeN(vtau,n):
    N = vtau.shape[0]
    phiN = 2.0*np.pi*n/N
    j = np.arange(0,N,dtype=np.float64)
    ej = np.exp(-1j * j * phiN)
    return phiN,np.sum(vtau*ej)/N

# Calling convention:
if __name__ == '__main__':
    # Max number of modes
    maxN = 128

    # Data keyed by future legend entries
    # value is list of [<data dir>,<phis>,<dispersion>,<dissipation>,<linestyle>]

    # 2d runs
    data = {r'$R = 2, \ell = 5$, No Weno':['data/NoWeno/ell5/R2_NX257_NY8/ckpt',[],[],[],'-r'],
            r'$R = 2, \ell = 5$, Weno':['data/Weno/ell5/R2_NX257_NY8/ckpt',[],[],[],'--r'],
            r'$R = 3, \ell = 5$, No Weno':['data/NoWeno/ell5/R3_NX257_NY8/ckpt',[],[],[],'-b'],
            r'$R = 3, \ell = 5$, Weno':['data/Weno/ell5/R3_NX257_NY8/ckpt',[],[],[],'--b']}

    # 3d runs
    # data = {r'$R = 2, \ell = 5$, No Weno':['data/NoWeno/ell5/R2_NX257_NY4_NZ4/ckpt',[],[],[],'-r'],
    #         r'$R = 2, \ell = 5$, Weno':['data/Weno/ell5/R2_NX257_NY4_NZ4/ckpt',[],[],[],'--r'],
    #         r'$R = 3, \ell = 5$, No Weno':['data/NoWeno/ell5/R3_NX257_NY4_NZ4/ckpt',[],[],[],'-b'],
    #         r'$R = 3, \ell = 5$, Weno':['data/Weno/ell5/R3_NX257_NY4_NZ4/ckpt',[],[],[],'--b']}
    
    phis = []

    for k,v in data.items():
        dataDir = v[0]
        # Loop over wavenumbers
        for n in range(1,maxN + 1):
            d0 = dataFromFile(dataDir + f'/DispRel{n:03d}_0000000.h5')
            d1 = dataFromFile(dataDir + f'/DispRel{n:03d}_0000001.h5')
            
            vt0 = d0['dens'][0,0,:]
            vt1 = d1['dens'][0,0,:]
            sig = d1['time'] / (d1['xcoord'][1] - d1['xcoord'][0])
            
            phi,vh0 = modeN(vt0,n)
            phi,vh1 = modeN(vt1,n)
            
            Phi = -np.log(vh1/vh0)/(1j * sig)
            
            v[1].append(phi)
            v[2].append(np.real(Phi))
            v[3].append(np.imag(Phi))

        # Convert to numpy arrays
        v[1] = np.array(v[1])
        v[2] = np.array(v[2])
        v[3] = np.array(v[3])
        phis = v[1]
    
    # Plot resulting curves
    fig,axs = plt.subplots(1,2)
    
    axs[0].plot(phis,phis,'--k',label='Reference')
    axs[1].plot(phis,np.zeros_like(phis),'--k',label='Reference')

    for k,v in data.items():
        axs[0].plot(v[1],v[2],v[4],label=k)
        axs[1].plot(v[1],v[3],v[4],label=k)

    for i in range(2):
        axs[i].grid('both')
        axs[i].set_xlabel(r'$\phi$')
        
    axs[0].set_ylabel(r'$Re\left(\Phi(\phi)\right)$')
    axs[1].set_ylabel(r'$Im\left(\Phi(\phi)\right)$')
    axs[1].legend(fontsize=18)
    
    plt.show()
    
