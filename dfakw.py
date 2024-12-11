#!/usr/bin/env python
import numpy as np
import os, sys
from numpy.polynomial.legendre import leggauss
from scipy import integrate

broad = 2e-2
if len(sys.argv)>1:
    broad = float(sys.argv[1])
    print('broad=', broad)
    

# load data
EF = np.loadtxt('EF.dat')
dat=[[],[]]
for i,fname in enumerate(['eigvals.dat', 'eigvalsdn.dat']):
    if os.path.isfile(fname):
        dat[i] = np.loadtxt(fname)
    elif os.path.isfile(fname+'.gz'):
        dat[i] = np.loadtxt(fname+'.gz')
    else:
        print('ERROR: couldnt find file', fname)

om = dat[0][:,0]        # first column is frequency
eks_up = dat[0][:,1::2]+dat[0][:,2::2]*1j   # eks_up[:nom,:nbands]
eks_dn = dat[1][:,1::2]+dat[1][:,2::2]*1j   # eks_dn[:nom,:nbands]

# find how many frequencies we have, and how many k-points
for niw in range(len(om)): # frequency is sorted, hence when it jumps from positve to negative, next k-point starts
    if om[niw+1]-om[niw]<0:
        break
niw+=1                 # now we know the number of frequencies

nkp = int(len(om)/niw) # than this is k-points
nbands = np.shape(eks_up)[1]
print('niw=', niw, 'nkp=', nkp, 'nbands=', nbands)

om = np.reshape(om, (nkp,niw))
eks_up = np.reshape(eks_up, (nkp,niw,nbands))
eks_dn = np.reshape(eks_dn, (nkp,niw,nbands))

diff1 = np.sum(abs(eks_up-eks_dn))/np.size(eks_up)
print('<|ek_up-ek_dn|>=', diff1)

eks_up[abs(eks_up.imag)<broad] -= broad*1j
eks_dn[abs(eks_dn.imag)<broad] -= broad*1j

Akw_up, Akw_dn = np.zeros((nkp,niw)), np.zeros((nkp,niw))
for ibnd in range(nbands):
    Akw_up += np.imag(1/(om+EF-eks_up[:,:,ibnd]))*(-1/np.pi)
    Akw_dn += np.imag(1/(om+EF-eks_dn[:,:,ibnd]))*(-1/np.pi)


# spectral function can be extremely sharply peaked, but we want to remove extremely large peaks from the plot.
# at the fermi surface Akw can be diverging, but we don't want to plot diverging Akw
intensity = 0.99
ht, bin_edges = np.histogram(np.vstack((Akw_up.ravel(),Akw_dn.ravel())),bins=5000)
xh = 0.5*(bin_edges[1:]+bin_edges[:-1])
cums = np.cumsum(ht)/sum(ht)
i = np.searchsorted(cums, intensity)
#print('with intensity=', intensity, 'we determine cutoff at Ak=', xh[i])
vmm = [0,xh[i]]
print('A_min=', vmm[0], 'A_max=', vmm[1], 'A_min_found=', np.min(Akw_up), 'A_max_found=', np.max(Akw_up) )
print('om_min=', om[0,0], 'om_max=', om[0,-1])
# Here we remove such divergencies, and just set them to large but finite value. It should not change results much, just improve numerics.
Akw_up[ (Akw_up > vmm[1]) ] = vmm[1]
Akw_dn[ (Akw_dn > vmm[1]) ] = vmm[1]

w = om[0,:]
dAkw_om = (Akw_up-Akw_dn)*om
dfs1 = abs(np.array([integrate.trapz(dAkw_om[ik,:], x=w) for ik in range(nkp)]))
dfs2 = abs(np.array([integrate.simpson(dAkw_om[ik,:], x=w) for ik in range(nkp)]))
# In k-point path where AM has no splitting should be removed, because otherwise result will strongly depend on the
# ratio between path with splitting and withouth splitting.
# We remove path where |diff|<1e-3 ( Checked by plot )

ht, bin_edges = np.histogram(dfs1,bins=5000)
xh = 0.5*(bin_edges[1:]+bin_edges[:-1])
unsplit_value = 0.05*xh[-1]
print('unsplit_vaue[eV]=', unsplit_value)
unsplit_kpoints = dfs1>unsplit_value
df1 = dfs1[unsplit_kpoints]
diff1 = sum(df1)/len(df1)
df2 = dfs2[unsplit_kpoints]
diff2 = sum(df2)/len(df2)


print('int dw{ |A(k,w,up)-Ak(k,w,dn)|*w } = ', abs(diff2))


#from pylab import *
#for ik in range(nkp):
#    df1 = integrate.simpson((Akw_up[ik,:]-Akw_dn[ik,:])*w, x=w)
#    df2 = integrate.trapz((Akw_up[ik,:]-Akw_dn[ik,:])*w, x=w)
#    print(ik, 'df1=', df1, 'df2=', df2, 'I_up=', integrate.simpson(Akw_up[ik,:], x=w), 'I_dn=', integrate.simpson(Akw_dn[ik,:], x=w))
#    plot(w, w*(Akw_up[ik,:]-Akw_dn[ik,:]), label='Aw_up')
#    plot(w, Akw_up[ik,:]-Akw_dn[ik,:], label='Aw_up')
#    #plot(w, Akw_dn[ik,:], label='Aw_dn')
#    show()

#subplot(2,1,1)
#plot(dfs)
#print('ht=', ht)
#print('be=', bin_edges)
#subplot(2,1,2)
#plot(xh, ht)
#show()

## We can sum over all k-points, and have a single function to integrate.
#dAkw = np.sum((Akw_up-Akw_dn),axis=0)/nkp
## frequency for any k-point
#w = om[0,:]
#diff2 = integrate.trapezoid(dAkw*w, x=w)
#diff3 = integrate.simpson(dAkw*w, x=w)
#print('int dw{ |A(k,w,up)-Ak(k,w,dn)|*w }[meV] = ', abs(diff2)*1e3, abs(diff3)*1e3)
#
#from pylab import *
#plot(w, dAkw*w)
#show()
##diff4 = np.sum( (Akw_up-Akw_dn)*(om-EF) )*(w[-1]-w[0])/(nkp*niw)
##print('diff4=', diff4)
#

if True:
    # if we want to plot it....
    from pylab import *
    import matplotlib as mpl
    # make custom the colormaps
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap1',['white','blue','navy'],N=256,gamma=1.0)
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['white','red','crimson'],N=256,gamma=1.0)
    # create the _lut array, with rgba values
    cmap2._init() 
    # create alpha array and fill the colormap with them.
    # here it is progressive, but you can create whathever you want
    alphas = np.linspace(0, 0.9, cmap2.N+3)
    cmap2._lut[:,-1] = alphas
    #subplot(3,1,1)
    imshow(Akw_up.T, interpolation='bilinear', cmap=cmap1, origin='lower', vmin=vmm[0], vmax=vmm[1])# extent=[1,nkp,om[0,0],om[0,-1]])#,aspect=aspect)
    #subplot(3,1,2)
    imshow(Akw_dn.T, interpolation='bilinear', cmap=cmap2, origin='lower', vmin=vmm[0], vmax=vmm[1])# extent=[xmin,xmax,ymin,ymax],aspect=aspect)
    #subplot(3,1,3)
    #imshow(Akw_up.T-Akw_dn.T, interpolation='bilinear', cmap=cmap1, origin='lower', vmin=vmm[0], vmax=0.01*vmm[1])# extent=[xmin,xmax,ymin,ymax],aspect=aspect)
    show()
