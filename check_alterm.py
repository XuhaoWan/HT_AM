#!/usr/bin/env python
import shutil
import os, sys
import subprocess
import re
import glob
import numpy as np
from numpy import *
from cif2struct import Cif2Struct, W2k_klist_band, CifParser_W2k

def CheckAltermagnet(cif, log):
    pymg_convent = cif.structure
    #symmetry_operations = cif.parser.symmetry_operations
    symmetry_operations = cif.original_symmetry_operations
    pgroups = cif.pgroups
    
    print(pymg_convent.composition.reduced_formula, file=log)
    matrix = pymg_convent.lattice.matrix

    symmetry_operations_type=[[] for i in range(len(symmetry_operations))]
    
    magnetic=False
    if hasattr(symmetry_operations[0], 'time_reversal'):
        magnetic=True
    else:
        return 'nonMagnetic'
    
    total_M = array([0.,0.,0.])
    for ii,site in enumerate(pymg_convent.sites):
        if 'magmom' in site.properties:
            print(float(site.properties["magmom"]), site.properties["magmom"].moment, file=log)
            total_M += site.properties["magmom"].moment
    total_Ms = linalg.norm(total_M)
    print('total Moment={:9.4f} == [{:6.3f},{:6.3f},{:6.3f}]'.format(total_Ms,*total_M), file=log)
    if (abs(total_Ms)>1e-6):
        return 'tot_M!=0'

        
    #print('magnetic=', magnetic)
    #pymg_convent.to(ciffile, fmt='cif', symprec=0.1)
    sorder={'I':0,'T':1,'P':2,'R':3}
    for isym,op in enumerate(symmetry_operations):
        timat = array(np.round(op.affine_matrix[:3,:3]),dtype=int)
        tau = op.affine_matrix[:3,3]
        #if magnetic: op.time_reversal
        if np.allclose(timat,np.identity(3,dtype=int)):
            if sum(abs(tau))==0:
                symmetry_operations_type[isym]='I'
            else:
                symmetry_operations_type[isym]='T'
        elif np.allclose(timat,-np.identity(3,dtype=int)) and sum(abs(tau))==0:
            symmetry_operations_type[isym]='P'
        else:
            symmetry_operations_type[isym]='R'
        #print(isym,':', symmetry_operations_type[isym], file=log)
        #for i in range(3):
        #    print('{:2d}{:2d}{:2d} {:10.8f}'.format(*timat[i,:], tau[i]), file=log)

    # First resort symmetry operations such that translations are first, followed by inversion, and last are rotations
    indx = sorted(range(len(symmetry_operations)), key=lambda i: sorder[symmetry_operations_type[i]])
    print('indx=', indx, file=log)
    _symmetry_operations_ = [symmetry_operations[i] for i in indx]
    _symmetry_operations_type_ = [symmetry_operations_type[i] for i in indx]
    symmetry_operations = _symmetry_operations_
    symmetry_operations_type = _symmetry_operations_type_
    # Print out sorted symmetry operations
    for isym,op in enumerate(symmetry_operations):
        timat = array(np.round(op.affine_matrix[:3,:3]),dtype=int)
        tau = op.affine_matrix[:3,3]
        if magnetic:
            print(isym, ':', 'Time='+str(op.time_reversal), symmetry_operations_type[isym], file=log)
        else:
            print(isym,':', symmetry_operations_type[isym], file=log)
            
        for i in range(3):
            print('{:2d}{:2d}{:2d} {:10.8f}'.format(*timat[i,:], tau[i]), file=log)


    for isym,op in enumerate(symmetry_operations):
        #print('CHECK:op.time_reversal=', op.time_reversal, 'symmetry_operations_type=', symmetry_operations_type[isym], file=log) 
        if op.time_reversal==-1 and symmetry_operations_type[isym]=='P':
            print('PT present!', file=log)
            return 'PT_present'
        if op.time_reversal==-1 and symmetry_operations_type[isym]=='T':
            print('Tt present!', file=log)
            return 'tT_present'

    print('pgroups=', pgroups, file=log)
    acoords = [site.frac_coords%1.0 for site in pymg_convent.sites] # all coordinates but always inside home unit cell
    anames = [site.species_string for site in pymg_convent.sites]
    amagmoms = [0 for site in pymg_convent.sites]
    im=-1
    for i,site in enumerate(pymg_convent.sites):
        if 'magmom' in site.properties:
            amagmoms[i] = site.properties["magmom"].moment
            if im<0 and linalg.norm(amagmoms[i])>1e-6: im=i
    iopposite = [i for i in range(len(amagmoms)) if linalg.norm(amagmoms[i]+amagmoms[im])<1e-6]

    which_group={}
    for i in range(len(pgroups)):
        for j in range(len(pgroups[i])):
            for k in range(len(pgroups[i][j])):
                g = pgroups[i][j][k]
                #print('g=', g, 'i,j=', i,j, file=log)
                which_group[g] = (i,j)
                
    #print('which_group=', which_group, file=log)
    #print('acoords[im]=', acoords[im], file=log)
    for ip in iopposite:
        inv_center = ((acoords[im]+acoords[ip])%1 ) /2.
        print('Trying inversion center=', inv_center, file=log)
        PT = True
        for j in range(len(acoords)):
            cj = acoords[j]-inv_center
            mcj = (-cj + inv_center) % 1.0
            Mj = linalg.norm(amagmoms[j])
            #
            diff = sum(abs(acoords - mcj),axis=1)
            ik = argmin(diff)
            #print('    ik=', ik, 'diff=', diff, file=log)
            Found=False
            if diff[ik]<1e-5:
                equivalent = which_group[j]==which_group[ik]
                mopposite = sum(abs(amagmoms[j]+amagmoms[ik]))<1e-5
                # For magnetic systems the atoms with opposite spin are already condidered non-equivalent
                # It is sufficient to be part of the same atom type and have exactly opposite magnetic moment
                if mopposite and Mj>1e-5 and anames[ik]==anames[j] and which_group[j][0]==which_group[ik][0]:
                    equivalent=True
                #print('        eq=',equivalent, 'mopp=', mopposite, file=log)
                if equivalent and mopposite:
                    Found=True
                    #print('have pair', (j,ik), equivalent, file=log)
            if Found:
                print('  PT['+str(j)+']='+str(ik), 'c0=',acoords[j], 'cj=', cj, 'mcj=', mcj, 'Found=', Found, file=log)
            else:
                print('  PT['+str(j)+']=none', 'c0=',acoords[j], 'cj=', cj, 'mcj=', mcj, 'Found=', Found, file=log)
            if not Found:
                PT = False
                break
        if PT:
            return 'PT present at '+str(inv_center)
    
    return 'YES'

    print('Conventional=', file=log)
    print(pymg_convent, file=log)

    acoords = [site.frac_coords%1.0 for site in pymg_convent.sites] # all coordinates but always inside home unit cell
    anames = [site.species_string for site in pymg_convent.sites]
    amagmoms = [0 for site in pymg_convent.sites]
    if magnetic:
        for i,site in enumerate(pymg_convent.sites):
            if 'magmom' in site.properties:
                amagmoms[i] = site.properties["magmom"].moment
    
    print('amagmoms=', amagmoms, file=log)
    groups=[]
    grnames=[]
    for name,indices in corr_names.items():
        #print('name=', name)
        grp = [[indices[0]]]  # we have just one group with the first entry in coords. All entries in coords will have index in grp
        #wop = [[0]]
        site = pymg_convent.sites[indices[0]]
        Mi = 0
        if magnetic:
            Mi = site.properties["magmom"].moment
            print(name, site.frac_coords, 'M=', site.properties["magmom"].moment, file=log)
        for ii in indices[1:]: # loop over possibly equivalent sites (of the same element)
            site = pymg_convent.sites[ii]
            #coord = acoords[ii]    # fractional coordinate of that site self.structure.sites[ii]
            coord = site.frac_coords
            Mj = 0
            if magnetic:
                Mj = site.properties["magmom"].moment
                print(name, coord, 'M=', site.properties["magmom"].moment, file=log)
            #print('checking', coord)
            Qequivalent=False  # for now we thing this might not be equivalent to any other site in grp
            for ig,op in enumerate(symmetry_operations): # loop over all symmetry operations
                ncoord = op.operate(coord) % 1.0       # apply symmetry operation to this site, and produce ncoord
                if magnetic:
                    Mjp = op.operate_magmom(Mj).moment
                for it,ty in enumerate(grp):           # if coord is equivalent to any existing group in grp, say grp[i]==ty, then one of ncoord should be equal to first entry in grp[i]
                    coord0 = acoords[ty[0]]            # the first entry in the group grp[i][0]
                    M0 = amagmoms[ty[0]]
                    equal_coords = sum(abs(ncoord-coord0))<1e-5
                    equal_moment = sum(abs(M0-Mjp))<1e-5 if magnetic else True
                    #print('ncoord=', ncoord, 'coord0['+str(ty[0])+']=', coord0 )
                    if equal_coords and equal_moment:   # is it equal to current coord when a symmetry operation is applied?
                        #print(' ncoord=', ncoord, 'Mjp=', Mjp, 'is equivalent to typ', ty) # yes, coord is in group ty
                        Qequivalent=True               # we found which group it corresponds to
                        ty.append(ii)                  # grp[i]=ty is extended with coord[i]
                        #wop[it].append(ig)
                        break                          # loop over groups and also symmetry operations can be finished
                if Qequivalent:                        # once we find which group this site corresponds to, we can finish the loop over symmetry operations
                    break
            if not Qequivalent:                        # if this site (coord) is not equivalent to any existing group in grp, than we create a new group with a single entry [i]
                grp.append([ii])
            #print(name+' grp=', grp)
        groups.append(grp)  # all groups for this element
        grnames.append(name)
        print('name=', name, 'grp=', grp, file=log)

    Odd_magnet=False
    for i in range(len(groups)):
        for grp in groups[i]:
            print('grp=', grp, 'len(grp)=', len(grp), len(grp)%2, file=log)
            if len(grp)%2==1:
                Odd_magnet=True
    print('Odd_magnet=', Odd_magnet, file=log)
    if Odd_magnet:
        return 'Odd_magnet'
    
    atms = list(chain(*groups))
    atms = list(chain(*atms))
    is_Rotation=False
    for i in range(len(groups)):
        for grp in groups[i]: # one group of equivalent atoms that could have compensated moments
            orient={}
            for j in grp:
                orient[j]=0
            orient[grp[0]]=1
            Bipartite=True

            nearest_neigh={}
            for j in grp:
                distance={}
                #print(j, acoords[j])
                for k in grp:
                    if k==j:
                        distance[k]=0
                    else:
                        dR = (acoords[j]-acoords[k]) % 1.0
                        for r in range(3):
                            if dR[r]>0.5: dR[r]-=1
                        dst = linalg.norm(dR @ matrix)
                        distance[k]=dst
                        #print('   {:2d}{:2d}{:13.10f}'.format(j,k,dst), dR)
                #print('distance=', distance, 'grp=', grp)
                sorted_atms = sorted(grp, key=lambda i:distance[i])
                nearest_negh=None
                for nn in range(2,len(grp)):
                    #print('nn=', nn, 'dif=', abs(distance[sorted_atms[1]]-distance[sorted_atms[nn]]))
                    if abs(distance[sorted_atms[1]]-distance[sorted_atms[nn]])>1e-5:
                        nearest_negh = sorted_atms[1:nn]
                        break
                if nearest_negh==None:
                    nearest_negh = sorted_atms[1:]
                nearest_neigh[j] = nearest_negh
                
                for k in nearest_negh:
                    print('   {:2d} {:2d} {:13.10f}'.format(j,k,distance[k]), file=log)
                    
                    
            for j in grp:
                if orient[j]==0: # site j not yet oriented. Try any nearest neighbor
                    for k in nearest_neigh[j]:
                        if orient[k]!=0:
                            orient[j]=-orient[k]
                            print('Bipartite: Found k has orient['+str(k)+']=',orient[k], 'hence orient['+str(j)+']=', orient[j], file=log)
                for k in nearest_neigh[j]:
                    if orient[k]==0: # neigbor not yet set, set it to opposite to site j
                        orient[k] = -orient[j]
                    else: # already set. Is it bipartitie?
                        if orient[k]==orient[j]: # not bipartite
                            print('Bipartite: For j,k=',j,k,'the structure cant be bipartite', file=log)
                            Bipartite=False
                            break
                print('nn['+str(j)+']=', nearest_neigh[j], 'orient=', orient, file=log)
            print('Bipartite=', Bipartite, 'orient=', orient, file=log)

            if not Bipartite:
                return 'not_Bipartite'

            for j in grp:
                igroup={}
                for ig,op in enumerate(symmetry_operations): # loop over all symmetry operations
                    ncoord = op.operate(acoords[j]) % 1.0    # apply symmetry operation to this site, and produce ncoord
                    for k in nearest_neigh[j]:
                        if sum(abs(ncoord-acoords[k]))<1e-5:   # is it equal to current coord when a symmetry operation is applied?
                            igroup[k]=ig               # we found which group it corresponds to
                            break
                    if len(igroup)==len(nearest_neigh[j]):
                        break
                        
                print('group operations['+anames[j]+'_'+str(j)+']=', igroup, file=log)
                print('operations=', [symmetry_operations_type[g] for g in igroup.values()], file=log)
                is_Rotation = is_Rotation or any([symmetry_operations_type[g]=='R' for g in igroup.values()])
            print('is_Rotation=', is_Rotation, file=log)

    if is_Rotation:
        return 'YES'
    else:
        return 'Not_Rotation'

