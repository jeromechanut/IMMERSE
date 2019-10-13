import numpy as np


def update_child_from_parent(tabp, tabc, iraf, jraf, imin, jmin, imax, jmax, nghost):
    # This updates a T-point child array from parent strictly inside the dynamical domain
    # (Nearest neighbor)
    # (Works for any kind of refinement)

    i0st = nghost + imin   # First Parent point inside zoom
    j0st = nghost + jmin
    i0end = nghost + imax   # Last Parent point inside zoom
    j0end = nghost + jmax

    for ip in range(i0st, i0end):
        ici = nghost + (ip-i0st)*iraf + 1
        ics = ici + iraf
        for jp in range(j0st, j0end):
            jci = nghost + (jp-j0st)*jraf + 1
            jcs = jci + jraf
            tabc[ici:ics, jci:jcs] = tabp[ip, jp]
    return tabc


def update_parent_from_child(tabc, tabp, iraf, jraf, imin, jmin, imax, jmax, nghost):
    # This update a T-point parent array strictly inside child domain
    # (Cell average)
    # (Works for any kind of refinement)
    i0st = nghost + imin  # First Parent point inside zoom
    j0st = nghost + jmin
    i0end = nghost + imax  # Last Parent point inside zoom
    j0end = nghost + jmax

    for ip in range(i0st, i0end):
        ici = nghost + (ip-i0st)*iraf + 1
        ics = ici + iraf
        for jp in range(j0st, j0end):
            jci = nghost + (jp-j0st)*jraf + 1
            jcs = jci + jraf
            tabp[ip, jp] = np.average(tabc[ici:ics, jci:jcs])
    return tabp


def update_child_from_parent2(tabp, tabc, iraf, jraf, imin, jmin, imax, jmax, nghost, nmatch):
    # This updates a T-point child array from parent in the ghost-cells area and
    # inside over nmatch parent points
    # (Nearest neighbor)
    # (Works for any kind of refinement)

    (jpi0, jpj0) = np.shape(tabp)
    (jpi1, jpj1) = np.shape(tabc)

    i0st = nghost + imin        # First Parent point inside zoom
    j0st = nghost + jmin
    i0end = nghost + imax-1     # First Parent point inside zoom
    j0end = nghost + jmax-1

    upd = np.zeros((jpi0,jpj0))
    if iraf>1:
# no extension in ghost cells:
#        	upd[i0st:i0st+nmatch,:]=1
#        	upd[i0end-nmatch+1:i0end+1,:]=1
        upd[:i0st+nmatch,:]=1
        upd[i0end-nmatch+1:,:]=1
    if jraf>1:
# no extension in ghost cells:
#        	upd[:,j0st:j0st+nmatch]=1
#        	upd[:,j0end-nmatch+1:j0end+1]=1
        upd[:,:j0st+nmatch]=1
        upd[:,j0end-nmatch+1:]=1

        
    ishift = np.int(np.ceil(np.float(nghost + 1)/np.float(iraf)))
    jshift = np.int(np.ceil(np.float(nghost + 1)/np.float(jraf)))
    if iraf==1: ishift = 0
    if jraf==1: jshift = 0

    for ip in range(i0st-ishift,i0end+1+ishift):
        ici = max(nghost + (ip-i0st)*iraf + 1,0)
        ics = min(ici + iraf, jpi1)
        for jp in range(j0st-jshift,j0end+1+jshift):
            jci = max(nghost + (jp-j0st)*jraf + 1,0)
            jcs = min(jci + jraf,jpj1)
            if upd[ip,jp]==1:
                tabc[ici:ics, jci:jcs] = tabp[ip,jp]
    return tabc

def update_child_from_parent3(tabp, tabc, iraf, jraf, imin, jmin, imax, jmax, nghost):
    # This updates a T-point child array from parent in the ghost-cells area and
    # inside over the whole overlap
    # (Nearest neighbor)
    # (Works for any kind of refinement)

    (jpi0,jpj0) = np.shape(tabp)
    (jpi1,jpj1) = np.shape(tabc)

    i0st  = nghost + imin     # First Parent point inside zoom
    j0st  = nghost + jmin
    i0end = nghost + imax - 1 # First Parent point inside zoom
    j0end = nghost + jmax - 1

    (jpi0,jpj0) = np.shape(tabp)
        
    ishift = np.int(np.ceil(np.float(nghost + 1)/np.float(iraf)))
    jshift = np.int(np.ceil(np.float(nghost + 1)/np.float(jraf)))

    for ip in range(i0st-ishift,i0end+1+ishift):
        ici = max(nghost + (ip-i0st)*iraf + 1,0)
        ics = min(ici + iraf, jpi1)
        for jp in range(j0st-jshift,j0end+1+jshift):
            jci = max(nghost + (jp-j0st)*jraf + 1,0)
            jcs = min(jci + jraf,jpj1)
            tabc[ici:ics, jci:jcs] = tabp[ip,jp]
    return tabc


def update_child_and_parent_max(tabp, tabc, kbot0, kbot1, e3t0, e3t1, e3t_1d0, depw_1d0, e3t_1d1, depw_1d1, \
                iraf, jraf, imin, jmin, imax, jmax, nghost, nmatch):

    (jpi0,jpj0) = np.shape(tabp)
    (jpi1,jpj1) = np.shape(tabc)

    i0st  = nghost + imin     # First Parent point inside zoom
    j0st  = nghost + jmin
    i0end = nghost + imax - 1 # First Parent point inside zoom
    j0end = nghost + jmax - 1

    (jpi0,jpj0) = np.shape(tabp)

    upd = np.zeros((jpi0,jpj0))
    if iraf>1:
            upd[:i0st+nmatch,:]=1
            upd[i0end-nmatch+1:,:]=1
    if jraf>1:
            upd[:,:j0st+nmatch]=1
            upd[:,j0end-nmatch+1:]=1
        
    ishift = np.int(np.ceil(np.float(nghost + 1)/np.float(iraf)))
    jshift = np.int(np.ceil(np.float(nghost + 1)/np.float(jraf)))
    if iraf==1: ishift = 0
    if jraf==1: jshift = 0

    for ip in range(i0st-ishift,i0end+1+ishift):
        ici = max(nghost + (ip-i0st)*iraf + 1,0)
        ics = min(ici + iraf, jpi1-1)
        for jp in range(j0st-jshift,j0end+1+jshift):
            jci = max(nghost + (jp-j0st)*jraf + 1,0)
            jcs = min(jci + jraf,jpj1-1)
            if kbot0[ip,jp]	> 0:
                k0 = kbot0[ip,jp]-1
                e3tbot0 = e3t0[ip,jp,k0]
                a = tabc[ici:ics, jci:jcs]
                if upd[ip,jp]==1 & np.size(a)>0:
                    h0 = tabp[ip,jp]
                    h1 = np.amax(a)
                    k1 = np.int(np.amax(kbot1[ici:ics, jci:jcs]))
                    e3tbot1 = np.amax(e3t1[ici:ics, jci:jcs,k1-1])
                    zmaxn = max(h0, h1)
                    #print('test1',ip,jp,h0-h1,h0,h1,e3tbot0, e3tbot1)
                    h0s = h0   ; h1s = h1 ; h0 = zmaxn ; h1 = zmaxn
                    if zmaxn > depw_1d0[k0+1] : k0 = k0 + 1
                    if zmaxn > depw_1d1[k1+1] : k1 = k1 + 1
                    if (zmaxn - h0) < 0.1 * e3t_1d0[k0] : h0 = h0s ; h1 = h0s
                    if (zmaxn - h1) < 0.1 * e3t_1d1[k1] : h0 = h1s ; h1 = h1s
                    tabc[ici:ics, jci:jcs] = h1
                    tabp[ip,jp] = h0
    return tabp, tabc

def update_parent_from_child_zps(tabc, tabp, kbot, e3t, e3w, depw, dept, depw_1d, dept_1d, e3t_1d, e3w_1d, iraf, jraf, imin, jmin, imax, jmax, nghost, nmatch):
    (jpi0,jpj0) = np.shape(tabp)
    jpk = np.size(depw_1d)

    i0st  = nghost + imin     # First Parent point inside zoom
    j0st  = nghost + jmin
    i0end = nghost + imax - 1 # First Parent point inside zoom
    j0end = nghost + jmax - 1

    upd = np.ones((jpi0,jpj0))
    if iraf>1:
        upd[0:i0st+nmatch,:]=0.
        upd[i0end-nmatch+1:jpi0,:]=0.
    if jraf>1:
        upd[:,0:j0st+nmatch]=0.
        upd[:,j0end-nmatch+1:jpj0]=0.

    for ip in range(i0st,i0end+1):
        ici = nghost + (ip-i0st)*iraf + 1
        ics = ici + iraf
        for jp in range(j0st,j0end+1):
            jci = nghost + (jp-j0st)*jraf + 1
            jcs = jci + jraf
            if upd[ip,jp]==1. :
                tabp[ip,jp] = np.average(tabc[ici:ics, jci:jcs])
                kbot[ip,jp] = jpk-1
                for k in np.arange(0,jpk-1,1):
                    if tabp[ip,jp]>depw_1d[k]: kbot[ip,jp] = k+1
                for k in np.arange(0,jpk,1):
                    e3t[ip,jp,k] = e3t_1d[k]
                    e3w[ip,jp,k] = e3w_1d[k]
                    depw[ip,jp,k] = depw_1d[k]
                    dept[ip,jp,k] = dept_1d[k]
                k=kbot[ip,jp]-1
                if k >= 0:
                    depw[ip,jp,k+1]=min(tabp[ip,jp],depw_1d[k+1])
                    e3t[ip,jp,k]=depw[ip,jp,k+1]-depw[ip,jp,k]
                    dept[ip,jp,k]=depw[ip,jp,k]+0.5*e3t[ip,jp,k]
                    e3w[ip,jp,k]=dept[ip,jp,k]-dept[ip,jp,k-1]

    return kbot, tabp, e3t, e3w, depw, dept


def diag_parent_child_vol(tabc, tabp, iraf, jraf, imin, jmin, imax, jmax, nghost):

    i0st  = nghost + imin     # First Parent point inside zoom
    j0st  = nghost + jmin
    i0end = nghost + imax - 1 # First Parent point inside zoom
    j0end = nghost + jmax - 1

    for ip in range(i0st, i0end+1):
        ici = nghost + (ip-i0st)*iraf + 1
        ics = ici + iraf
        for jp in range(j0st, j0end+1):
            jci = nghost + (jp-j0st)*jraf + 1
            jcs = jci + jraf
            print('Parent/child average depth mismatch', ip, jp, np.abs(tabp[ip,jp]-np.average(tabc[ici:ics, jci:jcs])))

def diag_child_parent_vol(tabc, tabp, iraf, jraf, imin, jmin, imax, jmax, nghost, nmatch):	
    (jpi0,jpj0) = np.shape(tabp)
    (jpi1,jpj1) = np.shape(tabc)

    i0st  = nghost + imin   # First Parent point inside zoom
    j0st  = nghost + jmin
    i0end = nghost + imax-1 # First Parent point inside zoom
    j0end = nghost + jmax-1

    (jpi0,jpj0) = np.shape(tabp)
    upd = np.zeros((jpi0, jpj0))

    upd[0:i0st+nmatch-1, 1:jpj0-1] = 1
    upd[i0end-nmatch+1-1:jpi0, 1:jpj0-1] = 1
    upd[1:jpi0, 0:j0st+nmatch] = 1
    upd[1:jpi0, j0end-nmatch+1:jpj0] = 1

    ishift = np.int(np.ceil(np.float(nghost + 1)/np.float(iraf)))
    jshift = np.int(np.ceil(np.float(nghost + 1)/np.float(jraf)))
    if iraf == 1: ishift = 0
    if jraf == 1: jshift = 0

    for ip in range(i0st-ishift, i0end+1+ishift):
        ici = max(nghost + (ip-i0st)*iraf + 1, 0)
        ics = min(ici + iraf, jpi1)
        for jp in range(j0st-jshift, j0end+1+jshift):
            jci = max(nghost + (jp-j0st)*jraf + 1, 0)
            jcs = min(jci + jraf,jpj1)
            a = tabc[ici:ics, jci:jcs]
            if upd[ip,jp] == 1 & np.size(a) > 0:
                print('Parent/child max depth mismatch', ip, jp, np.max(np.abs(tabp[ip,jp]-a)))

