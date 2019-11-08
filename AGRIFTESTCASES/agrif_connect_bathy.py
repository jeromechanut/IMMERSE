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
    i0end = nghost + imax-1     # Last Parent point inside zoom
    j0end = nghost + jmax-1

    upd = np.zeros((jpi0, jpj0))

    # Detect open boundaries and update update array accordingly:
    flagw = min(np.sum(tabp[i0st-1:i0st, j0st:j0end+1], axis=1), 1.)
    flage = min(np.sum(tabp[i0end+1:i0end+2, j0st:j0end+1], axis=1), 1.)
    flags = min(np.squeeze(np.sum(tabp[i0st:i0end+1, j0st-1:j0st], axis=0)), 1.)
    flagn = min(np.squeeze(np.sum(tabp[i0st:i0end+1, j0end+1:j0end+2], axis=0)), 1.)

    if flagw == 1:
        upd[0:i0st + nmatch, 1:jpj0 - 1] = 1
    else:
        upd[0:i0st, 1:jpj0 - 1] = 1
    if flage == 1:
        upd[i0end - nmatch + 1:jpi0, 1:jpj0 - 1] = 1
    else:
        upd[i0end + 1:jpi0, 1:jpj0 - 1] = 1
    if flags == 1:
        upd[1:jpi0 - 1, 0:j0st + nmatch] = 1
    else:
        upd[1:jpi0 - 1, 0:j0st] = 1
    if flagn == 1:
        upd[1:jpi0 - 1, j0end - nmatch + 1:jpj0] = 1
    else:
        upd[1:jpi0 - 1, j0end + 1:jpj0] = 1

    # Shift indexes to account for domain outside dynamical interface:
    ishift = np.int(np.ceil(np.float(nghost + 1)/np.float(iraf)))
    jshift = np.int(np.ceil(np.float(nghost + 1)/np.float(jraf)))

    if (flagw == 0) & (flage == 0):
        ishift = 0

    if (flags == 0) & (flagn == 0):
        jshift = 0

    for ip in range(i0st-ishift, i0end+1+ishift):
        ici = max(nghost + (ip-i0st)*iraf + 1, 0)
        ics = min(nghost + (ip-i0st+1)*iraf + 1, jpi1)
        for jp in range(j0st-jshift, j0end+1+jshift):
            jci = max(nghost + (jp-j0st)*jraf + 1, 0)
            jcs = min(nghost + (jp-j0st+1)*jraf + 1, jpj1)
            if upd[ip, jp] == 1:
                tabc[ici:ics, jci:jcs] = tabp[ip, jp]
    return tabc


def update_child_from_parent3(tabp, tabc, iraf, jraf, imin, jmin, imax, jmax, nghost):
    # This updates a T-point child array from parent in the ghost-cells area and
    # inside over the whole overlap
    # (Nearest neighbor)
    # (Works for any kind of refinement)

    (jpi1, jpj1) = np.shape(tabc)

    i0st = nghost + imin     # First Parent point inside zoom
    j0st = nghost + jmin
    i0end = nghost + imax-1  # Last Parent point inside zoom
    j0end = nghost + jmax-1

    # Shift indexes to account for domain outside dynamical interface:
    ishift = np.int(np.ceil(np.float(nghost + 1)/np.float(iraf)))
    jshift = np.int(np.ceil(np.float(nghost + 1)/np.float(jraf)))

    for ip in range(i0st-ishift, i0end+1+ishift):
        ici = max(nghost + (ip-i0st)*iraf + 1, 0)
        ics = min(nghost + (ip-i0st+1)*iraf + 1, jpi1)
        for jp in range(j0st-jshift, j0end+1+jshift):
            jci = max(nghost + (jp-j0st)*jraf + 1, 0)
            jcs = min(nghost + (jp-j0st+1)*jraf + 1, jpj1)
            tabc[ici:ics, jci:jcs] = tabp[ip, jp]
    return tabc


def update_child_and_parent_max(tabp, tabc, kbot0, kbot1, e3t0, e3t1, e3t_1d0, depw_1d0,
                                e3t_1d1, depw_1d1, iraf, jraf, imin, jmin, imax, jmax,
                                nghost, nmatch):
    # This updates a T-point child and parent bathymetry in
    # the connection zone in case they both have partial cells coordinates

    (jpi0, jpj0) = np.shape(tabp)
    (jpi1, jpj1) = np.shape(tabc)

    i0st = nghost + imin     # First Parent point inside zoom
    j0st = nghost + jmin
    i0end = nghost + imax-1  # Last Parent point inside zoom
    j0end = nghost + jmax-1

    upd = np.zeros((jpi0, jpj0))

    # Detect open boundaries and update update array accordingly:
    flagw = min(np.sum(tabp[i0st-1:i0st, j0st:j0end+1], axis=1), 1.)
    flage = min(np.sum(tabp[i0end+1:i0end+2, j0st:j0end+1], axis=1), 1.)
    flags = min(np.squeeze(np.sum(tabp[i0st:i0end+1, j0st-1:j0st], axis=0)), 1.)
    flagn = min(np.squeeze(np.sum(tabp[i0st:i0end+1, j0end+1:j0end+2], axis=0)), 1.)

    if flagw == 1:
        upd[0:i0st + nmatch, 1:jpj0 - 1] = 1
    else:
        upd[0:i0st, 1:jpj0 - 1] = 1
    if flage == 1:
        upd[i0end - nmatch + 1:jpi0, 1:jpj0 - 1] = 1
    else:
        upd[i0end + 1:jpi0, 1:jpj0 - 1] = 1
    if flags == 1:
        upd[1:jpi0 - 1, 0:j0st + nmatch] = 1
    else:
        upd[1:jpi0 - 1, 0:j0st] = 1
    if flagn == 1:
        upd[1:jpi0 - 1, j0end - nmatch + 1:jpj0] = 1
    else:
        upd[1:jpi0 - 1, j0end + 1:jpj0] = 1

    # Shift indexes to account for domain outside dynamical interface:
    ishift = np.int(np.ceil(np.float(nghost + 1)/np.float(iraf)))
    jshift = np.int(np.ceil(np.float(nghost + 1)/np.float(jraf)))

    if (flagw == 0) & (flage == 0):
        ishift = 0

    if (flags == 0) & (flagn == 0):
        jshift = 0

    for ip in range(i0st-ishift, i0end+1+ishift):
        ici = max(nghost + (ip-i0st)*iraf + 1, 0)
        ics = min(nghost + (ip-i0st+1)*iraf + 1, jpi1)
        for jp in range(j0st-jshift, j0end+1+jshift):
            jci = max(nghost + (jp-j0st)*jraf + 1, 0)
            jcs = min(nghost + (jp-j0st+1)*jraf + 1, jpj1)
            if kbot0[ip, jp] > 0:
                k0 = kbot0[ip, jp]-1
                h0 = tabp[ip, jp]
                a = tabc[ici:ics, jci:jcs]
                if (upd[ip, jp] == 1) & (np.size(a) > 0):
                    h1 = np.amax(a)
                    k1 = np.int(np.amax(kbot1[ici:ics, jci:jcs]))-1
                    #
                    zmaxn = max(h0, h1)
                    h0s = h0
                    h1s = h1
                    h0 = zmaxn
                    h1 = zmaxn
                    if zmaxn > depw_1d0[k0+1]:
                        k0 = k0 + 1
                    if zmaxn > depw_1d1[k1+1]:
                        k1 = k1 + 1
                    if (zmaxn - h0) < 0.1 * e3t_1d0[k0]:
                        h0 = h0s
                        h1 = h0s
                    if (zmaxn - h1) < 0.1 * e3t_1d1[k1]:
                        h0 = h1s
                        h1 = h1s
                    tabc[ici:ics, jci:jcs] = h1
                    tabp[ip, jp] = h0
    return tabp, tabc


def update_parent_from_child_zps(tabc, tabp, kbot, e3t, e3w, depw, dept, depw_1d,
                                 dept_1d, e3t_1d, e3w_1d, iraf, jraf, imin, jmin,
                                 imax, jmax, nghost, nmatch):
    # This updates parent zps vertical grid outside the connection zone
    # from the child volume averaged bathymetry

    (jpi0, jpj0) = np.shape(tabp)
    jpk = np.size(depw_1d)

    i0st = nghost + imin     # First Parent point inside zoom
    j0st = nghost + jmin
    i0end = nghost + imax-1  # Last Parent point inside zoom
    j0end = nghost + jmax-1

    upd = np.ones((jpi0, jpj0))

    # Detect open boundaries and update update array accordingly:
    flagw = min(np.sum(tabp[i0st-1:i0st, j0st:j0end+1], axis=1), 1.)
    flage = min(np.sum(tabp[i0end+1:i0end+2, j0st:j0end+1], axis=1), 1.)
    flags = min(np.squeeze(np.sum(tabp[i0st:i0end+1, j0st-1:j0st], axis=0)), 1.)
    flagn = min(np.squeeze(np.sum(tabp[i0st:i0end+1, j0end+1:j0end+2], axis=0)), 1.)

    if flagw == 1:
        upd[0:i0st + nmatch, 1:jpj0 - 1] = 0
    else:
        upd[0:i0st, 1:jpj0 - 1] = 0
    if flage == 1:
        upd[i0end - nmatch + 1:jpi0, 1:jpj0 - 1] = 0
    else:
        upd[i0end + 1:jpi0, 1:jpj0 - 1] = 0
    if flags == 1:
        upd[1:jpi0 - 1, 0:j0st + nmatch] = 0
    else:
        upd[1:jpi0 - 1, 0:j0st] = 0
    if flagn == 1:
        upd[1:jpi0 - 1, j0end - nmatch + 1:jpj0] = 0
    else:
        upd[1:jpi0 - 1, j0end + 1:jpj0] = 0

    for ip in range(i0st, i0end+1):
        ici = nghost + (ip-i0st)*iraf + 1
        ics = ici + iraf
        for jp in range(j0st, j0end+1):
            jci = nghost + (jp-j0st)*jraf + 1
            jcs = jci + jraf
            if upd[ip, jp] == 1.:
                # Fill with reference grid:
                for k in np.arange(0, jpk, 1):
                    e3t[ip, jp, k] = e3t_1d[k]
                    e3w[ip, jp, k] = e3w_1d[k]
                    depw[ip, jp, k] = depw_1d[k]
                    dept[ip, jp, k] = dept_1d[k]

                # Find new bottom level from child bathymetry:
                tabp[ip, jp] = np.average(tabc[ici:ics, jci:jcs])
                kbot[ip, jp] = jpk-1
                for k in np.arange(0, jpk-1, 1):
                    if tabp[ip, jp] > depw_1d[k]:
                        kbot[ip, jp] = k + 1

                # Special case for partial bottom cells
                k = kbot[ip, jp] - 1
                if k >= 0:
                    depw[ip, jp, k+1] = min(tabp[ip, jp], depw_1d[k+1])
                    e3t[ip, jp, k] = depw[ip, jp, k+1] - depw[ip, jp, k]
                    dept[ip, jp, k] = depw[ip, jp, k] + 0.5*e3t[ip, jp, k]
                    e3w[ip, jp, k] = dept[ip, jp, k] - dept[ip, jp, k-1]

    return kbot, tabp, e3t, e3w, depw, dept


def diag_parent_child_vol(tabc, tabp, iraf, jraf, imin, jmin, imax, jmax, nghost):
    # Print volume mismatch over overlapping area

    i0st = nghost + imin      # First Parent point inside zoom
    j0st = nghost + jmin
    i0end = nghost + imax - 1  # Last Parent point inside zoom
    j0end = nghost + jmax - 1

    print(' ')
    print('CHECK PARENT/CHILD VOLUME IN OVERLAPPING ZONE')
    for ip in range(i0st, i0end+1):
        ici = nghost + (ip-i0st)*iraf + 1
        ics = ici + iraf
        for jp in range(j0st, j0end+1):
            jci = nghost + (jp-j0st)*jraf + 1
            jcs = jci + jraf
            print('Parent/child average depth mismatch',
                  ip, jp, np.abs(tabp[ip, jp]-np.average(tabc[ici:ics, jci:jcs])))


def diag_child_parent_vol(tabc, tabp, iraf, jraf, imin, jmin, imax, jmax, nghost, nmatch):
    # Print depth mismatch over connection zone

    (jpi0, jpj0) = np.shape(tabp)
    (jpi1, jpj1) = np.shape(tabc)

    i0st = nghost + imin    # First Parent point inside zoom
    j0st = nghost + jmin
    i0end = nghost + imax-1  # Last Parent point inside zoom
    j0end = nghost + jmax-1

    upd = np.zeros((jpi0, jpj0))

    # Detect open boundaries and update update array accordingly:
    flagw = min(np.sum(tabp[i0st-1:i0st, j0st:j0end+1], axis=1), 1.)
    flage = min(np.sum(tabp[i0end+1:i0end+2, j0st:j0end+1], axis=1), 1.)
    flags = min(np.squeeze(np.sum(tabp[i0st:i0end+1, j0st-1:j0st], axis=0)), 1.)
    flagn = min(np.squeeze(np.sum(tabp[i0st:i0end+1, j0end+1:j0end+2], axis=0)), 1.)

    if flagw == 1:
        upd[0:i0st+nmatch, 1:jpj0-1] = 1
    if flage == 1:
        upd[i0end-nmatch+1:jpi0, 1:jpj0-1] = 1
    if flags == 1:
        upd[1:jpi0-1, 0:j0st+nmatch] = 1
    if flagn == 1:
        upd[1:jpi0-1, j0end-nmatch+1:jpj0] = 1

    # Shift indexes to account for domain outside dynamical interface:
    ishift = np.int(np.ceil(np.float(nghost + 1)/np.float(iraf)))
    jshift = np.int(np.ceil(np.float(nghost + 1)/np.float(jraf)))

    if (flagw == 0) & (flage == 0):
        ishift = 0

    if (flags == 0) & (flagn == 0):
        jshift = 0

    print(' ')
    print('CHECK PARENT/CHILD BATHYMETRIES IN CONNECTION ZONE')
    for ip in range(i0st-ishift, i0end+1+ishift):
        ici = max(nghost + (ip-i0st)*iraf + 1, 0)
        ics = min(nghost + (ip-i0st+1)*iraf + 1, jpi1)
        #
        for jp in range(j0st-jshift, j0end+1+jshift):
            jci = max(nghost + (jp-j0st)*jraf + 1, 0)
            jcs = min(nghost + (jp-j0st+1)*jraf + 1, jpj1)
            a = tabc[ici:ics, jci:jcs]
            if (upd[ip, jp] == 1) & (np.size(a) > 0):
                print('Parent/child max depth mismatch in connection zone (i, j, diff)',
                      ip, jp, ici, ics, jci, jcs, np.max(np.abs(tabp[ip, jp]-a)))


def set_scovgrid_step(bat, kbot, e3t, depw, e3u, e3v, nghost, nmatch, iraf, jraf):
    # Add steps in the sco case for connection and ghost points only
    (jpi, jpj, jpk) = np.shape(e3t)

    depw[:, :, 0] = 0.
    for k in np.arange(1, jpk, 1):
        depw[:, :, k] = depw[:, :, k-1] + e3t[:, :, k-1]

    batus = np.zeros((jpi, jpj))
    batvs = np.zeros((jpi, jpj))
    batu = np.zeros((jpi, jpj))
    batv = np.zeros((jpi, jpj))
    for i in np.arange(0, jpi - 1, 1):
        for j in np.arange(0, jpj - 1, 1):
            batus[i, j] = min(bat[i, j], bat[i + 1, j])
            batu[i, j] = 0.5 * (bat[i, j] + bat[i + 1, j])
            batvs[i, j] = min(bat[i, j], bat[i, j + 1])
            batv[i, j] = 0.5 * (bat[i, j] + bat[i, j + 1])

    # Detect open boundaries and update update array accordingly:
    flagw = min(np.sum(bat[nghost:nghost+1, :], axis=1), 1.)
    flage = min(np.sum(bat[jpi-nghost-1:jpi-nghost, :], axis=1), 1.)
    flags = min(np.squeeze(np.sum(bat[:, nghost:nghost+1], axis=0)), 1.)
    flagn = min(np.squeeze(np.sum(bat[:, jpj-nghost-1:jpj-nghost], axis=0)), 1.)

    # TODO: f-points case
    # and take a more careful look at update location
    upd_u = np.zeros((jpi, jpj))
    if flagw == 1:
        upd_u[0:nghost + nmatch*iraf, :] = 1
    if flage == 1:
        upd_u[jpi-nghost - nmatch*iraf - 1:jpi, :] = 1
    if flags == 1:
        upd_u[:, 0:nghost + nmatch*jraf] = 1
    if flagn == 1:
        upd_u[:, jpj - nghost - nmatch*jraf-1:jpj] = 1

    upd_v = np.zeros((jpi, jpj))
    if flagw == 1:
        upd_v[0:nghost + nmatch*iraf + 1, :] = 1
    if flage == 1:
        upd_v[jpi-nghost - nmatch*iraf:jpi, :] = 1
    if flags == 1:
        upd_v[:, 0:nghost + nmatch*jraf + 1] = 1
    if flagn == 1:
        upd_v[:, jpj - nghost - nmatch*jraf:jpj] = 1

    for i in np.arange(0, jpi - 1, 1):
        for j in np.arange(0, jpj - 1, 1):
            # U-points:
            zbot = batus[i, j]
            jkmax = np.int(min(kbot[i, j], kbot[i+1, j]) - 1)
            if (jkmax >= 0) & (upd_u[i, j] == 1):
                for k in np.arange(jkmax, 0, -1):
                    zsup = 0.5*(depw[i, j, k] + depw[i + 1, j, k])
                    if batus[i, j] < batu[i, j]:    # step to the right
                        zsup = min(zsup, zbot - 0.8 * e3t[i + 1, j, k])
                    else:                           # step to the left
                        zsup = min(zsup, zbot - 0.8 * e3t[i, j, k])
                    e3u[i, j, k] = zbot - zsup
                    zbot = zsup
            # V-points:
            zbot = batvs[i, j]
            jkmax = np.int(min(kbot[i, j], kbot[i, j+1]) - 1)
            if (jkmax >= 0) & (upd_v[i, j] == 1):
                for k in np.arange(jkmax, 0, -1):
                    zsup = 0.5 * (depw[i, j, k] + depw[i, j + 1, k])
                    if batvs[i, j] < batv[i, j]:    # step to the right
                        zsup = min(zsup, zbot - 0.8 * e3t[i, j + 1, k])
                    else:                           # step to the left
                        zsup = min(zsup, zbot - 0.8 * e3t[i, j, k])
                    e3v[i, j, k] = zbot - zsup
                    zbot = zsup

    return e3u, e3v
