import numpy as np


# A bottom streched vertical grid:
def sco_strech_bottom(z, thetab):
    return np.sinh(thetab*(z-1))/np.sinh(thetab) - 1.


# Function to define an uniform vertical grid:
def set_uniform_refvgrid(dz, jpk):
    depw = np.arange(0, dz * jpk, dz)
    dept = np.arange(0.5 * dz, dz * jpk, dz)
    e3w = np.ones(jpk)*dz
    e3t = np.ones(jpk)*dz
    return depw, dept, e3t, e3w


# Function to define vertical grid parameters in the full cells case:
def set_zcovgrid(bat, depw_1d, dept_1d, e3t_1d, e3w_1d):
    (jpi, jpj) = np.shape(bat)
    jpk = np.size(depw_1d)
    # set bottom level
    kbot = np.ones((jpi, jpj), dtype=int)*jpk - 1
    for k in np.arange(jpk - 2, -1, -1):
        kbot = np.where((bat < dept_1d[k]), k, kbot)

    # set scale factors and depths at T-points:
    e3t = np.zeros((jpi, jpj, jpk))
    e3w = np.zeros((jpi, jpj, jpk))
    depw = np.zeros((jpi, jpj, jpk))
    dept = np.zeros((jpi, jpj, jpk))
    bato = np.zeros((jpi, jpj))
    for k in np.arange(0, jpk, 1):
        e3t[:, :, k] = e3t_1d[k]
        e3w[:, :, k] = e3w_1d[k]
        depw[:, :, k] = depw_1d[k]
        dept[:, :, k] = dept_1d[k]

    for i in np.arange(0, jpi, 1):
        for j in np.arange(0, jpj, 1):
            for k in np.arange(0, kbot[i, j], 1):
                bato[i, j] = bato[i, j] + e3t[i, j, k]

    return kbot, bato, e3t, e3w, depw, dept


# Function to define vertical grid parameters in partial cell case:
def set_pstepvgrid(bat, depw_1d, dept_1d, e3t_1d, e3w_1d):
    (jpi, jpj) = np.shape(bat)
    jpk = np.size(depw_1d)
    # set bottom level
    kbot = np.ones((jpi, jpj), dtype=int)*jpk - 1
    for k in np.arange(jpk - 2, -1, -1):
        zmin = 0.1 * e3t_1d[k]
        kbot = np.where((bat < (depw_1d[k] + zmin)), k, kbot)

    # set scale factors and depths at T-points:
    e3t = np.zeros((jpi, jpj, jpk))
    e3w = np.zeros((jpi, jpj, jpk))
    depw = np.zeros((jpi, jpj, jpk))
    dept = np.zeros((jpi, jpj, jpk))
    bato = np.zeros((jpi, jpj))
    for k in np.arange(0, jpk, 1):
        e3t[:, :, k] = e3t_1d[k]
        e3w[:, :, k] = e3w_1d[k]
        depw[:, :, k] = depw_1d[k]
        dept[:, :, k] = dept_1d[k]

    for i in np.arange(0, jpi, 1):
        for j in np.arange(0, jpj, 1):
            k = np.int(kbot[i, j]) - 1
            if k >= 0:
                depw[i, j, k + 1] = min(bat[i, j], depw_1d[k + 1])
                e3t[i, j, k] = depw[i, j, k + 1] - depw[i, j, k]
                dept[i, j, k] = depw[i, j, k] + 0.5 * e3t[i, j, k]
                e3w[i, j, k] = dept[i, j, k] - dept[i, j, k - 1]

    for i in np.arange(0, jpi, 1):
        for j in np.arange(0, jpj, 1):
            for k in np.arange(0, kbot[i, j], 1):
                bato[i, j] = bato[i, j] + e3t[i, j, k]

    return kbot, bato, e3t, e3w, depw, dept


# Function to define vertical grid parameters in the s-coordinate case:
def set_scovgrid(bat, depw_1d, dept_1d, e3t_1d, e3w_1d, stype):
    (jpi, jpj) = np.shape(bat)
    jpk = np.size(depw_1d)
    jpkm1 = jpk - 1
    # set bottom level
    kbot = np.ones((jpi, jpj)) * jpkm1

    # set scale factors and depths at T-points:
    e3t = np.zeros((jpi, jpj, jpk))
    e3w = np.zeros((jpi, jpj, jpk))
    depw = np.zeros((jpi, jpj, jpk))
    dept = np.zeros((jpi, jpj, jpk))
    for k in np.arange(0, jpk, 1):
        e3t[:, :, k] = e3t_1d[k]
        e3w[:, :, k] = e3w_1d[k]
        depw[:, :, k] = depw_1d[k]
        dept[:, :, k] = dept_1d[k]

    # Bottom streched or uniform sigma distribution:
    for i in np.arange(0, jpi, 1):
        for j in np.arange(0, jpj, 1):
            if bat[i, j] > 0:
                for k in range(0, jpk):
                    z1 = np.float(k) / np.float(jpkm1)
                    z2 = (np.float(k)+0.5) / np.float(jpkm1)
                    if stype == 1:
                        depw[i, j, k] = bat[i, j] * sco_strech_bottom(z1, 4.)
                        dept[i, j, k] = bat[i, j] * sco_strech_bottom(z2, 4.)
                    elif stype == 0:
                        depw[i, j, k] = bat[i, j] * z1
                        dept[i, j, k] = bat[i, j] * z2

                for k in range(0, jpkm1):
                    e3t[i, j, k] = depw[i, j, k+1]-depw[i, j, k]
                    e3w[i, j, 0] = dept[i, j, 0]

                for k in range(1, jpk):
                    e3w[i, j, k] = dept[i, j, k] - dept[i, j, k-1]

    return kbot, bat, e3t, e3w, depw, dept


# Function to define vertical grid parameters in the s-coordinate on top of z case:
def set_scotopofzvgrid(bat, depmax, depw_1d, dept_1d, e3t_1d, e3w_1d):
    (jpi, jpj) = np.shape(bat)
    jpk = np.size(depw_1d)
    jpkm1 = jpk - 1

    # Set number of s levels:
    jpks = 0
    for k in range(0, jpk-1):
        if depw_1d[k+1] <= depmax:
            jpks = k

    # set bottom level in the z case first
    kbot = np.ones((jpi, jpj), dtype=int) * jpkm1
    for k in np.arange(jpk - 2, -1, -1):
        kbot = np.where((bat < dept_1d[k]), k, kbot)

    # set s coordinates on top:
    kbot = np.where(kbot < jpks, jpks, kbot)

    # set scale factors and depths at T-points:
    e3t = np.zeros((jpi, jpj, jpk))
    e3w = np.zeros((jpi, jpj, jpk))
    depw = np.zeros((jpi, jpj, jpk))
    dept = np.zeros((jpi, jpj, jpk))
    for k in np.arange(0, jpk, 1):
        e3t[:, :, k] = e3t_1d[k]
        e3w[:, :, k] = e3w_1d[k]
        depw[:, :, k] = depw_1d[k]
        dept[:, :, k] = dept_1d[k]

    # Uniform sigma distribution for the top jpks levels:
    for i in np.arange(0, jpi, 1):
        for j in np.arange(0, jpj, 1):
            if bat[i, j] > 0:
                zbat = min(bat[i, j], depw_1d[jpks])
                for k in range(0, jpks+1):
                    e3t[i, j, k] = zbat * e3t_1d[k] / depw_1d[jpks]
                    e3w[i, j, k] = zbat * e3w_1d[k] / depw_1d[jpks]
                e3w[i, j, 0] = 0.5 * e3t[i, j, 0]

    for i in np.arange(0, jpi, 1):
        for j in np.arange(0, jpj, 1):
            if bat[i, j] > 0.:
                dept[i, j, :] = np.cumsum(e3w[i, j, :])
                depw[i, j, :] = np.cumsum(e3t[i, j, :]) - e3t[i, j, 0]

    # Update bathymetry:
    for i in np.arange(0, jpi, 1):
        for j in np.arange(0, jpj, 1):
            bat[i, j] = depw[i, j, kbot[i, j]]

    return kbot, bat, e3t, e3w, depw, dept


def set_uvfvgrid(ln_zco, ln_zps, ln_sco, e3t_1d, e3w_1d, e3t, e3w):
    # Set vertical scale factors at U, V and F points
    #
    (jpi, jpj, jpk) = np.shape(e3t)
    e3u = np.zeros((jpi, jpj, jpk))
    e3v = np.zeros((jpi, jpj, jpk))
    e3uw = np.zeros((jpi, jpj, jpk))
    e3vw = np.zeros((jpi, jpj, jpk))
    e3f = np.zeros((jpi, jpj, jpk))

    for k in np.arange(0, jpk, 1):
        e3u[:, :, k] = e3t_1d[k]
        e3v[:, :, k] = e3t_1d[k]
        e3f[:, :, k] = e3t_1d[k]
        e3uw[:, :, k] = e3w_1d[k]
        e3vw[:, :, k] = e3w_1d[k]

    if (ln_zps == 1) | (ln_zco == 1):
        for i in np.arange(0, jpi - 1, 1):
            for j in np.arange(0, jpj - 1, 1):
                for k in np.arange(0, jpk - 1, 1):
                    e3u[i, j, k] = min(e3t[i, j, k], e3t[i + 1, j, k])
                    e3v[i, j, k] = min(e3t[i, j, k], e3t[i, j + 1, k])
                    e3uw[i, j, k] = min(e3w[i, j, k], e3w[i + 1, j, k])
                    e3vw[i, j, k] = min(e3w[i, j, k], e3w[i, j + 1, k])

        for i in np.arange(0, jpi - 1, 1):
            for j in np.arange(0, jpj - 1, 1):
                for k in np.arange(0, jpk - 1, 1):
                    e3f[i, j, k] = min(e3v[i, j, k], e3v[i + 1, j, k])
    elif ln_sco == 1:
        for i in np.arange(0, jpi - 1, 1):
            for j in np.arange(0, jpj - 1, 1):
                for k in np.arange(0, jpk - 1, 1):
                    e3u[i, j, k] = 0.5 * (e3t[i, j, k] + e3t[i + 1, j, k])
                    e3v[i, j, k] = 0.5 * (e3t[i, j, k] + e3t[i, j + 1, k])
                    e3uw[i, j, k] = 0.5 * (e3w[i, j, k] + e3w[i + 1, j, k])
                    e3vw[i, j, k] = 0.5 * (e3w[i, j, k] + e3w[i, j + 1, k])

        for i in np.arange(0, jpi - 1, 1):
            for j in np.arange(0, jpj - 1, 1):
                for k in np.arange(0, jpk - 1, 1):
                    e3f[i, j, k] = 0.5 * (e3v[i, j, k] + e3v[i + 1, j, k])

    return e3u, e3v, e3f, e3uw, e3vw
