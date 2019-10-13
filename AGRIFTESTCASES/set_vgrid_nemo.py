import numpy as np


# Function to define an uniform vertical grid:
def set_uniform_refvgrid(dz, jpk):
    depw = np.arange(0, dz * jpk, dz)
    dept = np.arange(0.5 * dz, dz * jpk, dz)
    e3w = np.ones(jpk)*dz
    e3t = np.ones(jpk)*dz
    return depw, dept, e3t, e3w


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
def set_scovgrid(bat, depw_1d, dept_1d, e3t_1d, e3w_1d):
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

    # Uniform sigma distribution:
    for i in np.arange(0, jpi, 1):
        for j in np.arange(0, jpj, 1):
            if bat[i, j] > 0:
                e3t[i, j, :] = bat[i, j] / np.float(jpkm1)
                e3w[i, j, :] = bat[i, j] / np.float(jpkm1)
                e3w[i, j, 0] = 0.5 * bat[i, j] / np.float(jpkm1)
                dept[i, j, :] = np.cumsum(e3w[i, j, :])
                depw[i, j, :] = np.cumsum(e3t[i, j, :]) - e3t[i, j, 0]

    return kbot, bat, e3t, e3w, depw, dept


def set_uvfvgrid(ln_zps, ln_sco, e3t_1d, e3w_1d, e3t, e3w):
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

    if ln_zps == 1:
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


def set_scovgrid_step(bat, depw_1d, dept_1d, e3t_1d, e3w_1d):
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

    # Uniform sigma distribution:
    for i in np.arange(0, jpi, 1):
        for j in np.arange(0, jpj, 1):
            if bat[i, j] > 0:
                e3t[i, j, :] = bat[i, j] / np.float64(jpkm1)
                e3w[i, j, :] = bat[i, j] / np.float64(jpkm1)
                e3w[i, j, 0] = 0.5 * bat[i, j] / np.float64(jpkm1)
                #
                dept[i, j, :] = np.cumsum(e3w[i, j, :])
                depw[i, j, :] = np.cumsum(e3t[i, j, :]) - e3t[i, j, 0]

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

    for i in np.arange(0, jpi - 1, 1):
        for j in np.arange(0, jpj - 1, 1):
            zbot = batus[i, j]
            for k in np.arange(jpk - 2, -1, -1):
                zsup = 0.5 * (depw[i, j, k] + depw[i + 1, j, k])
                if batus[i, j] < batu[i, j]:  # step to the right
                    zsup = min(zsup, zbot - 0.8 * e3t[i + 1, j, k])
                else: # step to the left
                    zsup = min(zsup, zbot - 0.8 * e3t[i, j, k])
                e3u[i, j, k] = zbot - zsup
                zbot = zsup

    return kbot, bat, e3t, e3w, depw, dept, e3u
