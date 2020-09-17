"""
Build DOME mesh with an AGRIF zoom
"""
import matplotlib.pyplot as plt
import numpy as np
from dom_wri_nemo import *
from set_vgrid_nemo import *
from agrif_connect_bathy import *

# Define Parent grid below:
dx0 = 10000.
dz0 = 100.
ln_zco0 = 0
ln_zps0 = 1
ln_sco0 = 0
stype0 = 0
ln_isfcav0 = 0
jperio0 = 0

# Define Child here:
zoom_ist = 138
zoom_iend = 178
zoom_jst = 58
zoom_jend = 80

iraf = 3 
jraf = 3 
dz1 = 50.
ln_zco1 = 0
ln_zps1 = 1
ln_sco1 = 0
stype1 = 0
ln_isfcav1 = 0
jperio1 = 0

# -------------------------------------------------
write_dom = 1
match_bat = 1
nghost = 3
nmatch = 2
# -------------------------------------------------
# Start the DOME special case
# -------------------------------------------------
dy0 = dx0
# Parent grid sizes
jpi0 = np.int(2000.e3 / dx0) + 2
jpj0 = np.int((800.e3 + 50.e3) / dy0) + 2 + 1
jpk0 = np.int(np.round(3600. / dz0) + 1)
zoffx0 = -(0.5 + 1700.e3/dx0)
zoffy0 = -(0.5 + 800.e3/dy0)
#
# Child grid sizes
dx1 = dx0 / np.float(iraf)
dy1 = dy0 / np.float(jraf)
jpi1 = (zoom_iend - zoom_ist) * iraf + 2 + 2 * nghost
jpj1 = (zoom_jend - zoom_jst) * jraf + 2 + 2 * nghost
jpk1 = np.int(np.round(3600. / dz1) + 1)
zoffx1 = -(0.5 + 1700.e3/dx1 + np.float(nghost)) + (nghost + zoom_ist - 1) * iraf
zoffy1 = -(0.5 + 800.e3/dx1 + np.float(nghost)) + (nghost + zoom_jst - 1) * jraf


def set_dome_hgrid(dx, dy, jpi, jpj, zoffx, zoffy):
    # Set grid positions [km]
    latt = np.zeros((jpi, jpj))
    lont = np.zeros((jpi, jpj))
    lonu = np.zeros((jpi, jpj))
    latu = np.zeros((jpi, jpj))
    lonv = np.zeros((jpi, jpj))
    latv = np.zeros((jpi, jpj))
    lonf = np.zeros((jpi, jpj))
    latf = np.zeros((jpi, jpj))

    for i in range(0, jpi):
        lont[i, :] = zoffx * dx * 1.e-3 + dx * 1.e-3 * np.float(i)
        lonu[i, :] = zoffx * dx * 1.e-3 + dx * 1.e-3 * (np.float(i) + 0.5)

    for j in range(0, jpj):
        latt[:, j] = zoffy * dy * 1.e-3 + dy * 1.e-3 * float(j)
        latv[:, j] = zoffy * dy * 1.e-3 + dy * 1.e-3 * (float(j) + 0.5)

    lonv = lont
    lonf = lonu
    latu = latt
    latf = latv

    e1t = np.ones((jpi, jpj)) * dx
    e2t = np.ones((jpi, jpj)) * dy
    e1u = np.ones((jpi, jpj)) * dx
    e2u = np.ones((jpi, jpj)) * dy
    e1v = np.ones((jpi, jpj)) * dx
    e2v = np.ones((jpi, jpj)) * dy
    e1f = np.ones((jpi, jpj)) * dx
    e2f = np.ones((jpi, jpj)) * dy

    # Set bathymetry [m]:
    batt = np.where((latt > 0.), 600., 600.-latt*1000.*0.01)

    # Set surface mask:
    ktop = np.zeros((jpi, jpj))
    ktop[1:jpi - 1, 1:jpj - 1] = 1
    ktop = np.where((latt > 0.), 0., ktop)
    ktop = np.where((latt > 0.) & (lont > -50.) & (lont < 50.), 1., ktop)
    ktop[:, jpj-1] = 0
    batt = np.where((ktop == 0.), 0., batt)

    # Set coriolis parameter:
    ff_t = np.zeros((jpi, jpj)) + 1.e-4
    ff_f = np.zeros((jpi, jpj)) + 1.e-4

    return lont, latt, lonu, latu, lonv, latv, lonf, latf, \
           e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f, batt, ktop, ff_f, ff_t

# Set parent horizontal grid:
(lont0, latt0, lonu0, latu0, lonv0, latv0, lonf0, latf0,
 e1t0, e2t0, e1u0, e2u0, e1v0, e2v0, e1f0, e2f0,
 batt0, ktop0, ff_f0, ff_t0) = set_dome_hgrid(dx0, dy0, jpi0, jpj0, zoffx0, zoffy0)

# Set child horizontal grid:
(lont1, latt1, lonu1, latu1, lonv1, latv1, lonf1, latf1,
 e1t1, e2t1, e1u1, e2u1, e1v1, e2v1, e1f1, e2f1,
 batt1, ktop1, ff_f1, ff_t1) = set_dome_hgrid(dx1, dy1, jpi1, jpj1, zoffx1, zoffy1)

# -------------------------------------------------
# Match domains
# -------------------------------------------------
if match_bat == 1:
    # Update parent bathymetry in overlapping region:
    batt0 = update_parent_from_child(batt1, batt0, iraf, jraf, zoom_ist, zoom_jst, zoom_iend, zoom_jend, nghost)
    # Update child bathymetry near the interface:
    batt1 = update_child_from_parent2(batt0, batt1, iraf, jraf, zoom_ist, zoom_jst, zoom_iend, zoom_jend, nghost, nmatch)
    # Or update child bathymetry over the whole domain:
    # batt1 = update_child_from_parent3(batt0, batt1, iraf, jraf, zoom_ist, zoom_jst, zoom_iend, zoom_jend, nghost)

# Set parent vertical grid at T-points:
(depw_1d0, dept_1d0, e3t_1d0, e3w_1d0) = set_uniform_refvgrid(dz0, jpk0)
if ln_zco0 == 1:
    (kbot0, batt0, e3t0, e3w0, depw0, dept0) = \
    set_zcovgrid(batt0, depw_1d0, dept_1d0, e3t_1d0, e3w_1d0)
elif ln_zps0 == 1:
    (kbot0, batt0, e3t0, e3w0, depw0, dept0) = \
    set_pstepvgrid(batt0, depw_1d0, dept_1d0, e3t_1d0, e3w_1d0)
elif ln_sco0 == 1:
    (kbot0, batt0, e3t0, e3w0, depw0, dept0) = \
    set_scovgrid(batt0, depw_1d0, dept_1d0, e3t_1d0, e3w_1d0, stype0)

# Set child vertical grid at T-points:
(depw_1d1, dept_1d1, e3t_1d1, e3w_1d1) = set_uniform_refvgrid(dz1, jpk1)
if ln_zco1 == 1:
    (kbot1, batt1, e3t1, e3w1, depw1, dept1) = \
    set_zcovgrid(batt1, depw_1d1, dept_1d1, e3t_1d1, e3w_1d1)
elif ln_zps1 == 1:
    (kbot1, batt1, e3t1, e3w1, depw1, dept1) = \
    set_pstepvgrid(batt1, depw_1d1, dept_1d1, e3t_1d1, e3w_1d1)
elif ln_sco1 == 1:
    (kbot1, batt1, e3t1, e3w1, depw1, dept1) = \
    set_scovgrid(batt1, depw_1d1, dept_1d1, e3t_1d1, e3w_1d1, stype1)

if match_bat == 1:
# Adjust grids with partial cells (beta):
    if (ln_zps0 == 1) & (ln_zps1 == 1):
        (batt0, batt1) = update_child_and_parent_max(batt0, batt1, kbot0, kbot1, e3t0, e3t1, e3t_1d0, depw_1d0,
	    					     e3t_1d1, depw_1d1, iraf, jraf, zoom_ist, zoom_jst, zoom_iend,
                                                     zoom_jend, nghost, nmatch)

        (kbot0, batt0, e3t0, e3w0, depw0, dept0) = set_pstepvgrid(batt0, depw_1d0, dept_1d0, e3t_1d0, e3w_1d0)
        (kbot1, batt1, e3t1, e3w1, depw1, dept1) = set_pstepvgrid(batt1, depw_1d1, dept_1d1, e3t_1d1, e3w_1d1)

    if (ln_sco0 == 0) & (ln_sco1 == 1):
        batt1 = update_child_from_parent2(batt0, batt1, iraf, jraf, zoom_ist, zoom_jst, zoom_iend, zoom_jend, nghost,
                                          nmatch)
        (kbot1, batt1, e3t1, e3w1, depw1, dept1) = \
            set_scovgrid(batt1, depw_1d1, dept_1d1, e3t_1d1, e3w_1d1, stype1)

    if (ln_sco0 == 1) & (ln_sco1 == 0):
        batt0 = update_parent_from_child(batt1, batt0, iraf, jraf, zoom_ist, zoom_jst, zoom_iend, zoom_jend, nghost)
        (kbot0, batt0, e3t0, e3w0, depw0, dept0) = set_scovgrid(batt0, depw_1d0, dept_1d0, e3t_1d0, e3w_1d0, stype0)

    if ln_zps0 == 1:
        (kbot0, batt0, e3t0, e3w0, depw0, dept0) = \
        update_parent_from_child_zps(batt1, batt0, kbot0, e3t0, e3w0, depw0, dept0, depw_1d0, dept_1d0,
                                     e3t_1d0, e3w_1d0, iraf, jraf, zoom_ist, zoom_jst, zoom_iend,
                                     zoom_jend, nghost, nmatch)

# Set vertical grids at UVF-points:
(e3u0, e3v0, e3f0, e3uw0, e3vw0) = set_uvfvgrid(ln_zco0, ln_zps0, ln_sco0, e3t_1d0, e3w_1d0, e3t0, e3w0)
(e3u1, e3v1, e3f1, e3uw1, e3vw1) = set_uvfvgrid(ln_zco1, ln_zps1, ln_sco1, e3t_1d1, e3w_1d1, e3t1, e3w1)

# TBD: Finalize steps with s-coordinate over overlapping region:
if (ln_sco1 == 1) & (ln_sco0 == 0):
    (e3u1, e3v1) = set_scovgrid_step(batt1, kbot1, e3t1, depw1, e3u1, e3v1, nghost, nmatch, iraf, jraf)

# -------------------------------------------------
# Checks:
# -------------------------------------------------
# Print volume difference on parent grid:
diag_parent_child_vol(batt1, batt0, iraf, jraf, zoom_ist, zoom_jst, zoom_iend, zoom_jend, nghost)

# Print mas depth difference in the connection zone:
diag_child_parent_vol(batt1, batt0, iraf, jraf, zoom_ist, zoom_jst, zoom_iend, zoom_jend, nghost, nmatch)

ktop0 = np.where((batt0 == 0.), 0, ktop0)  # Adjust top level in case land mask has changed in the matching process
kbot0 = np.where((batt0 == 0.), 0, kbot0)
ktop1 = np.where((batt1 == 0.), 0, ktop1)
kbot1 = np.where((batt1 == 0.), 0, kbot1)

# -------------------------------------------------
# Write output:
# -------------------------------------------------
if write_dom == 1:
    # Parent:
    nc_write_dom('DOME_domcfg.nc', ln_zco0, ln_zps0, ln_sco0, ln_isfcav0, jperio0,
                 batt0, lont0, latt0, lonu0, latu0, lonv0, latv0, lonf0, latf0,
                 e1t0, e2t0, e1u0, e2u0, e1v0, e2v0, e1f0, e2f0, ff_f0, ff_t0,
                 dept_1d0, e3t_1d0, e3w_1d0, e3t0, e3u0, e3v0, e3f0, e3w0, e3uw0, e3vw0, ktop0, kbot0)
    nc_write_bathy('DOME_bathy_meter.nc', lont0, latt0, batt0)
    nc_write_coord('DOME_coordinates.nc', lont0, latt0, lonu0, latu0, lonv0, latv0, lonf0, latf0,
                   e1t0, e2t0, e1u0, e2u0, e1v0, e2v0, e1f0, e2f0)
    # Child:
    nc_write_dom('1_DOME_domcfg.nc', ln_zco1, ln_zps1, ln_sco1, ln_isfcav1, jperio1,
                 batt1, lont1, latt1, lonu1, latu1, lonv1, latv1, lonf1, latf1,
                 e1t1, e2t1, e1u1, e2u1, e1v1, e2v1, e1f1, e2f1, ff_f1, ff_t1,
                 dept_1d1, e3t_1d1, e3w_1d1, e3t1, e3u1, e3v1, e3f1, e3w1, e3uw1, e3vw1, ktop1, kbot1)
    nc_write_bathy('1_DOME_bathy_meter.nc', lont1, latt1, batt1)
    nc_write_coord('1_DOME_coordinates.nc', lont1, latt1, lonu1, latu1, lonv1, latv1, lonf1, latf1,
                   e1t1, e2t1, e1u1, e2u1, e1v1, e2v1, e1f1, e2f1)

kbotu0 = np.zeros((jpi0, jpj0), dtype=int)
kbotv0 = np.zeros((jpi0, jpj0), dtype=int)
kbotu1 = np.zeros((jpi1, jpj1), dtype=int)
kbotv1 = np.zeros((jpi1, jpj1), dtype=int)
batu0 = np.zeros((jpi0, jpj0))
batv0 = np.zeros((jpi0, jpj0))
batu1 = np.zeros((jpi1, jpj1))
batv1 = np.zeros((jpi1, jpj1))

for i in np.arange(0, jpi0 - 1, 1):
    for j in np.arange(0, jpj0 - 1, 1):
        kbotu0[i, j] = min(kbot0[i, j], kbot0[i + 1, j])
        kbotv0[i, j] = min(kbot0[i, j], kbot0[i, j + 1])
        for k in np.arange(0, kbotu0[i, j], 1):
            batu0[i, j] = batu0[i, j] + e3u0[i, j, k]
        for k in np.arange(0, kbotv0[i, j], 1):
            batv0[i, j] = batv0[i, j] + e3v0[i, j, k]

for i in np.arange(0, jpi1 - 1, 1):
    for j in np.arange(0, jpj1 - 1, 1):
        kbotu1[i, j] = min(kbot1[i, j], kbot1[i + 1, j])
        kbotv1[i, j] = min(kbot1[i, j], kbot1[i, j + 1])
        for k in np.arange(0, kbotu1[i, j], 1):
            batu1[i, j] = batu1[i, j] + e3u1[i, j, k]
        for k in np.arange(0, kbotv1[i, j], 1):
            batv1[i, j] = batv1[i, j] + e3v1[i, j, k]

print(' ')
print('ZOOM POSITION')
print('South West corner Interface Index [], X Position [km] and Depth [m]:')
print('Parent', nghost + zoom_ist - 1, lonu0[nghost + zoom_ist - 1, nghost + zoom_jst],
      batu0[nghost + zoom_ist - 1, nghost + zoom_jst])
print('Child', nghost, nghost + 1, lonu1[nghost, nghost + 1],
      batu1[nghost, nghost + 1])
print('South West corner Interface Index [], Y Position [km] and Depth [m]:')
print('Parent', nghost + zoom_jst - 1, latv0[nghost + zoom_ist, nghost + zoom_jst - 1],
      batv0[nghost + zoom_ist, nghost + zoom_jst - 1])
print('Child', nghost + 1, nghost, latv1[nghost + 1, nghost],
      batv1[nghost + 1, nghost])
#
print('North East corner Interface Index [], X Position [km] and Depth [m]:')
print('Parent', nghost + zoom_iend - 1, lonu0[nghost + zoom_iend - 1, nghost + zoom_jend - 1],
      batu0[nghost + zoom_iend - 1, nghost + zoom_jend - 1])
print('Child', jpi1 - nghost - 2, lonu1[jpi1 - nghost - 2, jpj1 - nghost - 2],
      batu1[jpi1 - nghost - 2, jpj1 - nghost - 2])
print('North East corner Interface Index [] Y Position [km] and Depth [m]:')
print('Parent', nghost + zoom_jend - 1, latv0[nghost + zoom_iend - 1, nghost + zoom_jend - 1],
      batv0[nghost + zoom_iend - 1, nghost + zoom_jend - 1])
print('Child', jpj1 - nghost - 2, latv1[jpi1 - nghost - 2, jpj1 - nghost - 2],
      batv1[jpi1 - nghost - 2, jpj1 - nghost - 2])

# Draw a figure
depwu0 = np.zeros((jpi0, jpj0, jpk0))
kbot0u = np.zeros((jpi0, jpj0), dtype=int) + jpk0 - 1
for i in np.arange(0, jpi0 - 1, 1):
    for j in np.arange(0, jpj0 - 1, 1):
        kbot0u[i, j] = min(kbot0[i, j], kbot0[i + 1, j])

for i in np.arange(0, jpi0 - 1, 1):
    for j in np.arange(0, jpj0 - 1, 1):
        for k in np.arange(1, kbot0u[i, j]+1, 1):
            depwu0[i, j, k] = depwu0[i, j, k-1] + e3u0[i, j, k-1]
        for k in np.arange(kbot0u[i, j]+1, jpk0, 1):
            depwu0[i, j, k] = np.nan

depwu1 = np.zeros((jpi1, jpj1, jpk1))
kbot1u = np.zeros((jpi1, jpj1), dtype=int) + jpk1 - 1
for i in np.arange(0, jpi1 - 1, 1):
    for j in np.arange(0, jpj1 - 1, 1):
        kbot1u[i, j] = min(kbot1[i, j], kbot1[i + 1, j])

for i in np.arange(0, jpi1 - 1, 1):
    for j in np.arange(0, jpj1 - 1, 1):
        for k in np.arange(1, kbot1u[i, j]+1, 1):
            depwu1[i, j, k] = depwu1[i, j, k-1] + e3u1[i, j, k-1]
        for k in np.arange(kbot1u[i, j]+1, jpk1, 1):
            depwu1[i, j, k] = np.nan

fig = plt.figure()
ax = fig.add_subplot(111)
batt0 = np.where((batt0 == 0.), np.nan, batt0)
batt0 = np.ma.masked_invalid(batt0)
batt1 = np.where((batt1 == 0.), np.nan, batt1)
# remove ghosts from plot
batt1[0:nghost+1, :] = np.nan
batt1[jpi1-nghost-1:jpi1-1, :] = np.nan
batt1[:, 0:nghost+1] = np.nan
batt1[:, jpj1-nghost-1:jpj1-1] = np.nan
batt1 = np.ma.masked_invalid(batt1)
plt.pcolor(lont0-0.5*dx0*1.e-3, latt0-0.5*dy0*1.e-3, batt0, vmin=np.nanmin(batt0), vmax=np.nanmax(batt0), edgecolor='k')
plt.pcolor(lont1-0.5*dx1*1.e-3, latt1-0.5*dy1*1.e-3, batt1, vmin=np.nanmin(batt0), vmax=np.nanmax(batt0), edgecolor='r')
ax.set_xlim(-500, 250)
ax.set_ylim(-300,  70)
ax.set_aspect('equal')
plt.xlabel('X [km]')
plt.ylabel('Y [km]')
plt.title('DOME')
plt.savefig('DOME_zoom.png')
#
fig = plt.figure()
ax = fig.add_subplot(111)
isub = 1
i0p = 170
i0c = (i0p - zoom_ist) * iraf + nghost + 2
j0 = np.int((nghost + zoom_jst - 1) / isub)

for k in np.arange(0, jpk0, isub):
    plt.plot(latu0[i0p, nghost + zoom_jst - 1 - j0:nghost + zoom_jst - 1 + nmatch + 1],
             np.squeeze(depwu0[i0p, nghost + zoom_jst - 1 - j0:nghost + zoom_jst - 1 + nmatch + 1, k]), 'k')
for j in np.arange(0, nghost + zoom_jst, isub):
    plt.plot(latu0[i0p, j] * np.ones(jpk0), np.squeeze(depwu0[i0p, j, :]), 'k')
for k in np.arange(0, jpk1, isub):
    plt.plot(latu1[i0c, 1:jpj1-1],
             np.squeeze(depwu1[i0c, 1:jpj1-1, k]), 'r')
for j in np.arange(nghost, jpj1 - nghost - 1, isub):
    plt.plot(latu1[i0c, j] * np.ones(jpk1), np.squeeze(depwu1[i0c, j, :]), 'r')

plt.gca().invert_yaxis()
ax.set_xlim(-300,  70)
plt.xlabel('Y [km]')
plt.ylabel('Depth [m]')
plt.title('DOME')
plt.savefig('DOME_zoom_section.png')
plt.show()
