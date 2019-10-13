import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np


def nc_write_bathy(fileout, lont, latt, batt):
    dataset = Dataset(fileout, 'w', format='NETCDF4_CLASSIC')
    (nx, ny) = np.shape(lont)
    x = dataset.createDimension('x', nx)
    y = dataset.createDimension('y', ny)
    nav_lon = dataset.createVariable('nav_lon', np.float32, ('y', 'x'))
    nav_lat = dataset.createVariable('nav_lat', np.float32, ('y', 'x'))
    bathy_meter = dataset.createVariable('bathy_meter', np.float32, ('y', 'x'))
    nav_lat.units = 'km'
    nav_lon.units = 'km'
    nav_lat.long_name = 'Y'
    nav_lon.long_name = 'X'
    bathy_meter.long_name = 'Bathymetry'
    bathy_meter.units = 'm'
    bathy_meter.missing_value = 0.

    nav_lon[:, :] = lont.T
    nav_lat[:, :] = latt.T
    bathy_meter[:, :] = batt.T
    dataset.close()


def nc_write_coord(fileout, lont, latt, lonu, latu, lonv, latv, lonf, latf,
                   e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f):
    dataset = Dataset(fileout, 'w', format='NETCDF4_CLASSIC')
    (nx, ny) = np.shape(lont)
    x = dataset.createDimension('x', nx)
    y = dataset.createDimension('y', ny)
    nav_lon = dataset.createVariable('nav_lon', np.float32, ('y', 'x'))
    nav_lat = dataset.createVariable('nav_lat', np.float32, ('y', 'x'))
    glamt = dataset.createVariable('glamt', np.float64, ('y', 'x'))
    glamu = dataset.createVariable('glamu', np.float64, ('y', 'x'))
    glamv = dataset.createVariable('glamv', np.float64, ('y', 'x'))
    glamf = dataset.createVariable('glamf', np.float64, ('y', 'x'))
    gphit = dataset.createVariable('gphit', np.float64, ('y', 'x'))
    gphiu = dataset.createVariable('gphiu', np.float64, ('y', 'x'))
    gphiv = dataset.createVariable('gphiv', np.float64, ('y', 'x'))
    gphif = dataset.createVariable('gphif', np.float64, ('y', 'x'))
    ge1t = dataset.createVariable('e1t', np.float64, ('y', 'x'))
    ge1u = dataset.createVariable('e1u', np.float64, ('y', 'x'))
    ge1v = dataset.createVariable('e1v', np.float64, ('y', 'x'))
    ge1f = dataset.createVariable('e1f', np.float64, ('y', 'x'))
    ge2t = dataset.createVariable('e2t', np.float64, ('y', 'x'))
    ge2u = dataset.createVariable('e2u', np.float64, ('y', 'x'))
    ge2v = dataset.createVariable('e2v', np.float64, ('y', 'x'))
    ge2f = dataset.createVariable('e2f', np.float64, ('y', 'x'))
    nav_lat.units = 'km'
    nav_lon.units = 'km'
    nav_lat.long_name = 'Y'
    nav_lon.long_name = 'X'
    nav_lon[:, :] = lont.T
    nav_lat[:, :] = latt.T
    #
    glamt[:, :] = lont.T
    glamu[:, :] = lonu.T
    glamv[:, :] = lonv.T
    glamf[:, :] = lonf.T
    gphit[:, :] = latt.T
    gphiu[:, :] = latu.T
    gphiv[:, :] = latv.T
    gphif[:, :] = latf.T
    #
    ge1t[:, :] = e1t.T
    ge1u[:, :] = e1u.T
    ge1v[:, :] = e1v.T
    ge1f[:, :] = e1f.T
    ge2t[:, :] = e2t.T
    ge2u[:, :] = e2u.T
    ge2v[:, :] = e2v.T
    ge2f[:, :] = e2f.T
    #
    dataset.close()


def nc_write_dom(fileout, ln_zco, ln_zps, ln_sco, ln_isfcav, jperio,
                 bat, lont, latt, lonu, latu, lonv, latv, lonf, latf,
                 e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f, ff_f, ff_t,
                 dept_1d, e3t_1d, e3w_1d, e3t, e3u, e3v, e3f, e3w, e3uw, e3vw,
                 ktop, kbot):
    dataset = Dataset(fileout, 'w', format='NETCDF4_CLASSIC')
    (nx, ny, nz) = np.shape(e3t)
    #
    x = dataset.createDimension('x', nx)
    y = dataset.createDimension('y', ny)
    z = dataset.createDimension('z', nz)
    nav_lon = dataset.createVariable('nav_lon', np.float32, ('y', 'x'))
    nav_lat = dataset.createVariable('nav_lat', np.float32, ('y', 'x'))
    nav_lev = dataset.createVariable('nav_lev', np.float32, 'z')
    giglo = dataset.createVariable('jpiglo', "i4")
    gjglo = dataset.createVariable('jpjglo', "i4")
    gkglo = dataset.createVariable('jpkglo', "i4")
    gperio = dataset.createVariable('jperio', "i4")
    gzco = dataset.createVariable('ln_zco', "i4")
    gzps = dataset.createVariable('ln_zps', "i4")
    gsco = dataset.createVariable('ln_sco', "i4")
    gcav = dataset.createVariable('ln_isfcav', "i4")

    ge3t1d = dataset.createVariable('e3t_1d', np.float64, 'z')
    ge3w1d = dataset.createVariable('e3w_1d', np.float64, 'z')
    gitop = dataset.createVariable('top_level', "i4", ('y', 'x'))
    gibot = dataset.createVariable('bottom_level', "i4", ('y', 'x'))
    gbat = dataset.createVariable('Bathymetry', np.float64, ('y', 'x'))
    glamt = dataset.createVariable('glamt', np.float64, ('y', 'x'))
    glamu = dataset.createVariable('glamu', np.float64, ('y', 'x'))
    glamv = dataset.createVariable('glamv', np.float64, ('y', 'x'))
    glamf = dataset.createVariable('glamf', np.float64, ('y', 'x'))
    gphit = dataset.createVariable('gphit', np.float64, ('y', 'x'))
    gphiu = dataset.createVariable('gphiu', np.float64, ('y', 'x'))
    gphiv = dataset.createVariable('gphiv', np.float64, ('y', 'x'))
    gphif = dataset.createVariable('gphif', np.float64, ('y', 'x'))
    ge1t = dataset.createVariable('e1t', np.float64, ('y', 'x'))
    ge1u = dataset.createVariable('e1u', np.float64, ('y', 'x'))
    ge1v = dataset.createVariable('e1v', np.float64, ('y', 'x'))
    ge1f = dataset.createVariable('e1f', np.float64, ('y', 'x'))
    ge2t = dataset.createVariable('e2t', np.float64, ('y', 'x'))
    ge2u = dataset.createVariable('e2u', np.float64, ('y', 'x'))
    ge2v = dataset.createVariable('e2v', np.float64, ('y', 'x'))
    ge2f = dataset.createVariable('e2f', np.float64, ('y', 'x'))
    gfff = dataset.createVariable('ff_f', np.float64, ('y', 'x'))
    gfft = dataset.createVariable('ff_t', np.float64, ('y', 'x'))
    ge3t = dataset.createVariable('e3t_0', np.float64, ('z', 'y', 'x'))
    ge3w = dataset.createVariable('e3w_0', np.float64, ('z', 'y', 'x'))
    ge3u = dataset.createVariable('e3u_0', np.float64, ('z', 'y', 'x'))
    ge3v = dataset.createVariable('e3v_0', np.float64, ('z', 'y', 'x'))
    ge3f = dataset.createVariable('e3f_0', np.float64, ('z', 'y', 'x'))
    ge3uw = dataset.createVariable('e3uw_0', np.float64, ('z', 'y', 'x'))
    ge3vw = dataset.createVariable('e3vw_0', np.float64, ('z', 'y', 'x'))

    nav_lat.units = 'km'
    nav_lon.units = 'km'
    nav_lat.long_name = 'Y'
    nav_lon.long_name = 'X'

    giglo[:] = nx
    gjglo[:] = ny
    gkglo[:] = nz
    gzco[:] = ln_zco
    gzps[:] = ln_zps
    gsco[:] = ln_sco
    gcav[:] = ln_isfcav
    gperio[:] = jperio

    nav_lon[:, :] = lont.T
    nav_lat[:, :] = latt.T
    nav_lev[:] = dept_1d
    ge3t1d[:] = e3t_1d
    ge3w1d[:] = e3w_1d
    #
    gitop[:, :] = ktop.T
    gibot[:, :] = kbot.T
    #
    gbat[:, :] = bat.T
    #
    glamt[:, :] = lont.T
    glamu[:, :] = lonu.T
    glamv[:, :] = lonv.T
    glamf[:, :] = lonf.T
    gphit[:, :] = latt.T
    gphiu[:, :] = latu.T
    gphiv[:, :] = latv.T
    gphif[:, :] = latf.T
    #
    ge1t[:, :] = e1t.T
    ge1u[:, :] = e1u.T
    ge1v[:, :] = e1v.T
    ge1f[:, :] = e1f.T
    ge2t[:, :] = e2t.T
    ge2u[:, :] = e2u.T
    ge2v[:, :] = e2v.T
    ge2f[:, :] = e2f.T
    gfff[:, :] = ff_f.T
    gfft[:, :] = ff_t.T
    #
    ge3t[:, :, :] = e3t.T
    ge3w[:, :, :] = e3w.T
    ge3u[:, :, :] = e3u.T
    ge3v[:, :, :] = e3v.T
    ge3f[:, :, :] = e3f.T
    ge3uw[:, :, :] = e3uw.T
    ge3vw[:, :, :] = e3vw.T
    #
    dataset.close()
