# -*- coding: utf-8 -*-
"""
Functions for calculating spherical harmonics, and synthesising geomagnetic 
field values using spherical harmonic models of the geomagnetic field such as 
the International Geomagnetic Reference Field (IGRF).

1. gd2gc - geodetic to geocentric colatitude conversion
2. rad_powers : powers of radial parameter (a/r)

Notes for improvements and additions:
------------------------------------
Some functions could be written as generators

21 Feb 2021
-----------


"""

import numpy as np

# Useful abbreviations
d2r = np.deg2rad
r2d = np.rad2deg

#==============================================================================

def gd2gc(h, gdcolat):
    
    """
    Geodetic to geocentric colatitude conversion.
    Uses WGS84 values for equatorial radius and reciprocal flattening.
      
    Parameters
    ----------
    h: 
        altitude (above the reference ellipsoid) in km
    gdcolat:
        geodetic colatitude
    
    Returns
    ------
    A tuple: 4 floats
        rad: the geocentric radius
        thc: geocentric colatitude
        sd and cd: cos and sin 'delta' - used to rotate geomagnetic X, Y and Z
        components from the geocentric frame into the geodetic frame.
        
    Dependencies
    ------------
    numpy

    """
    
    EQRAD = 6378.137
    FLAT  = 1/298.257223563
    
    plrad = EQRAD*(1-FLAT)
    
    e2    = EQRAD*EQRAD
    e4    = e2*e2
    p2    = plrad*plrad
    p4    = p2*p2
    
    ctgd  = np.cos(d2r(gdcolat))
    stgd  = np.sin(d2r(gdcolat))
    c2    = ctgd*ctgd
    s2    = 1-c2
    rho   = np.sqrt(e2*s2 + p2*c2)
    rad   = np.sqrt(h*(h+2*rho) + (e4*s2+p4*c2)/rho**2)
    cd    = (h+rho)/rad
    sd    = (e2-p2)*ctgd*stgd/(rho*rad)
    cthc  = ctgd*cd - stgd*sd       # Also: sthc = stgd*cd + ctgd*sd
    thc   = r2d(np.arccos(cthc))    # arccos returns values in [0, pi]
    return (rad, thc, sd, cd)

#==============================================================================

def rad_powers(nmax, a, r):
    
    """
    Calculate values of (a/r)^(n+2) for n=0, 1, 2 ..., nmax as required for 
    synthesising geomagnetic field values using spherical harmonic models of 
    the geomagnetic field such as the IGRF.
      
    Parameters
    ----------
    nmax: int
        The degree of the spherical harmonic model
    a: float
        The reference radius of the Earth (km)
    r: float
        The geocentric radius of the point at which geomagnetic field values
        are calculated
    
    Returns
    ------
    rp: a list of floats
        contains the values of (a/r)^(n+2) for n=0, 1, 2 ..., nmax
        
    Dependencies
    ------------
    None

    """
    
    f = a/r
    rp = [f*f]
    for i in range(nmax):
        rp.append(f*rp[-1])
    return rp

#==============================================================================

def csmphi(mmax,phi):
        
    """
    Populate arrays with values of cos(m*phi), sin(m*phi) m=0, 1, 2 ..., nmax 
    as used in synthesising geomagnetic field values using spherical harmonic 
    models of the geomagnetic field such as the IGRF.
      
    Parameters
    ----------
    mmax: int
        The maximum order (=degree) of the spherical harmonic model
    phi: float
        The longitude in degrees
    
    Returns
    ------
    rp: a tuple with two elements
        cmp: contains the values of cos(m*phi) for m=0, 1, 2 ..., mmax
        smp: contains the values of sin(m*phi) for m=0, 1, 2 ..., mmax
        
    Dependencies
    ------------
    numpy

    """

    mr = range(mmax+1)
    cmp = [np.cos(d2r(m*phi%360)) for m in mr]
    smp = [np.sin(d2r(m*phi%360)) for m in mr]
    return (cmp,smp)

#==============================================================================

def gh_phi_rad(gh, nmax, cp, sp, rp):
    """Populate arrays with terms such as g(3,2)*cos(2*phi)*(a/r)**5 """
    rx = np.zeros(nmax*(nmax+3)//2+1)
    ry = np.zeros(nmax*(nmax+3)//2+1)
    idx=-1
    hdx=-1
    for i in range(nmax+1):
        idx += 1
        hdx += 1
        rx[idx]= gh[hdx]*rp[i]
        for j in range(1,i+1):
            hdx += 2
            idx += 1
            rx[idx] = (gh[hdx-1]*cp[j] + gh[hdx]*sp[j])*rp[i]
            ry[idx] = (gh[hdx-1]*sp[j] - gh[hdx]*cp[j])*rp[i]
    return (rx, ry)

#==============================================================================

def gh_phi(gh, nmax, cp, sp):
    """As for gh_phi_rad but without the radial function"""
    rx = np.zeros(nmax*(nmax+3)//2+1)
    idx=-1
    hdx=-1
    for i in range(nmax+1):
        idx += 1
        hdx += 1
        rx[idx]= gh[hdx]
        for j in range(1,i+1):
            hdx += 2
            idx += 1
            rx[idx] = (gh[hdx-1]*cp[j] + gh[hdx]*sp[j])
    return rx

#==============================================================================

def idx_find(s, n, m):
    d = {
        'g': lambda n, m: n*n if m==0 else n*n+2*m-1,
        'h': lambda n, m: n*n+2*m,
        'p': lambda n, m: n*(n+1)//2+m
        }
    return d[s](n, m)
    
#==============================================================================

def pnm_calc(nmax, th):
    """Calculate arrays of the Associated Legendre Polynomials pnm"""

    # Initialise
    nel   = nmax*(nmax+3)//2+1
    pnm   = np.zeros(nel)
    ct    = np.cos(np.deg2rad(th)); st = np.sin(np.deg2rad(th))
    pnm[0] = 1
    if(nmax==0): return(pnm)
    pnm[1] = ct
    pnm[2] = st

    for i in range(2,nmax+1): # Loop over degree
        idx0 = idx_find('p',i,i)
        idx1 = idx_find('p',i-1,i-1)
        t1   = np.sqrt(1-1/(2*i))
        pnm[idx0] = t1*st*pnm[idx1]       
        for j in range(i):   # Loop over order
            idx0 = idx_find('p',i,j)
            idx1 = idx_find('p',i-1,j)
            idx2 = idx_find('p',i-2,j)
            t1 = (2*i-1)
            t2 = np.sqrt((i-1+j)*(i-1-j))
            t3 = np.sqrt((i+j)*(i-j))
            pnm[idx0] = (t1*ct*pnm[idx1] - t2*pnm[idx2])/t3           
    return pnm

#==============================================================================

def pxyznm_calc(nmax, th):
    """Calculate arrays of the Associated Legendre Polynomials pnm and the related 
    values xnm, ynm and znm which are needed to compute the X, Y and Z
    geomagnetic field components

    """
    # Initialise
    nel   = nmax*(nmax+3)//2+1
    pnm   = np.zeros(nel); xnm = np.zeros(nel)
    ynm   = np.zeros(nel); znm = np.zeros(nel)
    ct    = np.cos(np.deg2rad(th)); st = np.sin(np.deg2rad(th))
    pnm[0] = 1
    if(nmax==0): return(pnm)
    pnm[1] =  ct; pnm[2] = st
    xnm[0] = 0; xnm[1] = -st; xnm[2] = ct
    ynm[2] = 1
    znm[0] =-1; znm[1] =-2*ct; znm[2] =-2*st
    eps    = 10**(-6)

    for i in range(2,nmax+1): # Loop over degree
        idx0 = idx_find('p',i,i)
        idx1 = idx_find('p',i-1,i-1)
        t1   = np.sqrt(1-1/(2*i))
        pnm[idx0] = t1*st*pnm[idx1]
        xnm[idx0] = t1*(st*xnm[idx1] + ct*pnm[idx1])
        znm[idx0] =-(i+1)*pnm[idx0]
        if(np.abs(st) > eps):
            ynm[idx0] = i*pnm[idx0]/st
        else:
            ynm[idx0] = xnm[idx0]*ct

        for j in range(i):   # Loop over order
            idx0 = idx_find('p',i,j)
            idx1 = idx_find('p',i-1,j)
            idx2 = idx_find('p',i-2,j)
            t1 = (2*i-1)
            t2 = np.sqrt((i-1+j)*(i-1-j))
            t3 = np.sqrt((i+j)*(i-j))
            pnm[idx0] = (t1*ct*pnm[idx1] - t2*pnm[idx2])/t3
            xnm[idx0] = (t1*(ct*xnm[idx1] - st*pnm[idx1]) - t2*xnm[idx2])/t3
            znm[idx0] =-(i+1)*pnm[idx0]
            if(np.abs(st) > eps):
                ynm[idx0] = j*pnm[idx0]/st
            else:
                ynm[idx0] = xnm[idx0]*ct

    return (pnm, xnm, ynm, znm)

#==============================================================================

def shm_calculator(gh, nmax, altitude, colat, long, coord):
    """Compute values of the geomagnetic field from a model gh of 
    maximum degree and order nmax for either geodetic ot geocentric coordinates.
    """
    RREF     = 6371.2
    degree   = nmax
    phi      = long

    if (coord == 'Geodetic'):
        # Geodetic to geocentric conversion
        rad, theta, sd, cd = gd2gc(altitude, colat)
    else:
        rad   = altitude
        theta = colat

    # Create array with  values of (a/r)^(n+2) for n=0,1, 2 ..., degree
    rpow = rad_powers(degree, RREF, rad)
    # Create arrays with cos(m*phi), sin(m*phi)
    cmphi, smphi = csmphi(degree,phi)
    # Create arrays with terms such as g(3,2)*cos(2*phi)*(a/r)**5 
    ghxz, ghy = gh_phi_rad(gh, degree, cmphi, smphi, rpow)
    # Calculate arrays of the Associated Legendre Polynomials
    pnm, xnm, ynm, znm = pxyznm_calc(degree, theta)
    # Calculate geomagnetic field components are calculated as a dot product
    X =  np.dot(ghxz, xnm)
    Y =  np.dot(ghy,  ynm)
    Z =  np.dot(ghxz, znm)
    # Convert back to geodetic (X, Y, Z) if required
    if (coord == 'Geodetic'):
        t = X
        X = X*cd + Z*sd
        Z = Z*cd - t*sd

    return (X, Y, Z)

#==============================================================================


#====== Superseded functions ==================================================
# def pnmindex(n,m):
#     """Index for terms of degree=n and order=m in arrays pnm, xnm, 
# ynm and znm"""
#     return(n*(n+1)//2+m)
# 
# 
# def gnmindex(n,m):
#     if(m==0):
#         idx = n*n
#     else:
#         idx = n*n+2*m-1
#     return(idx)
# 
# 
# def hnmindex(n,m):
#     return(n*n+2*m)
# =============================================================================
# 
# def rad_powers(n, a, r):
#     """ Calculate values of (a/r)^(n+2) for n=0, 1, 2 ..., nmax."""
#     arp = np.zeros(n+1)
#     t0  = a/r
#     arp[0] = t0*t0
#     for i in range(n):
#         arp[i+1] = t0*arp[i]
#     return arp
# 
# =============================================================================
# def csmphi(m,phi):
#     """Populate arrays with cos(m*phi), sin(m*phi)."""
#     cmp = np.zeros(m+1)
#     smp = np.zeros(m+1)
#     cmp[0] = 1
#     smp[0] = 0
#     cp = np.cos(np.deg2rad(phi))
#     sp = np.sin(np.deg2rad(phi))
#     for i in range(m):
#         cmp[i+1] = cmp[i]*cp - smp[i]*sp
#         smp[i+1] = smp[i]*cp + cmp[i]*sp
#     return (cmp,smp)
# =============================================================================
# =============================================================================