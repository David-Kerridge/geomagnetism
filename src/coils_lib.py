# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:49:20 2021

@author: david
"""

def b_rect_coil(xp, yp, zp, zc, lx, ly, amps):
    
    """
    Function b_rect_coil to compute the axial and radial field components 
    of the magnetic field produced by a single (filamentary) rectangular coil. 
    The centre of the coil is assumed to be at x=y=0. The coil current is 
    taken as positive when it flows anticlockwise when viewed from z>zc.
    
    (Units: lengths are in metres, the coil current is in Amps and the 
     magnetic field values returned are in nT.)
 
    Parameters
    ----------
    xp, yp, zp: the coordinates of the point where the magnetic field
                components are to be computed
    zc: the z-coordinate of the centre of the coil
    lx, ly: lengths of the coil in the x and y directions
    amps: the coil current
    
    Returns
    ------
    A tuple:
        bx, by, bz: the magnetic field in the coordinate directions
        
    Dependencies
    ------------
    numpy
    
    Example
    -------
    b_rect_coil(0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 1.0) -> -97.54, -19.94, 794.10

    """
    
    import numpy as np
    
    func_1 = lambda p1, p2, p3, p4: (p1+p2)/((p1+p2)**2 + (p3+p4)**2)
    
    func_2 = lambda p1, p2, p3, p4, p5, p6: (p1+p2) \
                   /np.sqrt((p1+p2)**2 + (p3+p4)**2 + (p5+p6)**2)
                   
    to_nT = 100.0*amps     # mu_0/(4*pi) * 10**9 * current
    x2 = lx/2
    y2 = ly/2
    zc = -zc
  
# x-component
    sx1  = func_1(zp, zc, xp, -x2)
    sx2  = func_1(zp, zc, xp,  x2)
    tx1  = func_2(yp, -y2, xp, -x2, zp, zc)
    tx2  = func_2(yp,  y2, xp, -x2, zp, zc)
    tx3  = func_2(yp, -y2, xp,  x2, zp, zc)
    tx4  = func_2(yp,  y2, xp,  x2, zp, zc)
    bx   = -sx1*(tx1-tx2) + sx2*(tx3-tx4)
    
# y-component
    sy1  = func_1(zp, zc, yp, -y2)
    sy2  = func_1(zp, zc, yp,  y2)
    ty1  = func_2(xp, -x2, yp, -y2, zp, zc)
    ty2  = func_2(xp,  x2, yp, -y2, zp, zc)
    ty3  = func_2(xp, -x2, yp,  y2, zp, zc)
    ty4  = func_2(xp,  x2, yp,  y2, zp, zc)
    by   = -sy1*(ty1-ty2) + sy2*(ty3-ty4)
    
# z-component
    sz1  = func_1(xp, -x2, zp, zc)
    sz2  = func_1(xp,  x2, zp, zc)
    sz3  = func_1(yp, -y2, zp, zc)
    sz4  = func_1(yp,  y2, zp, zc)
    bz   = sz1*(tx1-tx2) - sz2*(tx3-tx4) + sz3*(ty1-ty2) - sz4*(ty3-ty4)

    return bx*to_nT, by*to_nT, bz*to_nT

#=============================================================================

def b_circ_coil(rp, zp, zc, a, amps):
    
    """
    Function b_circ_coil to compute the radial and axial magnetic fields 
    produced by a single circular coil. The general formulae involve elliptic 
    integrals. The coordinate system is assumed to be cylindrical. The 
    magnetic field is only a function of the radial and axial coordinates
    (r and z).
    
    (Units: lengths are in metres, the coil current is in Amps and the 
     magnetic field values returned are in nT.)
      
    Parameters
    ----------
    rp: the radial coordinate of the point where the field is to be calculated
    zp: the axial coordinate of the point where the field is to be calculated
    zc: the z-coordinate of the centre of the coil
    a: the coil radius
    amps: the coil current
    
    Returns
    ------
    A tuple: 2 floats
        br: the radial magnetic field
        bz: the axial magnetic field
        
    Dependencies
    ------------
    numpy
    scipy: for the complete elliptical integrals E and K. (These can also be
    computed using the formulae given in Abromowitz and Stegun (1965) - see
    functions and E_approx and K_approx.)

    """

    import numpy as np
    from scipy.special import ellipk as K
    from scipy.special import ellipe as E

    to_nT    = 200*amps     # mu_0/(2*pi) * 10**9 * current
    apr  = a+rp
    apr2 = apr*apr
    amr  = a-rp
    amr2 = amr*amr
    dz   = zp-zc
    dz2  = dz*dz
    k2   = 4*a*rp/(apr2+dz2)
    t1   = apr*amr-dz2
    t2   = amr2+dz2
    t3   = 1/np.sqrt(apr2+dz2)
    bz   = (K(k2)+E(k2)*t1/t2)*t3
    t1   = (apr2+amr2)/2+dz2
    if rp == 0:
        br = 0
    else:
        br =(E(k2)*t1/t2-K(k2))*t3*dz/rp
        
    return br*to_nT, bz*to_nT

#=============================================================================
def K_approx(m):

    """    
    Complete elliptic integral of the first kind (K) using the approximate 
    formula given in Abromowitz and Stegun (1965)
    
    In Legendre notation, K(phi,k), k is the elliptic modulus and is 
    related to the parameter m by m = k**2
    
    Parameters
    ----------
    m: the (square root of the elliptic modulus)
    
    Returns
    ------
    km: the elliptic integral (K)
        
    Dependencies
    ------------
    numpy
    
    """

    import numpy as np
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012
    m1 = 1-m
    km = a0 + a1*m1 + a2*m1**2 + a3*m1**3 + a4*m1**4
    km = km + (b0 + b1*m1 + b2*m1**2 + b3*m1**3 + b4*m1**4)*np.log(1/m1)
    return km

#=============================================================================

def E_approx(m):
    
    """
    Complete elliptic integral of the second kind (E) using the approximate 
    formula given in Abromowitz and Stegun (1965)
    
    In Legendre notation, E(phi,k), k is the elliptic modulus and is related
    to the parameter m by m = k**2
    
    Parameters
    ----------
    m: the (square root of the elliptic modulus)
    
    Returns
    ------
    em: the elliptic integral (E)
        
    Dependencies
    ------------
    numpy
    
    """

    import numpy as np
    a0 = 1.0
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b0 = 0.0
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639
    m1 = 1-m
    em = a0 + a1*m1 + a2*m1**2 + a3*m1**3 + a4*m1**4
    em = em + (b0+b1*m1 + b2*m1**2 + b3*m1**3 + b4*m1**4)*np.log(1/m1)
    return em

#=============================================================================


if __name__ == "__main__":

    #
    # Set up constants
    I    = 1           # Total current in amps (equivalent to nturns*current)
    xl   = 1           # Length of the coil in the x direction in m
    yl   = 2           # Length of the coil in the y direction in m
    zc   = 0.5         # z coordinate of the coil centre in m
    xp   = 0.1       # x coordinate of the test point in m
    yp   = 0.2        # y coordinate of the test point in m
    zp   = 0.3        # z coordinate of the test point in m
    Bx, By, Bz = b_rect_coil(xp,yp,zp,zc,xl,yl,I)
    print(Bx,By,Bz)
