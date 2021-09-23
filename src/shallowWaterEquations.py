import numpy as np
try:
  from postProcessCore import *
  canWriteToVtk = True
except:
  print('Error importing postproocessing core, cannot use VTK')
  canWriteToVtk = False
class shallowWaterEquations:
  nvars = 3
  def __init__(self,bc_type=None,bc_function=None,source=None,params=None):
    self.computeSource = source
    self.params = params
    self.bc_type = bc_type 
    if (bc_type == "CUSTOM_BCS"):
      self.getBoundaryStateFromInteriorState = bc_function
    else:
      self.getBoundaryStateFromInteriorState = None

  def writeSol(self,string,U,grid,writeToVtk=True):
    if canWriteToVtk and writeToVtk:
      triangle_faces_to_VTK(string,x=grid.triQ.points[:,0],y=grid.triQ.points[:,1],z=grid.triQ.points[:,0]*0,faces = grid.triQ.vertices,point_data=None,\
               cell_data={'h':np.mean(U[0].flatten()[grid.triQ.vertices[:,:]],axis=1),'hU':np.mean(U[1].flatten()[grid.triQ.vertices[:,:]],axis=1),\
               'hV':np.mean(U[2].flatten()[grid.triQ.vertices[:,:]],axis=1)})
    np.savez(string,U=U)

  def flux(self,u):
    fx = np.zeros(np.shape(u))
    fy = np.zeros(np.shape(u))
    g = 9.8
    h = u[0]
    hU = u[1]
    hV = u[2]
    fx[0] = u[1]
    fx[1] = hU*hU/h + 0.5*g*(h**2)
    fx[2] = hU*hV/h

    fy[0] = u[2]
    fy[1] = hU*hV/(h)
    fy[2] = hV*hV/(h) + 0.5*g*(h**2)
    return fx,fy

  def fluxJac(self,u):
    nvars = np.shape(u)[0]
    shp = (np.append(nvars,np.shape(u)))
    g = 9.8
    h = u[0]
    hU = u[1]
    hV = u[2]

    fxJac = np.zeros(shp)
    fyJac = np.zeros(shp)

    fxJac[0,0] = 0.
    fxJac[0,1] = 1.
    fxJac[0,2] = 0.
    fxJac[1,0] = g*h - hU**2/h**2
    fxJac[1,1] = 2.*hU/h 
    fxJac[1,2] = 0.
    fxJac[2,0] = -hU*hV/h**2 
    fxJac[2,1] = hV/h
    fxJac[2,2] = hU/h

    fyJac[0,0] = 0.
    fyJac[0,1] = 0.
    fyJac[0,2] = 1.
    fyJac[1,0] = -hU*hV/h**2 
    fyJac[1,1] = hV/h 
    fyJac[1,2] = hU/h 
    fyJac[2,0] = g*h - hV**2/h**2 
    fyJac[2,1] = 0. 
    fyJac[2,2] = 2.*hV/h 
    return fxJac,fyJac
   

  def inviscidFlux(self,UL,UR,n):
  # PURPOSE: This function calculates the flux for the SWE equations
  # using the rusanov flux function
  #
  # INPUTS:
  #    UL: conservative state vector in left cell
  #    UR: conservative state vector in right cell
  #    n: normal pointing from the left cell to the right cell
  #
  # OUTPUTS:
  #  F   : the flux out of the left cell (into the right cell)
  #  smag: the maximum propagation speed of disturbance
  #
    g = 9.8 
    #process left state
    es = 1.e-30
    hL = UL[0]
    uL = UL[1]/(hL + es)
    vL = UL[2]/(hL + es)
    unL = uL*n[0] + vL*n[1]
  
    pL = 0.5*g*(hL**2)
    FL = np.zeros(np.shape(UL),dtype=UL.dtype)
    FL[0] = hL*unL
    FL[1] = UL[1]*unL + pL*n[0]
    FL[2] = UL[2]*unL + pL*n[1]

    # process right state
    hR = UR[0]
    uR = UR[1]/(hR + es)
    vR = UR[2]/(hR + es)
    unR = uR*n[0] + vR*n[1] 
    pR = 0.5*g*(hR**2)
    # right flux
    FR = np.zeros(np.shape(UR),dtype=UR.dtype)
    FR[0] = hR*unR
    FR[1] = UR[1]*unR + pR*n[0]
    FR[2] = UR[2]*unR + pR*n[1]
  
    # difference in states
    du = UR - UL
    # rho average
    hm = 0.5*(hL + hR)
    um = (unL*(hL)**0.5 + unR*(hR)**0.5 )/((hL)**0.5 + (hR)**0.5  + es) 
    #% eigenvalues
    smax = np.abs(um) + np.abs((g*hm)**0.5)
    F = np.zeros(np.shape(FL))  # for allocation
    F[0]    = 0.5*(FL[0]+FR[0])-0.5*smax*(UR[0] - UL[0])
    F[1]    = 0.5*(FL[1]+FR[1])-0.5*smax*(UR[1] - UL[1])
    F[2]    = 0.5*(FL[2]+FR[2])-0.5*smax*(UR[2] - UL[2])
    return F


  def inviscidFluxJac(self,UL,UR,n):
  ## Computes the flux Jacobian for the shallow water equations  

    nvars = np.shape(UL)[0]
    shp = (np.append(nvars,np.shape(UL)))

    g = 9.8
    #process left state
    es = 1.e-30
    hL = UL[0]
    uL = UL[1]/(hL + es)
    vL = UL[2]/(hL + es)
    unL = uL*n[0] + vL*n[1]
 
    # process right state
    hR = UR[0]
    uR = UR[1]/(hR + es)
    vR = UR[2]/(hR + es)
    unR = uR*n[0] + vR*n[1]

    # rho average
    hm = 0.5*(hL + hR)
    um = (unL*(hL)**0.5 + unR*(hR)**0.5 )/((hL)**0.5 + (hR)**0.5  + es)
    #% eigenvalues
    smax = np.abs(um) + np.abs((g*hm)**0.5)

    termL = (n[0]*UL[1] + n[1]*UL[2]) /UL[0]**2
    termR = (n[0]*UR[1] + n[1]*UR[2]) / UR[0]**2

    hL_sqrt = np.sqrt(hL)
    hR_sqrt = np.sqrt(hR)

    hsqrt_un = hL_sqrt*unL + hR_sqrt*unR + es
    dsmaxL = np.zeros(np.shape(UL),dtype=UL.dtype)
    dsmaxR = np.zeros(np.shape(UR),dtype=UR.dtype)

    JL = np.zeros(shp,dtype=UL.dtype)
    JR = np.zeros(shp,dtype=UR.dtype)

    dsmaxL[0] = - abs( hsqrt_un ) / (2.*hL_sqrt*pow(hL_sqrt + hR_sqrt,2.) ) +\
              (0.5*unL / hL_sqrt - hL_sqrt*termL )*hsqrt_un  / ( (hL_sqrt + hR_sqrt)*abs( hsqrt_un  ) ) +\
              g/(pow(2.,3./2.)*pow(g*(hL + hR),0.5) );
    dsmaxL[1] = n[0]*hsqrt_un / ( hL_sqrt*(hL_sqrt + hR_sqrt) * abs( hsqrt_un) )
    dsmaxL[2] = n[1]*hsqrt_un / ( hL_sqrt*(hL_sqrt + hR_sqrt) * abs( hsqrt_un) )


    dsmaxR[0] = - abs( hsqrt_un ) / (2.*hR_sqrt*pow(hL_sqrt + hR_sqrt,2.) ) + \
                (0.5*unR / hR_sqrt - hR_sqrt*termR )*hsqrt_un  / ( (hL_sqrt + hR_sqrt)*abs( hsqrt_un ) ) + \
                g/(pow(2.,3./2.)*pow(g*(hL + hR),0.5) )
    dsmaxR[1] = n[0]*hsqrt_un / ( hR_sqrt*(hL_sqrt + hR_sqrt) * abs( hsqrt_un) )
    dsmaxR[2] = n[1]*hsqrt_un / ( hR_sqrt*(hL_sqrt + hR_sqrt) * abs( hsqrt_un) )

    # jacobian w.r.p to the left state
    JL[0][0] = -0.5*dsmaxL[0]*(UR[0] - UL[0]) + 0.5*(n[0]*uL + n[1]*vL - UL[0]*termL)  + 0.5*smax;
    JL[0][1] = 0.5*n[0] - 0.5*dsmaxL[1]*(UR[0] - UL[0]);
    JL[0][2] = 0.5*n[1] - 0.5*dsmaxL[2]*(UR[0] - UL[0]);

    JL[1][0] = 0.5*(g*n[0]*UL[0] - UL[1]*termL) - 0.5*dsmaxL[0]*(UR[1] - UL[1]);
    JL[1][1] = n[0]*uL  + 0.5*n[1]*vL + 0.5*smax -0.5*dsmaxL[1]*(UR[1] - UL[1]);
    JL[1][2] = 0.5*n[1]*uL  -0.5*dsmaxL[2]*(UR[1] - UL[1]);

    JL[2][0] = 0.5*(g*n[1]*UL[0] - UL[2]*termL) - 0.5*dsmaxL[0]*(UR[2] - UL[2]);
    JL[2][1] = 0.5*n[0]*vL  -0.5*dsmaxL[1]*(UR[2] - UL[2]);
    JL[2][2] = n[1]*vL + 0.5*n[0]*uL + 0.5*smax -0.5*dsmaxL[2]*(UR[2] - UL[2]);

    #jacobian w.r.p to the right state
    JR[0][0] = -0.5*dsmaxR[0]*(UR[0] - UL[0]) + 0.5*(n[0]*uR + n[1]*vR - UR[0]*termR)  - 0.5*smax;
    JR[0][1] = 0.5*n[0] - 0.5*dsmaxR[1]*(UR[0] - UL[0]);
    JR[0][2] = 0.5*n[1] - 0.5*dsmaxR[2]*(UR[0] - UL[0]);

    JR[1][0] = 0.5*(g*n[0]*UR[0] - UR[1]*termR) - 0.5*dsmaxR[0]*(UR[1] - UL[1]);
    JR[1][1] = n[0]*uR + 0.5*n[1]*vR - 0.5*smax - 0.5*dsmaxR[1]*(UR[1] - UL[1]);
    JR[1][2] = 0.5*n[1]*uR -0.5*dsmaxR[2]*(UR[1] - UL[1]);

    JR[2][0] = 0.5*(g*n[1]*UR[0]  - UR[2]*termR) - 0.5*dsmaxR[0]*(UR[2] - UL[2]);
    JR[2][1] = 0.5*n[0]*vR - 0.5*dsmaxR[1]*(UR[2] - UL[2]);
    JR[2][2] = n[1]*vR + 0.5*n[0]*uR - 0.5*smax - 0.5*dsmaxR[2]*(UR[2] - UL[2]);

    return JL,JR

def gaussianICS(x,y,z):
  nx,ny = np.shape(x)
  q = np.zeros((3,nx,ny))
  r = np.sqrt((x-1)**2 + (y-1)**2)
  q[0] = 1. + 0.125*np.exp( -(r)**2 )
  q[1] = 0.
  q[2] = 0.
  return q 

