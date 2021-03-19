import numpy as np
try:
  from postProcessCore import *
  canWriteToVtk = True
except:
  print('Error importing postproocessing core, cannot use VTK')
  canWriteToVtk = False
class shallowWaterEquations:
  nvars = 3
  def writeSol(self,string,U,grid):
    if canWriteToVtk:
      triangle_faces_to_VTK(string,x=grid.triQ.points[:,0],y=grid.triQ.points[:,1],z=grid.triQ.points[:,0]*0,faces = grid.triQ.vertices,point_data=None,\
               cell_data={'h':np.mean(U[0].flatten()[grid.triQ.vertices[:,:]],axis=1),'hU':np.mean(U[1].flatten()[grid.triQ.vertices[:,:]],axis=1),\
               'hV':np.mean(U[2].flatten()[grid.triQ.vertices[:,:]],axis=1)})
    np.savez(string,x=grid.triQ.points[:,0],y=grid.triQ.points[:,1],U=U)

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

def gaussianICS(x,y,z):
  nx,ny = np.shape(x)
  q = np.zeros((3,nx,ny))
  r = np.sqrt((x-1)**2 + (y-1)**2)
  q[0] = 1. + 0.125*np.exp( -(r)**2 )
  q[1] = 0.
  q[2] = 0.
  return q 

