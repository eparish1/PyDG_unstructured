from pylab import *
from basisFunctions import *
import numpy as np
try:
  from postProcessCore import *
  canWriteToVtk = True
except:
  print('Error importing postproocessing core, cannot use VTK')
  canWriteToVtk = False 
class diffusionEquation:
  nvars = 1
  def __init__(self,bc_type=None,bc_function=None,source=None,params=None):
    self.computeSource = source
    self.params = params
    self.bc_type = bc_type
    self.viscosity = 1. 
    if (bc_type == "CUSTOM_BCS"):
      self.getBoundaryStateFromInteriorState = bc_function
    else:
      self.getBoundaryStateFromInteriorState = None


  def writeSol(self,string,U,grid):
    Ux,Uy = reconstructStateGradient(U,grid.tri,grid.tri.ellzetaq,grid.tri.elletaq)
    if canWriteToVtk:
      triangle_faces_to_VTK(string,x=grid.triQ.points[:,0],y=grid.triQ.points[:,1],z=grid.triQ.points[:,0]*0,faces = grid.triQ.vertices,point_data=None,\
               cell_data={'U':np.mean(U[0].flatten()[grid.triQ.vertices[:,:]],axis=1),'Ux':np.mean(Ux[0].flatten()[grid.triQ.vertices[:,:]],axis=1),'Uy':np.mean(Uy[0].flatten()[grid.triQ.vertices[:,:]],axis=1)})
    np.savez(string,x=grid.triQ.points[:,0],y=grid.triQ.points[:,1],U=U)

  def flux(self,u):
    fx = np.zeros(np.shape(u))
    fy = np.zeros(np.shape(u))
    return fx,fy
  

  def inviscidFlux(self,UL,UR,n):
    F = np.zeros(np.shape(UL))  # for allocation
    return F


  def centralFlux(self,UL,UR,n):
    F = 0.5*(UL + UR)
    return F


  def viscousFlux(self,U,Ux,Uy):
    fx = np.zeros(np.shape(U),dtype=U.dtype)
    fx[0] = self.viscosity*Ux[0]
    fy = np.zeros(np.shape(U),dtype=U.dtype)
    fy[0] = self.viscosity*Uy[0]
    return fx,fy


  def getGs_X(u,main,mu,V):
    nvars = np.shape(u)[0]
    fvG11 = np.zeros(np.shape(u))
    fvG21 = np.zeros(np.shape(u))
    fvG11[0] = mu*V[0]
    return fvG11,fvG21

  def getGs_Y(u,main,mu,V):
    nvars = np.shape(u)[0]
    fvG12 = np.zeros(np.shape(u))
    fvG22 = np.zeros(np.shape(u))
    fvG22[0] = mu*V[0]
    return fvG12,fvG22

