import numpy as np
try:
  from postProcessCore import *
  canWriteToVtk = True
except:
  print('Error importing postproocessing core, cannot use VTK')
  canWriteToVtk = False 
class eulerEquations:
  nvars = 4
  def writeSol(self,string,U,grid):
    if canWriteToVtk:
      p = (1.4 - 1.)*(U[3] - 0.5*U[1]**2/U[0] - 0.5*U[2]**2/U[0])
      triangle_faces_to_VTK(string,x=grid.triQ.points[:,0],y=grid.triQ.points[:,1],z=grid.triQ.points[:,0]*0,faces = grid.triQ.vertices,point_data=None,\
               cell_data={'rho':np.mean(U[0].flatten()[grid.triQ.vertices[:,:]],axis=1),'rhoU':np.mean(U[1].flatten()[grid.triQ.vertices[:,:]],axis=1),\
               'rhoV':np.mean(U[2].flatten()[grid.triQ.vertices[:,:]],axis=1),'rhoE':np.mean(U[3].flatten()[grid.triQ.vertices[:,:]],axis=1),\
               'p':np.mean(p.flatten()[grid.triQ.vertices[:,:]],axis=1)})
    np.savez(string,x=grid.triQ.points[:,0],y=grid.triQ.points[:,1],U=U)

  def flux(self,u):
    fx = np.zeros(np.shape(u))
    fy = np.zeros(np.shape(u))

    gamma = 1.4
    p = (gamma - 1.)*(u[3] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0])
    fx[0] = u[1]
    fx[1] = u[1]**2/u[0] + p
    fx[2] = u[1]*u[2]/u[0]
    fx[3] = (u[3] + p)*u[1]/u[0]

    fy[0] = u[2]
    fy[1] = u[1]*u[2]/u[0]
    fy[2] = u[2]**2/u[0] + p
    fy[3] = (u[3] + p)*u[2]/u[0]
    return fx,fy
  

  def inviscidFlux(self,UL,UR,n):
  # PURPOSE: This function calculates the flux for the Euler equations
  # using the Roe flux function
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
    Ntris = np.shape(UL)[-1]
    gamma = 1.4
    gmi = gamma-1.0
    #process left state
    rL = UL[0]
    uL = UL[1]/rL
    vL = UL[2]/rL
    unL = uL*n[0] + vL*n[1]
    qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2])/rL
    pL = (gamma-1)*(UL[3] - 0.5*rL*qL**2.)
    rHL = UL[3] + pL
    HL = rHL/rL
    cL = np.sqrt(gamma*pL/rL)
    # left flux
    FL = np.zeros(np.shape(UL))
    FL[0] = rL*unL
    FL[1] = UL[1]*unL + pL*n[0]
    FL[2] = UL[2]*unL + pL*n[1]
    FL[3] = rHL*unL
  
    # process right state
    rR = UR[0]
    uR = UR[1]/rR
    vR = UR[2]/rR
    unR = uR*n[0] + vR*n[1]
    qR = np.sqrt(UR[1]**2. + UR[2]**2.)/rR
    pR = (gamma-1)*(UR[3] - 0.5*rR*qR**2.)
    rHR = UR[3] + pR
    HR = rHR/rR
    cR = np.sqrt(gamma*pR/rR)
    # right flux
    FR = np.zeros(np.shape(UR))
    FR[0] = rR*unR
    FR[1] = UR[1]*unR + pR*n[0]
    FR[2] = UR[2]*unR + pR*n[1]
    FR[3] = rHR*unR

    # difference in states
    du = UR - UL
  
    # Roe average
    di     = np.sqrt(rR/rL)
    d1     = 1.0/(1.0+di)
  
    ui     = (di*uR + uL)*d1
    vi     = (di*vR + vL)*d1
    Hi     = (di*HR + HL)*d1
  
    af     = 0.5*(ui*ui+vi*vi )
    ucp    = ui*n[0] + vi*n[1]
    c2     = gmi*(Hi - af)
    ci     = np.sqrt(c2)
    ci1    = 1.0/ci
   
    #% eigenvalues
  
    sh = np.shape(ucp)
    lsh = np.append(3,sh)
    l = np.zeros(lsh)
    l[0] = ucp+ci
    l[1] = ucp-ci
    l[2] = ucp
  
    #% entropy fix
    epsilon = ci*.1
    #print(l,epsilon)
    labs = np.abs(l)
    for i in range(0,3):
      l[i,labs[i]<epsilon] =  (epsilon[labs[i]<epsilon] + l[i,labs[i]<epsilon]**2)/(2.*epsilon[labs[i]<epsilon])
  #    l[i] = np.maximum(
  #  if ((l[i]<epsilon) and (l[i]>-epsilon)):
  #      l[i] = 0.5*(epsilon + l[i]*l[i]/epsilon)
  
    l = np.abs(l)
    l3 = l[2]
    # average and half-difference of 1st and 2nd eigs
    s1    = 0.5*(l[0] + l[1])
    s2    = 0.5*(l[0] - l[1])
  
    # left eigenvector product generators (see Theory guide)
    G1    = gmi*(af*du[0] - ui*du[1] - vi*du[2] + du[3])
    G2    = -ucp*du[0]+du[1]*n[0]+du[2]*n[1]
  
    # required functions of G1 and G2 (again, see Theory guide)
    C1    = G1*(s1-l3)*ci1*ci1 + G2*s2*ci1
    C2    = G1*s2*ci1          + G2*(s1-l3)
   
    # flux assembly
    F = np.zeros(np.shape(FL))  # for allocation
    F[0]    = 0.5*(FL[0]+FR[0])-0.5*(l3*du[0] + C1   )
    F[1]    = 0.5*(FL[1]+FR[1])-0.5*(l3*du[1] + C1*ui + C2*n[0])
    F[2]    = 0.5*(FL[2]+FR[2])-0.5*(l3*du[2] + C1*vi + C2*n[1])
    F[3]    = 0.5*(FL[3]+FR[3])-0.5*(l3*du[3] + C1*Hi + C2*ucp  )

    #% max wave speed
    smag = np.amax(l)
    return F


###Premade initial conditions for the euler equations
def vortexICS(x,y,t):
  nx,ny = np.shape(x)
  #hw conditions
  uinf,vinf = 1./np.sqrt(2.),1./np.sqrt(2.)
  rc = 1.
  x0,y0 = 0.,0.
  rhoinf = 1.
  Minf = 0.5
  gamma = 1.4
  eps = 0.3
  t = 0.
  #derived quantities
  Vmag = np.sqrt(uinf**2 + vinf**2)
  cinf = Vmag/Minf
  pinf = cinf**2*rhoinf/gamma
  f0x = x - x0 - uinf*t
  f0y = y - y0 - vinf*t
  r = np.sqrt(f0x**2 + f0y**2)
  f0 = 1. - r**2/rc**2
  f1 = 1. - eps**2*(gamma - 1.)*Minf**2*np.exp(f0)/(8.*np.pi**2)
  f2 = eps*Vmag/(2.*np.pi*rc)*np.exp(f0/2.)
  rho = rhoinf*f1**(1./(gamma - 1.))
  u = uinf - f2*(y - y0 - vinf*t)
  v = vinf + f2*(x - x0 - uinf*t)
  p = pinf*f1**(gamma/(gamma - 1.))
  q = np.zeros((4,nx,ny))
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = p/(gamma - 1.) + 0.5*rho*(u**2 + v**2)
  return q

