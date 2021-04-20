import numpy as np
from scipy.spatial import Delaunay
from meshing import *
from myQuadRules import *
from basisFunctions import *
import time

def computeFlux(tri,u_edges,eqns):
  u1 = u_edges[:,:,tri.IE[:,4],tri.IE[:,2]]
  u2 = u_edges[:,::-1,tri.IE[:,5],tri.IE[:,3]]
  normals = tri.normals[:,tri.IE[:,4],tri.IE[:,2]]
  flux = eqns.inviscidFlux(u1,u2,normals)

  if not tri.periodic:
    uBC = u_edges[:,:,tri.BE[:,3],tri.BE[:,2] ]
    uBC2 = eqns.getBoundaryStateFromInteriorState(tri,uBC)
    normalsBC = tri.normals[:,tri.BE[:,3],tri.BE[:,2]]
    flux_bc = eqns.inviscidFlux(uBC,uBC2,normalsBC)
  else:
    flux_bc = None
  return flux,flux_bc

def computeFluxJac(tri,u_edges,eqns):
  u1 = u_edges[:,:,tri.IE[:,4],tri.IE[:,2]]
  u2 = u_edges[:,::-1,tri.IE[:,5],tri.IE[:,3]]
  normals = tri.normals[:,tri.IE[:,4],tri.IE[:,2]]
  fluxJacL,fluxJacR = eqns.inviscidFluxJac(u1,u2,normals)

  if not tri.periodic:
    uBC = u_edges[:,:,tri.BE[:,3],tri.BE[:,2] ]
    uBC2 = eqns.getBoundaryStateFromInteriorState(tri,uBC)
    normalsBC = tri.normals[:,tri.BE[:,3],tri.BE[:,2]]
    flux_bcJacL,flux_bcJacR = eqns.inviscidFluxJac(uBC,uBC2,normalsBC)
  else:
    flux_bcJacL,flux_bcJacR = None,None
  return fluxJacL,fluxJacR,flux_bcJacL,flux_bcJacR


def computeFluxResid(U,tri,eqns):
  nbasis_functions = np.shape(tri.ell_edges)[0] 
  u_edges = reconstructEdges(U,tri.ell_edges) ##reconstruct u at the edges
  edge_flux,edge_flux_bc = computeFlux(tri,u_edges,eqns) ##get fluxes at edges
  triFlux = edgeToTri2(tri,edge_flux,edge_flux_bc) #same for y flux
  tmp = triFlux[:,None]*tri.ell_edges[None,:,:,:,None]
  edgeResid = np.sum( tri.JedgeTri[None,None,:,:]*np.einsum('j,...jkl->...kl',tri.sweights,tmp) , axis=2)
  return edgeResid

def computeVolResid(U,tri,eqns):
  nbasis_functions = np.shape(tri.ell)[0] #get number of basis functions (# of equations)
  UQ = reconstructU(U,tri.ellq) #constuct U at the quadrature points
  f1,f2 = eqns.flux(UQ)  #evaluate the x and y fluxes
  # compute volume residual
  tmp = f1[:,None]*(tri.ellzetaq[None,:,:,None]*tri.Jinv[None,None,0,0] + tri.elletaq[None,:,:,None]*tri.Jinv[None,None,1,0]) + \
        f2[:,None]*(tri.ellzetaq[None,:,:,None]*tri.Jinv[None,None,0,1] + tri.elletaq[None,:,:,None]*tri.Jinv[None,None,1,1])
  volResid  = np.einsum('j,...jk->...k',tri.weights,tmp)*tri.Jdet[None,None] #integrate over the edges
  return volResid 

def computeVolResidJac(U,tri,eqns):
  nbasis_functions = np.shape(tri.ell)[0] #get number of basis functions (# of equations)
  vol_jac = np.zeros((eqns.nvars,eqns.nvars,nbasis_functions,nbasis_functions,tri.nsimplex),dtype=U.dtype)
  UQ = reconstructU(U,tri.ellq) #constuct U at the quadrature points
  f1,f2 = eqns.fluxJac(UQ) #evaluate the x and y fluxes
  tmp = f1[:,:,None,None]*(tri.ellzetaq[None,None,:,None,:,None]*tri.ellq[None,None,None,:,:,None]*tri.Jinv[None,None,None,None,0,0] + \
                               tri.elletaq[None,None,:,None,:,None]*tri.ellq[None,None,None,:,:,None]*tri.Jinv[None,None,None,None,1,0]) + \
        f2[:,:,None,None]*(tri.ellzetaq[None,None,:,None,:,None]*tri.ellq[None,None,None,:,:,None]*tri.Jinv[None,None,None,None,0,1] +\
                               tri.elletaq[None,None,:,None,:,None]*tri.ellq[None,None,None,:,:,None]*tri.Jinv[None,None,None,None,1,1])
  vol_jac[:,:,:,:,:] = np.einsum('j,...jk->...k',tri.weights,tmp)*tri.Jdet[None,None,None,None]
  return vol_jac

def computeFluxResidJac(U,tri,eqns):
  nbasis_functions = np.shape(tri.ell_edges)[0] 
  u_edges = reconstructEdges(U,tri.ell_edges) ##reconstruct u at the edges
  efL,efR,ebfL,ebfR = computeFluxJac(tri,u_edges,eqns) ##get fluxes at edges
  edgeResidJac = np.zeros((eqns.nvars,eqns.nvars,nbasis_functions,tri.nsimplex,nbasis_functions,tri.nsimplex))

  tmp = efL[:,:,None,None]*(tri.ell_edges[:,:,tri.IE[:,4]][None,None,:,None,:])*(tri.ell_edges[:,:,tri.IE[:,4]][None,None,None,:,:])
  tmp = tri.JedgeTri[None,None,None,tri.IE[:,4],tri.IE[:,2]]*np.einsum('j,...ijl->...il',tri.sweights,tmp)
  for i in range(0,nbasis_functions):
    for j in range(0,nbasis_functions):
      for k in range(0,eqns.nvars):
        for l in range(0,eqns.nvars):
           np.add.at(edgeResidJac[l,k,i,:,j],(tri.IE[:,2],tri.IE[:,2]),tmp[l,k,j,i]) 

  tmp = -efR[:,:,None,None,::-1]*(tri.ell_edges[:,:,tri.IE[:,5]][None,None,:,None,:])*(tri.ell_edges[:,:,tri.IE[:,5]][None,None,None,:,:])
  tmp = tri.JedgeTri[None,None,None,tri.IE[:,5],tri.IE[:,3]]*np.einsum('j,...ijl->...il',tri.sweights,tmp)
  for i in range(0,nbasis_functions):
    for j in range(0,nbasis_functions):
      for k in range(0,eqns.nvars):
        for l in range(0,eqns.nvars):
           np.add.at(edgeResidJac[l,k,i,:,j],(tri.IE[:,3],tri.IE[:,3]),tmp[l,k,j,i]) 


  tmp = efR[:,:,None,None]*(tri.ell_edges[:,:,tri.IE[:,4]][None,None,:,None,:])*(tri.ell_edges[:,:,tri.IE[:,5]][None,None,None,:,::-1]) 
  tmp = tri.JedgeTri[None,None,None,tri.IE[:,4],tri.IE[:,2]]*np.einsum('j,...ijl->...il',tri.sweights,tmp)
  tmp = np.rollaxis(np.rollaxis(tmp,3,5),3,0)
  edgeResidJac[:,:,:,tri.IE[:,2],:,tri.IE[:,3]] += tmp[:]


  tmp = -efL[:,:,None,None,::-1]*(tri.ell_edges[:,:,tri.IE[:,5]][None,None,:,None,:])*(tri.ell_edges[:,:,tri.IE[:,4]][None,None,None,:,::-1]) 
  tmp = tri.JedgeTri[None,None,None,tri.IE[:,5],tri.IE[:,3]]*np.einsum('j,...ijl->...il',tri.sweights,tmp)
  tmp = np.rollaxis(np.rollaxis(tmp,3,5),3,0)
  edgeResidJac[:,:,:,tri.IE[:,3],:,tri.IE[:,2]] += tmp[:]
        #if not tri.periodic:
        #  tmp = tri.JedgeTri[None,tri.BE[:,3],tri.BE[:,2]]*np.einsum('j,ijl->il',tri.sweights,ebfL[:,k]*(tri.ell_edges[:,:,tri.BE[:,3]][None,i,:])*(tri.ell_edges[:,:,tri.BE[:,3]][None,j,:]))
        #  i1 = range(0,np.shape(tmp)[0])
        #  for l in range(0,eqns.nvars):
        #    np.add.at(edgeResidJac[l,k,i,:,j],(tri.BE[:,2],tri.BE[:,2]),tmp[l]) 
        # 
        #  tmp = tri.JedgeTri[None,tri.BE[:,3],tri.BE[:,2]]*np.einsum('j,ijl->il',tri.sweights,ebfR[:,k]*(tri.ell_edges[:,:,tri.BE[:,3]][None,i,:])*(tri.ell_edges[:,:,tri.BE[:,3]][None,j,:]))
        #  i1 = range(0,np.shape(tmp)[0])
        #  for l in range(0,eqns.nvars):
        #    np.add.at(edgeResidJac[l,k,i,:,j],(tri.BE[:,2],tri.BE[:,2]),tmp[l]) 


  return edgeResidJac

 
def computeRHS(U,tri,eqns):
  fluxResid = computeFluxResid(U,tri,eqns) #get inviscid flux contribution to residual
  volResid = computeVolResid(U,tri,eqns) #get interior volume contribution to residual
  RHS = -fluxResid + volResid
  return RHS


def constructUIC_ell2Projection(grid,IC_function):
  verts = grid.tri.points[grid.tri.vertices].transpose()
  J = grid.tri.J
  p = grid.p
  zeta = grid.tri.zeta
  tri = grid.tri
  order,quadpoints = np.shape(tri.ellq)
  Ntris = np.shape(tri.points[tri.vertices].transpose())[-1]
  zetaLoc = np.zeros((2,quadpoints,Ntris))
  xGlob = np.zeros((2,order,Ntris))
  XQ = getXFromZeta(tri,tri.zetaq)
  UQIC = IC_function(XQ[0],XQ[1],0)
  nvars = np.shape(UQIC)[0]
  b = np.zeros((nvars,order,Ntris))
  M,Minv = massMatrix(tri.weights,tri.ellq,tri)
  Minv = 1./tri.Jdet[None,None,:]*Minv[:,:,None]
  for i in range(0,order):
    b[:,i] = integrateTri(tri.ellq[i][None,:,None]*UQIC,tri)
  U = np.einsum('ijk,njk->nik',Minv,b)
  xGlob = getXFromZeta(tri,tri.zeta)
  return U,xGlob

def constructUIC_collocation(grid,IC_function):
  verts = grid.tri.points[grid.tri.vertices].transpose()
  J = grid.tri.J
  p = grid.p
  zeta = grid.tri.zeta
  dum,order = np.shape(zeta)
  Ntris = np.shape(tri.points[tri.vertices].transpose())[-1]
  zetaLoc = np.zeros((2,order,Ntris))
  xGlob = np.zeros((2,order,Ntris))
  for i in range(0,Ntris):
    xGlob[:,:,i] = dot(J[:,:,i],zeta[:,:])
    xGlob[0,:,i] += verts[0,0,i]
    xGlob[1,:,i] += verts[1,0,i]
  u = IC_function(xGlob[0],xGlob[1],0)
  return u,xGlob


