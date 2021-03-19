import numpy as np
from scipy.spatial import Delaunay
from evtk.hl import pointsToVTK 
from meshing import *
from myQuadRules import *
from basisFunctions import *
import time

def computeFlux(tri,u_edges,eqns):
  u1 = u_edges[:,:,tri.IE[:,4],tri.IE[:,2]]
  u2 = u_edges[:,::-1,tri.IE[:,5],tri.IE[:,3]]
  normals = tri.normals[:,tri.IE[:,4],tri.IE[:,2]]
  flux = eqns.inviscidFlux(u1,u2,normals)
  return flux


def computeFluxResid(U,tri,eqns):
  nbasis_functions = np.shape(tri.ell_edges)[0] 
  u_edges = reconstructEdges(U,tri.ell_edges) ##reconstruct u at the edges
  edge_flux = computeFlux(tri,u_edges,eqns) ##get fluxes at edges
  triFlux = edgeToTri2(tri.IE,edge_flux) #same for y flux
  edgeResid = np.zeros((eqns.nvars,nbasis_functions,tri.nsimplex))
  for i in range(0,nbasis_functions):
    edgeResid[:,i] = np.sum( integrateEdge(triFlux*tri.ell_edges[None,i,:,:,None],tri) ,axis=1) ## compute integral on each edge and sum over all edges
  return edgeResid

def computeVolResid(U,tri,eqns):
  nbasis_functions = np.shape(tri.ell)[0] #get number of basis functions (# of equations)
  UQ = reconstructU(U,tri.ellq) #constuct U at the quadrature points
  f1,f2 = eqns.flux(UQ) #evaluate the x and y fluxes
  volResid = np.zeros((eqns.nvars,nbasis_functions,tri.nsimplex)) #create array for the volume residual
  for i in range(0,nbasis_functions):
    ## differentiate in local coodinates
    volResid[:,i] = integrateTri(f1*(tri.ellzetaq[None,i,:,None]*tri.Jinv[None,0,0] + tri.elletaq[None,i,:,None]*tri.Jinv[None,1,0]) + f2*(tri.ellzetaq[None,i,:,None]*tri.Jinv[None,0,1] + tri.elletaq[None,i,:,None]*tri.Jinv[None,1,1]) ,tri)
  return volResid 

def computeRHS(U,tri,eqns):
  fluxResid = computeFluxResid(U,tri,eqns) #get inviscid flux contribution to residual
  volResid = computeVolResid(U,tri,eqns) #get interior volume contribution to residual
  RHS = -fluxResid + volResid
  return RHS


def constructUIC_ell2Projection(verts,J,p,zeta,tri,IC_function):
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

def constructUIC_collocation(verts,J,p,zeta,IC_function):
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


