import numpy as np
from scipy.spatial import Delaunay
from evtk.hl import pointsToVTK 
from meshing import *
from myQuadRules import *
from basisFunctions import *
import time

def computeFluxHyper(tri,u_edges,eqns):
  u1 = u_edges[:,:,tri.IE[:,4],tri.IE[:,2]]
  u2 = u_edges[:,::-1,tri.IE[:,5],tri.IE[:,3]]
  normals = tri.normals[:,tri.IE[:,4],tri.IE[:,2]]
  flux = eqns.inviscidFlux(u1,u2,normals)
  return flux


def computeFluxResidHyper(U,tri,eqns):
  nbasis_functions = np.shape(tri.ell_edges)[0] 
  u_edges = reconstructEdges(U,tri.ell_edges) ##reconstruct u at the edges
  edge_flux = computeFluxHyper(tri,u_edges,eqns) ##get fluxes at edges
  triFlux = edgeToTri2(tri.IE,edge_flux) #same for y flux
  edgeResid = np.zeros((eqns.nvars,nbasis_functions,tri.nsimplex))
  for i in range(0,nbasis_functions):
    edgeResid[:,i] = np.sum( integrateEdge(triFlux*tri.ell_edges[None,i,:,:,None],tri) ,axis=1) ## compute integral on each edge and sum over all edges
  return edgeResid

def computeVolResidHyper(U,tri,eqns):
  nbasis_functions = np.shape(tri.ell)[0] #get number of basis functions (# of equations)
  UQ = reconstructU(U,tri.ellq) #constuct U at the quadrature points
  f1,f2 = eqns.flux(UQ) #evaluate the x and y fluxes
  volResid = np.zeros((eqns.nvars,nbasis_functions,tri.nsimplex)) #create array for the volume residual
  for i in range(0,nbasis_functions):
    ## differentiate in local coodinates
    volResid[:,i] = integrateTri(f1*(tri.ellzetaq[None,i,:,None]*tri.Jinv[None,0,0] + tri.elletaq[None,i,:,None]*tri.Jinv[None,1,0]) + f2*(tri.ellzetaq[None,i,:,None]*tri.Jinv[None,0,1] + tri.elletaq[None,i,:,None]*tri.Jinv[None,1,1]) ,tri)
  return volResid 

def computeRHSHyper(U,tri,eqns):
  fluxResid = computeFluxResidHyper(U,tri,eqns) #get inviscid flux contribution to residual
  volResid = computeVolResidHyper(U,tri,eqns) #get interior volume contribution to residual
  RHS = -fluxResid + volResid
  return RHS



