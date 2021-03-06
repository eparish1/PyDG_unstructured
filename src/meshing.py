import numpy as np
import copy
from scipy.spatial import Delaunay
from myQuadRules import *
from basisFunctions import *
#given a tri class and zeta array of local quad points in zeta,eta coords,
#returns a global x and y array
def massMatrix(weights,ellq,tri):
  nbasis_functions = np.shape(ellq)[0]
  M = np.zeros((nbasis_functions,nbasis_functions))
  for i in range(0,nbasis_functions):
    for j in range(0,nbasis_functions):
      M[i,j] =  np.sum(weights[:]*ellq[i,:]*ellq[j,:],axis=0) 
  return M,np.linalg.inv(M)


def getXFromZeta(tri,zeta):
  verts = tri.points[tri.vertices].transpose()
  Ntris = tri.nsimplex#shape(tri.points[tri.vertices].transpose())[-1]
  dum,order = np.shape(zeta)
  xGlob = np.zeros((2,order,Ntris))
  for i in range(0,Ntris):
    xGlob[:,:,i] = np.dot(tri.J[:,:,i],zeta[:,:])# = tensordot(J,zeta,axes=([1],0))[:,:,0]
    xGlob[0,:,i] += verts[0,0,i]
    xGlob[1,:,i] += verts[1,0,i]
  return xGlob

def edgeToTri2BC(BE,flux_bc,triFlux):
  triInd = BE[:,2]
  edgeInd = BE[:,3]
  triFlux[:,:,edgeInd,triInd] = flux_bc[:,:,:]
  return triFlux

def edgeToTriForJac(tri,fluxL,fluxR,flux_bcL,flux_bcR):
  IE = tri.IE
  BE = tri.BE
  nedges = np.shape(IE)[0]
  edge_quads = np.shape(fluxL)[2]
  nvars = np.shape(fluxL)[0]
  Ntris = tri.nsimplex 

  triFlux = np.zeros((nvars,nvars,edge_quads,3,Ntris,3,Ntris))

  triInd = IE[:,2]
  edgeInd = IE[:,4]
  triInd2 = IE[:,3]
  edgeInd2 = IE[:,5]
  triFlux[:,:,:,edgeInd,triInd,edgeInd,triInd] = fluxL[:]
  triFlux[:,:,:,edgeInd,triInd,edgeInd2,triInd2] = fluxR[:,:]

  triFlux[:,:,:,edgeInd2,triInd2,edgeInd,triInd] = -fluxL[:,:,::-1]
  triFlux[:,:,:,edgeInd2,triInd2,edgeInd2,triInd2] = -fluxR[:,:,::-1]

  if not tri.periodic:
    triFlux[:,:,:,BE[:,3],BE[:,2],BE[:,3],BE[:,2]] = flux_bcL[:]
  return triFlux

def edgeToTriInterior(tri,fluxx,flux_bc):
  IE = tri.IE
  BE = tri.BE
  nedges = np.shape(IE)[0]
  edge_quads = np.shape(fluxx)[1]
  nvars = np.shape(fluxx)[0]
  Ntris = tri.nsimplex 

  triFlux = np.zeros((nvars,edge_quads,3,Ntris))
  triInd = IE[:,2]
  edgeInd = IE[:,4]
  triInd2 = IE[:,3]
  edgeInd2 = IE[:,5]
  triFlux[:,:,edgeInd,triInd] = fluxx[:,:,:]
#  if not tri.periodic:
#    triFlux[:,:,BE[:,3],BE[:,2]] = flux_bc[:]
  return triFlux

def edgeToTriExterior(tri,fluxx,flux_bc):
  IE = tri.IE
  BE = tri.BE
  nedges = np.shape(IE)[0]
  edge_quads = np.shape(fluxx)[1]
  nvars = np.shape(fluxx)[0]
  Ntris = tri.nsimplex 

  triFlux = np.zeros((nvars,edge_quads,3,Ntris))
  triInd = IE[:,2]
  edgeInd = IE[:,4]
  triInd2 = IE[:,3]
  edgeInd2 = IE[:,5]
  triFlux[:,:,edgeInd2,triInd2] = -fluxx[:,::-1,:]
  return triFlux


def edgeToTri2(tri,fluxx,flux_bc):
  IE = tri.IE
  BE = tri.BE
  nedges = np.shape(IE)[0]
  edge_quads = np.shape(fluxx)[1]
  nvars = np.shape(fluxx)[0]
  Ntris = tri.nsimplex 

  triFlux = np.zeros((nvars,edge_quads,3,Ntris))
  triInd = IE[:,2]
  edgeInd = IE[:,4]
  triInd2 = IE[:,3]
  edgeInd2 = IE[:,5]
  triFlux[:,:,edgeInd,triInd] = fluxx[:,:,:]
  triFlux[:,:,edgeInd2,triInd2] = -fluxx[:,::-1,:]

  if not tri.periodic:
    triFlux[:,:,BE[:,3],BE[:,2]] = flux_bc[:]
  return triFlux

'''
def edgeToTri(IE,fluxx):
  nedges = np.shape(IE)[0]
  edge_quads = np.shape(fluxx)[1]
  nvars = np.shape(fluxx)[0]
  Ntris = np.amax(IE) + 1
  triFlux = np.zeros((nvars,edge_quads,3,Ntris))
  triInd = IE[:,2]
  edgeInd = IE[:,4]
  triInd2 = IE[:,3]
  edgeInd2 = IE[:,5]
  triFlux[:,:,edgeInd,triInd] = fluxx[:,:,:]
  triFlux[:,:,edgeInd2,triInd2] = fluxx[:,::-1,:]
  return triFlux
'''

#same for a scalar
def edgeToTriScalar(tri,s):
  IE = tri.IE
  BE = tri.BE
  nedges = np.shape(IE)[0]
  Ntris = tri.nsimplex
  triFlux = np.zeros((3,Ntris))
  for i in range(0,nedges):
    triInd = IE[i,2]
    edgeInd = IE[i,4]
    triInd2 = IE[i,3]
    edgeInd2 = IE[i,5]
    triFlux[edgeInd,triInd] = s[i]
    triFlux[edgeInd2,triInd2] = s[i]
  if not tri.periodic:
    nbedges = np.shape(BE)[0]
    for i in range(0,nbedges):
      triInd = BE[i,2]
      edgeInd = BE[i,3]
      triFlux[edgeInd,triInd] = s[i + nedges]
  return triFlux

## function to compute normals of a triangular mesh, given a tri class from delauny
## function adds normals array to tri class
def computeNormals(tri):
  tri.normals = np.zeros((2,3,tri.nsimplex)) #3 edges on each triangle, x and y normal
  tri.normals[:,0] = np.flipud( tri.points[tri.vertices[:,1]].transpose() - tri.points[tri.vertices[:,0]].transpose() )
  tri.normals[:,1] = np.flipud( tri.points[tri.vertices[:,2]].transpose() - tri.points[tri.vertices[:,1]].transpose() )
  tri.normals[:,2] = np.flipud( tri.points[tri.vertices[:,0]].transpose() - tri.points[tri.vertices[:,2]].transpose() )
  tri.normals[1,:] = -tri.normals[1,:]
  normals_mag = np.sqrt(tri.normals[0,:]**2 + tri.normals[1,:]**2)
  tri.normals[0,:] = tri.normals[0,:]/normals_mag
  tri.normals[1,:] = tri.normals[1,:]/normals_mag

## get's the jacobian for a triangular mesh 
def getJacobian(tri):
  verts = tri.points[tri.vertices].transpose()
  Ntris = np.shape(tri.points[tri.vertices].transpose())[-1]
  tri.J = np.zeros((2,2,Ntris))
  tri.Jinv = np.zeros((2,2,Ntris))
  tri.Jdet = np.zeros((Ntris))
  tri.J[0,0] = verts[0,1] - verts[0,0]
  tri.J[0,1] = verts[0,2] - verts[0,0]
  tri.J[1,0] = verts[1,1] - verts[1,0]
  tri.J[1,1] = verts[1,2] - verts[1,0]
  for i in range(0,Ntris):
    tri.Jdet[i] = np.linalg.det(tri.J[:,:,i])
    tri.Jinv[:,:,i] = np.linalg.inv(tri.J[:,:,i])

## gets the Jacobians of the edges
def getEdgeJacobian(tri):
  nedges = np.shape(tri.IE)[0]
  if not tri.periodic:
    nbedges = np.shape(tri.BE)[0]
  else:
    nbedges = 0

  tri.Jedge = np.zeros(nedges+nbedges)
  for i in range(0,nedges):
    x1 = tri.points[tri.IE[i,0]][0]
    x2 = tri.points[tri.IE[i,1]][0]
    y1 = tri.points[tri.IE[i,0]][1]
    y2 = tri.points[tri.IE[i,1]][1]
    tri.Jedge[i] = np.sqrt( (x2 - x1)**2 + (y1 - y2)**2 ) / 2.
  if not tri.periodic:
    for i in range(0,nbedges):
      x1 = tri.points[tri.BE[i,0]][0]
      x2 = tri.points[tri.BE[i,1]][0]
      y1 = tri.points[tri.BE[i,0]][1]
      y2 = tri.points[tri.BE[i,1]][1]
      tri.Jedge[i + nedges] = np.sqrt( (x2 - x1)**2 + (y1 - y2)**2 ) / 2.




## return IE (interior edge) array with form
## IE = [vert1, vert2, el1, el2, el1_edge #, el2_edge #]
def edgeHash(tri,periodic=False):
  Nel = np.shape(tri.vertices)[0]
  H = np.zeros((tri.npoints,tri.npoints))
  H2 = np.zeros((tri.npoints,tri.npoints))
  IE = np.zeros((int(np.ceil(Nel*3./2.)),6),dtype='int32')
  niedge = 0
  for elem in range(0,Nel):
    for edge in range(0,3):
      n1 = tri.vertices[elem,edge%3]
      n2 = tri.vertices[elem,(edge+1)%3]
      if (H[n1,n2] == 0):
        H[n1,n2] = elem+1
        H[n2,n1] = elem+1
        H2[n1,n2] = edge
        H2[n2,n1] = edge
      else:
        oldelem = H[n1,n2] - 1
        oldedge = H2[n1,n2]
        if (oldelem < 0):
          print('Mesh input error')
        IE[niedge,:] = np.array([n1,n2,oldelem,elem,oldedge,edge])
        niedge = niedge + 1
        H[n1,n2] = -1
        H[n2,n1] = -1
  IE = IE[0:niedge,:]
  nbedge = 0
  BE = np.zeros((Nel*3,4),dtype='int32')
  for elem in range(0,Nel):
     for edge in range(0,3):
      n1 = tri.vertices[elem,edge%3]
      n2 = tri.vertices[elem,(edge+1)%3]
      if (H[n1,n2] > 0):
        BE[nbedge,:] = np.array([int(n1),int(n2),int(elem),edge])
        nbedge += 1
  BE = BE[0:nbedge,:]
  tri.BE = BE

  xedges = 0
  BEx = np.zeros((Nel*3,4),dtype='int32')
  BEy = np.zeros((Nel*3,4),dtype='int32')
  for i in range(0,nbedge):
    if ( abs(tri.points[BE[i,0]][0] - tri.points[BE[i,1]][0] )  <= 1e-9):
      BEx[xedges,:] = BE[i,:]
      xedges += 1

  yedges = 0
  for i in range(0,nbedge):
    if ( abs(tri.points[BE[i,0]][1] - tri.points[BE[i,1]][1] )  <= 1e-9):
      BEy[yedges,:] = BE[i,:]
      yedges += 1

  BEx = BEx[0:xedges,:]
  BEy = BEy[0:yedges,:]

  if periodic:
    ## Get paired edges and add to IE
    # for periodic, we assme that x or y nodal locations match 
    symedgesx = np.zeros((Nel*3,6),dtype='int32')
    counter = 0
    for i in range(0,xedges):
      for j in range(0,xedges):
        if ( abs( tri.points[BEx[i,0]][1] - tri.points[BEx[j,0]][1] ) < 1e-10 and \
             abs( tri.points[BEx[i,1]][1] - tri.points[BEx[j,1]][1] ) < 1e-10 ) or\
           ( abs( tri.points[BEx[i,1]][1] - tri.points[BEx[j,0]][1] ) < 1e-10 and \
             abs( tri.points[BEx[i,0]][1] - tri.points[BEx[j,1]][1] ) < 1e-10 ):
          if j != i:
            symedgesx[counter,:] = np.array([BEx[i,0],BEx[i,1],BEx[i,2],BEx[j,2],BEx[i,3],BEx[j,3]])
            counter += 1
    symedgesx = symedgesx[0:counter,:]
  
  
    symedgesy = np.zeros((Nel*3,6),dtype='int32')
    counter = 0
    for i in range(0,yedges):
      for j in range(0,yedges):
        if ( abs( tri.points[BEy[i,0]][0] - tri.points[BEy[j,0]][0] ) < 1e-10 and \
             abs( tri.points[BEy[i,1]][0] - tri.points[BEy[j,1]][0] ) < 1e-10 ) or\
           ( abs( tri.points[BEy[i,1]][0] - tri.points[BEy[j,0]][0] ) < 1e-10 and \
             abs( tri.points[BEy[i,0]][0] - tri.points[BEy[j,1]][0] ) < 1e-10 ):
          if j != i:
            symedgesy[counter,:] = np.array([BEy[i,0],BEy[i,1],BEy[i,2],BEy[j,2],BEy[i,3],BEy[j,3]])
            counter += 1
    symedgesy = symedgesy[0:counter,:]
  
    ## now delete repeated edges
    j = 0
    for i in range(0,np.shape(symedgesx)[0]):
      if (symedgesx[j,2] > symedgesx[j,3]):
        symedgesx = np.delete(symedgesx,j,axis=0)
        j = j-1
      j = j+1
  
    ## now delete repeated edges in y
    j = 0
    for i in range(0,np.shape(symedgesy)[0]):
      if (symedgesy[j,2] > symedgesy[j,3]):
        symedgesy = np.delete(symedgesy,j,axis=0)
        j = j-1
      j = j+1
  
    IE = np.append(IE,symedgesx,axis=0)
    IE = np.append(IE,symedgesy,axis=0)
    tri.BE = None

  tri.IE = np.zeros(np.shape(IE),dtype='int32')
  tri.IE[:] = IE[:]


class createGrid:
  def __init__(self,X,p,quad_order,periodic=False):
    tri = Delaunay(X.transpose())
    tri.periodic=periodic
    computeNormals(tri)
    ntris = tri.nsimplex
    getJacobian(tri)
    edgeHash(tri,periodic) 
    getEdgeJacobian(tri)
    tri.JedgeTri = edgeToTriScalar(tri,tri.Jedge)
    self.p = p
    self.quad_order = quad_order
    ## ============= Quadrature ===========================
    ## Now set up quadrature stuff
    triQuadPoints(quad_order,tri) #gets quadrature for enterior integration (tri.zetaq,tri.weights)
    gaussPoints(quad_order,tri) #gets quadrature points for edges (tri.s,tri.sweights)
    ## Now get basis functions.
    order_glob = int( (p+1)*(p+2)/2 )
    self.order_glob = order_glob
    getZetaLagrange(p,tri) #gets zeta,eta for a basic lagrange basis
    tri.ell,tri.ellzeta,tri.elleta = lagrangeBasis(p,tri.zeta) #gets basis functions and first derivs
    tri.ellq,tri.ellzetaq,tri.elletaq = lagrangeBasis(p,tri.zetaq) #gets basis functions with DOFs at the quad poitns
    zeta01 = np.array([tri.s/2.,tri.s*0])  #we also need basis functions with DOFs at the edges
    zeta12 = np.array([ np.flipud(tri.s)/2*np.cos(np.pi/4.)*np.sqrt(2.),tri.s/2.*np.cos(np.pi/4.)*np.sqrt(2.)])
    zeta20 = np.array([tri.s*0, 1. - tri.s/2])
    tri.ell_edges = np.zeros((int((p+1)*(p+2)/2),quad_order,3) )
    tri.ell_edges_zeta = np.zeros((int((p+1)*(p+2)/2),quad_order,3) )
    tri.ell_edges_eta = np.zeros((int((p+1)*(p+2)/2),quad_order,3) )
    tri.ell_edges[:,:,0],tri.ell_edges_zeta[:,:,0],tri.ell_edges_eta[:,:,0] = lagrangeBasis(p,zeta01)
    tri.ell_edges[:,:,1],tri.ell_edges_zeta[:,:,1],tri.ell_edges_eta[:,:,1] = lagrangeBasis(p,zeta12)
    tri.ell_edges[:,:,2],tri.ell_edges_zeta[:,:,2],tri.ell_edges_eta[:,:,2] = lagrangeBasis(p,zeta20)
    self.tri = tri
    M,Minv = massMatrix(tri.weights,tri.ellq,tri)
    Minv = 1./tri.Jdet[None,None,:]*Minv[:,:,None]
    M = M[:,:,None]*tri.Jdet[None,None,:]
    self.M = M
    self.Minv = Minv
    XQ = getXFromZeta(tri,tri.zeta)
    xtmp = XQ[0].flatten()
    ytmp = XQ[1].flatten()
    XQ = np.array([xtmp,ytmp])
    self.triQ = Delaunay(XQ.transpose())
    self.tri.Nelements = tri.nsimplex


class createHyperGrid:
  def __init__(self,grid,eqns,sampleElements):
    self.order_glob = grid.order_glob

    self.tri = copy.deepcopy(grid.tri)
    self.tri.periodic = grid.tri.periodic
    nInteriorEdges = np.shape(grid.tri.IE)[0]
    self.tri.IE = np.zeros( (0,np.shape(grid.tri.IE)[1] ) ,dtype='int')
    stencilElements = copy.deepcopy(sampleElements)
    for i in range(0,nInteriorEdges):
      triInd = grid.tri.IE[i,2]
      edgeInd = grid.tri.IE[i,4]
      triInd2 = grid.tri.IE[i,3]
      edgeInd2 = grid.tri.IE[i,5]     
      # check if edge touches a sample cell
      if (triInd in sampleElements or triInd2 in sampleElements):
        #if so, add edge to sample edge mesh, and add both touching cells to stencil mmesh
        self.tri.IE = np.append(self.tri.IE,grid.tri.IE[i,:][None,:],axis=0)
        stencilElements = np.append(stencilElements,triInd)
        stencilElements = np.append(stencilElements,triInd2)
        stencilElements = np.unique(stencilElements)

    
    if not self.tri.periodic:    
      nExteriorEdges = np.shape(grid.tri.BE)[0]
      self.tri.BE = np.zeros( (0,np.shape(grid.tri.BE)[1] ) ,dtype='int')
      for i in range(0,nExteriorEdges):
        triInd = grid.tri.BE[i,2]
        edgeInd = grid.tri.BE[i,3]
        # check if edge touches a sample cell
        if (triInd in sampleElements):
          #if so, add edge to sample edge mesh, and add both touching cells to stencil mmesh
          self.tri.BE = np.append(self.tri.BE,grid.tri.BE[i,:][None,:],axis=0)
          stencilElements = np.append(stencilElements,triInd)
          stencilElements = np.unique(stencilElements)
    else:
      nExteriorEdges = 0
      self.tri.BE = None
    ## tri IE and tri BE have global coordinates, not local coordinates
    ## now that the stencil mesh is built, we can convert to local coordinates
    ## such that indices run  from 0 to size(sampleElements)
    for i in range(0,np.shape(self.tri.IE)[0]):
      triGlobInd = self.tri.IE[i,2]
      triLocInd = np.where(stencilElements == triGlobInd)[0][0]
      self.tri.IE[i,2] = triLocInd
      triGlobInd2 = self.tri.IE[i,3]
      triLocInd2 = np.where(stencilElements == triGlobInd2)[0][0]
      self.tri.IE[i,3] = triLocInd2

    if not self.tri.periodic:    
      for i in range(0,np.shape(self.tri.BE)[0]):
        triGlobInd = self.tri.BE[i,2]
        triLocInd = np.where(stencilElements == triGlobInd)[0][0]
        self.tri.BE[i,2] = triLocInd

    self.tri.nsimplex = int( np.size(stencilElements) )
    self.tri.JedgeTri = grid.tri.JedgeTri[:,sampleElements]
    self.stencilElements = stencilElements 
    self.tri.Jinv = grid.tri.Jinv[:,:,sampleElements]
    self.tri.Jdet = grid.tri.Jdet[sampleElements]
    self.Minv = grid.Minv[:,:,sampleElements]
    ## Loop over all sample cells to determine where they are in the stencil mesh
    sampleElementsIds = np.zeros(0,dtype='int') 
    sampleElementsIdsForPressio = np.zeros(0,dtype='int') 
    dofsPerElement = eqns.nvars*grid.order_glob
    for i in range(0,np.size(sampleElements)):
      smGid = sampleElements[i]
      if (smGid in stencilElements):
        smLid = np.where(stencilElements == smGid)[0][0]
        sampleElementsIds = np.append(sampleElementsIds,smLid)
        localIds = range(smLid*dofsPerElement,(smLid+1)*dofsPerElement)
        sampleElementsIdsForPressio = np.append(sampleElementsIdsForPressio,localIds)


    self.tri.sampleElementsIds = sampleElementsIds
    self.sampleElementsIds = sampleElementsIds
    self.sampleElementsIdsForPressio = sampleElementsIdsForPressio

    self.sampleElements = sampleElements
    self.tri.normals = grid.tri.normals[:,:,stencilElements]
    self.tri.Nelements = np.size(sampleElements) 
