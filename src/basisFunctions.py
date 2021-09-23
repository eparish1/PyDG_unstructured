import numpy as np

# get local grid points for lagrange basis functions
def getZetaLagrange(p,tri):
  points = int( (p+1)*(p+2)/2 );
  if (p == 0):
    zetatmp = np.array([1./3.])
    etatmp = np.array([1./3.])
  else:
    zetatmp = np.linspace(0,1,p+1)
    etatmp = np.linspace(0,1,p+1)
  zeta = np.zeros((2,points))
  k = 0
  for j in range(0,p+1):
    for i in range(0,p+1-j):
      zeta[0,k] = zetatmp[i]
      zeta[1,k] = etatmp[j]
      k += 1
  tri.zeta = np.zeros(np.shape(zeta))
  tri.zeta[:] = zeta[:]
#  return zeta


def reconstructEdges(a,ell_edges):
  return np.einsum('pie,npk->niek',ell_edges,a)


def reconstructStateGradientAtEdges(a,tri):
  U_zeta = np.einsum('pie,npk->niek',tri.ell_edges_zeta,a)
  U_eta =  np.einsum('pie,npk->niek',tri.ell_edges_eta,a)
  Ux_edge =  U_zeta*tri.Jinv[None,None,None,0,0] + U_eta*tri.Jinv[None,None,None,1,0]
  Uy_edge =  U_zeta*tri.Jinv[None,None,None,0,1] + U_eta*tri.Jinv[None,None,None,1,1]
  return Ux_edge , Uy_edge

## function to go from the basis coefficients
## a_j to the quad points in ell
def reconstructU(a,ell):
  return np.einsum('pi,npk->nik',ell,a)

def reconstructStateGradient(a,tri,ellzeta,elleta):
  U_zeta = np.einsum('pi,npk->nik',ellzeta,a)
  U_eta = np.einsum('pi,npk->nik',elleta,a)
  Ux =  U_zeta*tri.Jinv[None,None,0,0] + U_eta*tri.Jinv[None,None,1,0]
  Uy =  U_zeta*tri.Jinv[None,None,0,1] + U_eta*tri.Jinv[None,None,1,1]
  return Ux,Uy

def lagrangeBasis(p,zeta):
  points = int( (p+1)*(p+2)/2 )
  quadpoints = int( np.size(zeta[0]) )
  ell  = np.zeros((points,quadpoints))
  ellzeta  = np.zeros((points,quadpoints))
  elleta  = np.zeros((points,quadpoints))
  if (p == 0):
    ell[0] = 1.
    ellzeta[0] = 0.
    ellzeta[0] = 0.
  if (p == 1):
    ell[0] = 1. - zeta[0] - zeta[1]
    ell[1] = zeta[0]
    ell[2] = zeta[1]
    ellzeta[0] = -1.
    ellzeta[1] = 1.
    ellzeta[2] = 0.
    elleta[0] = -1.
    elleta[1] = 0.
    elleta[2] = 1.
  if (p == 2):
    ell[0] = 2.*zeta[1]**2 + 4.*zeta[1]*zeta[0] - 3.*zeta[1] + 2.*zeta[0]**2 - 3.*zeta[0] + 1.
    ell[1] = 4.*zeta[0] - 4.*zeta[1]*zeta[0] - 4.*zeta[0]**2
    ell[2] = 2.*zeta[0]**2 - zeta[0]
    ell[3] = 4.*zeta[1] - 4.*zeta[1]*zeta[0] - 4.*zeta[1]**2
    ell[4] = 4.*zeta[1]*zeta[0]
    ell[5] = 2.*zeta[1]**2 - zeta[1]
    ellzeta[0] = 4.*zeta[1] + 4.*zeta[0] - 3.
    ellzeta[1] = 4. - 8.*zeta[0] - 4.*zeta[1]
    ellzeta[2] = 4.*zeta[0] - 1.
    ellzeta[3] = -4.*zeta[1]
    ellzeta[4] =  4.*zeta[1]
    ellzeta[5] = 0.
    elleta[0] = 4.*zeta[1] + 4.*zeta[0] - 3.
    elleta[1] = -4.*zeta[0]
    elleta[2] = 0.
    elleta[3] = 4. - 4.*zeta[0] - 8.*zeta[1]
    elleta[4] = 4.*zeta[0]
    elleta[5] = 4.*zeta[1] - 1.
  if (p == 3):
    ell[0] = - (9*zeta[1]**3)/2 - (27*zeta[1]**2*zeta[0])/2 + 9*zeta[1]**2 - (27*zeta[1]*zeta[0]**2)/2 + 18*zeta[1]*zeta[0] - (11*zeta[1])/2 - (9*zeta[0]**3)/2 + 9*zeta[0]**2 - (11*zeta[0])/2 + 1
    ell[1] = (27*zeta[1]**2*zeta[0])/2 + 27*zeta[1]*zeta[0]**2 - (45*zeta[1]*zeta[0])/2 + (27*zeta[0]**3)/2 - (45*zeta[0]**2)/2 + 9*zeta[0]
    ell[2] = (9*zeta[1]*zeta[0])/2 - (9*zeta[0])/2 - (27*zeta[1]*zeta[0]**2)/2 + 18*zeta[0]**2 - (27*zeta[0]**3)/2
    ell[3] = (9*zeta[0]**3)/2 - (9*zeta[0]**2)/2 + zeta[0]
    ell[4] = (27*zeta[1]**3)/2 + 27*zeta[1]**2*zeta[0] - (45*zeta[1]**2)/2 + (27*zeta[1]*zeta[0]**2)/2 - (45*zeta[1]*zeta[0])/2 + 9*zeta[1]
    ell[5] = - 27*zeta[1]**2*zeta[0] - 27*zeta[1]*zeta[0]**2 + 27*zeta[1]*zeta[0]
    ell[6] = (27*zeta[1]*zeta[0]**2)/2 - (9*zeta[1]*zeta[0])/2
    ell[7] = (9*zeta[1]*zeta[0])/2 - (9*zeta[1])/2 - (27*zeta[1]**2*zeta[0])/2 + 18*zeta[1]**2 - (27*zeta[1]**3)/2
    ell[8] = (27*zeta[0]*zeta[1]**2)/2 - (9*zeta[0]*zeta[1])/2
    ell[9] = zeta[1] - (9*zeta[1]**2)/2 + (9*zeta[1]**3)/2
    ellzeta[0] =  18*zeta[1] + 18*zeta[0] - 27*zeta[1]*zeta[0] - (27.*zeta[1]**2)/2. - (27*zeta[0]**2.)/2. - 11./2.
    ellzeta[1] =  (27.*zeta[1]**2.)/2. + 54*zeta[1]*zeta[0] - (45.*zeta[1])/2. + (81.*zeta[0]**2)/2. - 45.*zeta[0] + 9
    ellzeta[2] =  (9.*zeta[1])/2. + 36.*zeta[0] - 27.*zeta[1]*zeta[0] - (81.*zeta[0]**2.)/2. - 9./2.
    ellzeta[3] =  (27.*zeta[0]**2.)/2. - 9.*zeta[0] + 1.
    ellzeta[4] =  27.*zeta[1]*zeta[0] - (45.*zeta[1])/2. + 27.*zeta[1]**2.
    ellzeta[5] =  27.*zeta[1] - 54.*zeta[1]*zeta[0] - 27.*zeta[1]**2.
    ellzeta[6] =  27*zeta[1]*zeta[0] - (9.*zeta[1])/2.
    ellzeta[7] =  (9.*zeta[1])/2. - (27.*zeta[1]**2.)/2.
    ellzeta[8] =  (27.*zeta[1]**2.)/2. - (9.*zeta[1])/2.
    ellzeta[9] = 0.
    elleta[0] = 18.*zeta[1] + 18*zeta[0] - 27.*zeta[1]*zeta[0] - (27.*zeta[1]**2.)/2. - (27.*zeta[0]**2.)/2. - 11./2.
    elleta[1] = 27*zeta[1]*zeta[0] - (45.*zeta[0])/2. + 27.*zeta[0]**2.
    elleta[2] = (9.*zeta[0])/2. - (27.*zeta[0]**2.)/2.
    elleta[3] = 0.
    elleta[4] = (81*zeta[1]**2)/2 + 54.*zeta[1]*zeta[0] - 45.*zeta[1] + (27.*zeta[0]**2.)/2. - (45.*zeta[0])/2. + 9.
    elleta[5] = 27*zeta[0] - 54*zeta[1]*zeta[0] - 27.*zeta[0]**2.
    elleta[6] = (27.*zeta[0]**2.)/2. - (9.*zeta[0])/2
    elleta[7] = 36.*zeta[1] + (9.*zeta[0])/2. - 27.*zeta[1]*zeta[0] - (81.*zeta[1]**2.)/2. - 9./2.
    elleta[8] = 27.*zeta[1]*zeta[0] - (9.*zeta[0])/2.
    elleta[9] = (27.*zeta[1]**2.)/2. - 9.*zeta[1] + 1
  return ell,ellzeta,elleta

