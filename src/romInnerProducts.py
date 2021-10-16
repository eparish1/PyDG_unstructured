import numpy as np
import sys

def conservative_to_entropy(U):
  gamma = 1.4
  gamma_bar = gamma - 1.
  p = (gamma - 1.)*(U[3] - 0.5*U[1]**2/U[0] - 0.5*U[2]**2/U[0] )

  V = np.zeros(np.shape(U))
  s = np.log(p) - gamma*np.log(U[0])
  V[0] = -s/(gamma - 1.) + (gamma + 1.)/(gamma - 1.) - U[3]/p
  V[1] = U[1]/p
  V[2] = U[2]/p
  V[3] = -U[0]/p
  return V

def entropy_to_conservative(V):
  gamma = 1.4
  U = np.zeros(np.shape(V))
  gamma1 = gamma - 1.
  igamma1 = 1./gamma1
  gmogm1 = gamma*igamma1
  iu3 = 1./V[3]  #- p / rho
  u = -iu3*V[1]
  v = -iu3*V[2]
  t0 = -0.5*iu3*(V[1]**2 + V[2]**2)
  t1 = V[0] - gmogm1 + t0
  t2 =np.exp(-igamma1*np.log(-V[3]) )
  t3 = np.exp(t1)
  U[0] = t2*t3
  H = -iu3*(gmogm1 + t0)
  E = (H + iu3)
  U[1] = U[0]*u
  U[2] = U[0]*v
  U[3] = U[0]*E
  return U


class createEnergyBasisVariable:
  def __init__(self,Phi):
    self.qEnergy = Phi #of size K x N 

  def reconstructState(self,xhat):
    return np.dot(self.qEnergy.transpose(),xhat)


class createEntropyBasisVariable:
  def __init__(self,Phi):
    self.qEntropy = Phi #of size K x N 

  def reconstructState(self,xhat):
    return np.dot(self.qEntropy.transpose(),xhat)

class createL2BasisVariable:
  def __init__(self,Phi):
    self.qConserved = Phi 

  def reconstructState(self,xhat):
    return np.dot(self.qConserved.transpose(),xhat)

class createStateVariable:
  def __init__(self,U):
    self.qConserved = U
    self.qEnergy = U*1.
    self.qEnergy[:,-1] -= 0.5/U[:,0]*(U[:,1]**2 + U[:,2]**2)
    self.qEnergy /= (np.sqrt(U[:,0])[:,None])

    self.qEntropy = np.zeros(np.shape(U))
    gamma = 1.4
    gamma_bar = gamma - 1.
    p = (gamma - 1.)*(U[:,3] - 0.5*U[:,1]**2/U[:,0] - 0.5*U[:,2]**2/U[:,0] )
    s = np.log(p) - gamma*np.log(U[:,0])

    self.qEntropy[:,0] = -s/(gamma - 1.) + (gamma + 1.)/(gamma - 1.) - U[:,3]/p
    self.qEntropy[:,1] = U[:,1]/p
    self.qEntropy[:,2] = U[:,2]/p
    self.qEntropy[:,3] = -U[:,0]/p

    if (np.size(np.shape(U)) == 4):
      nSamples,nVars,nInTri,nElements = np.shape(U)
      self.nVars,self.nInTri,self.nElements = nVars,nInTri,nElements
      self.nSpace = nVars*nInTri*nElements
    else:
      nVars,nInTri,nElements = np.shape(U)
      self.nVars,self.nInTri,self.nElements = nVars,nInTri,nElements
      self.nSpace = nVars*nInTri*nElements

  def flattenVariable(self,U):
    if (np.size(np.shape(U)) == 4):
      nSamples = np.shape(U)[0]
      return np.reshape(U,(nSamples,self.nSpace))
    else:
      return U.flatten()

  def unflattenVariable(self,U):
    if (np.size(np.shape(U)) == 2):
      nSamples = np.shape(U)[0]
      return np.reshape(U,(nSamples,self.nVars,self.nInTri,self.nElements))
    else:
      return np.reshape(U,(self.nVars,self.nInTri,self.nElements))



class createEntropyInnerProductSpace:
  def __init__(self,U,M,Uref):
    nSamples,nVars,nInTri,nElements = np.shape(U)
    self.nVars,self.nInTri,self.nElements = nVars,nInTri,nElements
    self.nSpace = nVars*nInTri*nElements
    self.M = M

    gamma = 1.4
    gamma_bar = gamma - 1.
    p = (gamma - 1.)*(Uref[3] - 0.5*Uref[1]**2/Uref[0] - 0.5*Uref[2]**2/Uref[0] )

    V = np.zeros(4)
    s = np.log(p) - gamma*np.log(Uref[0])
    V[0] = -s/(gamma - 1.) + (gamma + 1.)/(gamma - 1.) - Uref[3]/p
    V[1] = Uref[1]/p
    V[2] = Uref[2]/p
    V[3] = -Uref[0]/p

    es = 1.e-30
    H = (Uref[3] + p) / Uref[0]
    asqr = gamma*p/Uref[0]
    A0 = np.zeros((4,4))
    A0[0,:] = Uref[:]
    A0[1,0] = A0[0,1]
    A0[1,1] = Uref[1]**2/Uref[0] + p#Uref[1]*(1./(V[1]+es) - V[1]/V[4])
    A0[1,2] = -Uref[1]*V[2]/V[3]
    A0[1,3] = -Uref[1]/V[3] - V[1]*Uref[3]/V[3]
    A0[2,0] = A0[0,2]
    A0[2,1] = A0[1,2]
    A0[2,2] = Uref[2]**2/Uref[0] + p
    A0[2,3] = -Uref[2]/V[3] - V[2]*Uref[3]/V[3]
    A0[3,0] = A0[0,3]
    A0[3,1] = A0[1,3]
    A0[3,2] = A0[2,3]
    A0[3,3] = Uref[0]*H**2 - asqr*p/(gamma - 1.)
    self.A0 = A0
    self.A0sqrt = np.linalg.cholesky(self.A0).transpose()
    self.A0sqrtInv = np.linalg.inv(self.A0sqrt)
    self.Msqrt =    np.rollaxis(np.rollaxis(np.linalg.cholesky(np.rollaxis(M,2,0)),2,1),0,3)
    self.MsqrtInv = np.rollaxis(np.linalg.inv(np.rollaxis(np.linalg.cholesky(np.rollaxis(M,2,0)),2,1)),0,3) 

    self.Vref = V

  def innerProduct(self,U,V):
    MV = np.einsum('ijk,...jk->...ik',self.M,V)
    WMV = np.einsum('ij,kj...->ki...',self.A0,MV)
    nSamplesV = np.shape(WMV)[0]
    nSamplesU = np.shape(U)[0]
    WMV = np.reshape(WMV,(nSamplesV,self.nSpace))
    U = np.reshape(U,(nSamplesU,self.nSpace))
    result = np.dot(U,WMV.transpose())
    return result

  def innerProductNoMass(self,U,V):
    WV = np.einsum('ij,kj...->ki...',self.A0,V)
    nSamplesV = np.shape(WV)[0]
    nSamplesU = np.shape(U)[0]
    WV = np.reshape(WV,(nSamplesV,self.nSpace))
    U = np.reshape(U,(nSamplesU,self.nSpace))
    result = np.dot(U,WV.transpose())
    return result


  def makeBasisViaPod(self,stateVariable,K=0,tol=0):
    if (K  == 0 and tol == 0):
      print('Error, must specify either K or tol')
      sys.exit()
    if (K  != 0 and tol != 0):
      print('Error, can only specify one of K and tol')
      sys.exit()

    Kern = self.innerProduct(stateVariable.qEntropy,stateVariable.qEntropy)
    self.Kern = Kern
    Lam,E = np.linalg.eig(Kern)
    sigma = np.sqrt(Lam)
    self.sigma = sigma
    self.Lam = Lam

    rel_energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    Phi = np.real( np.dot(stateVariable.flattenVariable(stateVariable.qEntropy).transpose(), np.dot(E , np.diag(1./sigma)) ))
    if (K == 0):
      K = np.size(rel_energy[rel_energy < tol]) + 1
      entropyBasis = createEntropyBasisVariable( Phi[:,0:K].transpose() ) 
    else:
      entropyBasis = createEntropyBasisVariable( Phi[:,0:K].transpose() ) 
    return entropyBasis

  def makeBasisViaGeneralizedSvd(self,stateVariable,K=0,tol=0):
    if (K  == 0 and tol == 0):
      print('Error, must specify either K or tol')
      sys.exit()
    if (K  != 0 and tol != 0):
      print('Error, can only specify one of K and tol')
      sys.exit()

    MQ = np.einsum('ijk,...jk->...ik',self.Msqrt,stateVariable.qEntropy)
    AMQ = np.einsum('ij,nj...->ni...',self.A0sqrt,MQ)
    nSamplesU = np.shape(AMQ)[0]
    AMQ = np.reshape(AMQ,(nSamplesU,self.nSpace))
    U,sigma,_ = np.linalg.svd(AMQ.transpose(),full_matrices=False)
    U = U[:,0:K]
    U = stateVariable.unflattenVariable(U.transpose())
    Phi = np.einsum('ij,nj...->ni...',self.A0sqrtInv,U)
    Phi = np.einsum('ijk,...jk->...ik',self.MsqrtInv,Phi)
    self.sigma = sigma
    rel_energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    if (K == 0):
      K = np.size(rel_energy[rel_energy < tol]) + 1
      entropyBasis = createEntropyBasisVariable( stateVariable.flattenVariable(Phi[:,:] )) 
    else:
      entropyBasis = createEntropyBasisVariable( stateVariable.flattenVariable(Phi[:,:] )) 
    return entropyBasis





class createEnergyInnerProductSpace:
  def __init__(self,U,M):
    nSamples,nVars,nInTri,nElements = np.shape(U)
    self.nVars,self.nInTri,self.nElements = nVars,nInTri,nElements
    self.nSpace = nVars*nInTri*nElements
    self.M = M
    self.W = np.zeros((4,4))
    self.W[0,-1] = 1.
    self.W[1,1] = 1.
    self.W[2,2] = 1.
    self.W[3,0] = 1.
    self.W *= 0.5

  def innerProduct(self,U,V):
    MV = np.einsum('ijk,...jk->...ik',self.M,V)
    WMV = np.einsum('ij,kj...->ki...',self.W,MV)
    nSamplesV = np.shape(WMV)[0]
    nSamplesU = np.shape(U)[0]
    WMV = np.reshape(WMV,(nSamplesV,self.nSpace))
    U = np.reshape(U,(nSamplesU,self.nSpace))
    result = np.dot(U,WMV.transpose())
    return result

  def innerProductNoMass(self,U,V):
    WV = np.einsum('ij,kj...->ki...',self.W,V)
    nSamplesV = np.shape(WV)[0]
    nSamplesU = np.shape(U)[0]
    WV = np.reshape(WV,(nSamplesV,self.nSpace))
    U = np.reshape(U,(nSamplesU,self.nSpace))
    result = np.dot(U,WV.transpose())
    return result


  def makeBasisViaPod(self,stateVariable,K=0,tol=0):
    if (K  == 0 and tol == 0):
      print('Error, must specify either K or tol')
      sys.exit()
    if (K  != 0 and tol != 0):
      print('Error, can only specify one of K and tol')
      sys.exit()

    Kern = self.innerProduct(stateVariable.qEnergy,stateVariable.qEnergy)
    Lam,E = np.linalg.eig(Kern)
    print(Lam)
    sigma = np.sqrt(Lam)
    self.sigma = sigma
    self.Lam = Lam
    rel_energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    Phi = np.real( np.dot(stateVariable.flattenVariable(stateVariable.qEnergy).transpose(), np.dot(E , np.diag(1./sigma)) ))
    if (K == 0):
      K = np.size(rel_energy[rel_energy < tol]) + 1
      energyBasis = createEnergyBasisVariable( Phi[:,0:K].transpose() ) 
    else:
      energyBasis = createEnergyBasisVariable( Phi[:,0:K].transpose() ) 
    return energyBasis

class createL2InnerProductSpace:
  def __init__(self,U,M):
    nSamples,nVars,nInTri,nElements = np.shape(U)
    self.nVars,self.nInTri,self.nElements = nVars,nInTri,nElements
    self.nSpace = nVars*nInTri*nElements
    self.M = M
    self.Msqrt =    np.rollaxis(np.rollaxis(np.linalg.cholesky(np.rollaxis(M,2,0)),2,1),0,3)
    self.MsqrtInv = np.rollaxis(np.linalg.inv(np.rollaxis(np.linalg.cholesky(np.rollaxis(M,2,0)),2,1)),0,3) 

  def innerProduct(self,U,V):
    MV = np.einsum('ijk,...jk->...ik',self.M,V)
    nSamplesV = np.shape(MV)[0]
    nSamplesU = np.shape(U)[0]
    MV = np.reshape(MV,(nSamplesV,self.nSpace))
    U = np.reshape(U,(nSamplesU,self.nSpace))
    result = np.dot(U,MV.transpose())
    return result

  def innerProductNoMass(self,U,V):
    nSamplesV = np.shape(V)[0]
    nSamplesU = np.shape(U)[0]
    V = np.reshape(V,(nSamplesV,self.nSpace))
    U = np.reshape(U,(nSamplesU,self.nSpace))
    result = np.dot(U,V.transpose())
    return result


  def makeBasisViaPod(self,stateVariable,K=0,tol=0):
    if (K  == 0 and tol == 0):
      print('Error, must specify either K or tol')
      sys.exit()
    if (K  != 0 and tol != 0):
      print('Error, can only specify one of K and tol')
      sys.exit()
    Kern = self.innerProduct(stateVariable.qConserved,stateVariable.qConserved)
    Lam,E = np.linalg.eig(Kern)
    sigma = np.sqrt(Lam)
    rel_energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    Phi = np.real( np.dot(stateVariable.flattenVariable(stateVariable.qConserved).transpose(), np.dot(E , np.diag(1./sigma)) ))
    if (K == 0):
      K = np.size(rel_energy[rel_energy < tol]) + 1
      L2Basis = createL2BasisVariable( Phi[:,0:K].transpose() ) 
    else:
      L2Basis = createL2BasisVariable( Phi[:,0:K].transpose() ) 
    return L2Basis



  def makeBasisViaGeneralizedSvd(self,stateVariable,K=0,tol=0):
    if (K  == 0 and tol == 0):
      print('Error, must specify either K or tol')
      sys.exit()
    if (K  != 0 and tol != 0):
      print('Error, can only specify one of K and tol')
      sys.exit()

    MQ = np.einsum('ijk,...jk->...ik',self.Msqrt,stateVariable.qConserved)
    nSamplesU = np.shape(MQ)[0]
    MQ = np.reshape(MQ,(nSamplesU,self.nSpace))
    U,sigma,_ = np.linalg.svd(MQ.transpose(),full_matrices=False)
    U = U[:,0:K]
    U = stateVariable.unflattenVariable(U.transpose())
    Phi = np.einsum('ijk,...jk->...ik',self.MsqrtInv,U)
    self.sigma = sigma
    rel_energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    if (K == 0):
      K = np.size(rel_energy[rel_energy < tol]) + 1
      L2Basis = createL2BasisVariable( stateVariable.flattenVariable(Phi[:,:] )) 
    else:
      L2Basis = createL2BasisVariable( stateVariable.flattenVariable(Phi[:,:] )) 
    return L2Basis 



class createNonDimensionalL2InnerProductSpace:
  def __init__(self,U,M,rhoInf,uInf):
    #
    self.weighting = np.zeros(4)
    self.weighting[0] = 1./rhoInf
    self.weighting[1] = 1./(rhoInf*uInf)
    self.weighting[2] = 1./(rhoInf*uInf)
    self.weighting[3] = 1./(rhoInf*uInf**2)
    nSamples,nVars,nInTri,nElements = np.shape(U)
    self.nVars,self.nInTri,self.nElements = nVars,nInTri,nElements
    self.nSpace = nVars*nInTri*nElements
    self.M = M
    self.Msqrt =    np.rollaxis(np.rollaxis(np.linalg.cholesky(np.rollaxis(M,2,0)),2,1),0,3)
    self.MsqrtInv = np.rollaxis(np.linalg.inv(np.rollaxis(np.linalg.cholesky(np.rollaxis(M,2,0)),2,1)),0,3) 

  def innerProduct(self,U,V):
    WV = self.weighting[None,:,None,None]**2 * V
    MWV = np.einsum('ijk,...jk->...ik',self.M,WV)
    nSamplesV = np.shape(MWV)[0]
    nSamplesU = np.shape(U)[0]
    MWV = np.reshape(MWV,(nSamplesV,self.nSpace))
    U = np.reshape(U,(nSamplesU,self.nSpace))
    result = np.dot(U,MWV.transpose())
    return result

  def innerProductNoMass(self,U,V):
    WV = self.weighting[None,:,None,None]**2*V
    nSamplesV = np.shape(V)[0]
    nSamplesU = np.shape(U)[0]
    WV = np.reshape(WV,(nSamplesV,self.nSpace))
    U = np.reshape(U,(nSamplesU,self.nSpace))
    result = np.dot(U,WV.transpose())
    return result


  def makeBasisViaPod(self,stateVariable,K=0,tol=0):
    if (K  == 0 and tol == 0):
      print('Error, must specify either K or tol')
      sys.exit()
    if (K  != 0 and tol != 0):
      print('Error, can only specify one of K and tol')
      sys.exit()
    Kern = self.innerProduct(stateVariable.qConserved,stateVariable.qConserved)
    Lam,E = np.linalg.eig(Kern)
    sigma = np.sqrt(Lam)
    rel_energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    Phi = np.real( np.dot(stateVariable.flattenVariable(stateVariable.qConserved).transpose(), np.dot(E , np.diag(1./sigma)) ))
    if (K == 0):
      K = np.size(rel_energy[rel_energy < tol]) + 1
      L2Basis = createL2BasisVariable( Phi[:,0:K].transpose() ) 
    else:
      L2Basis = createL2BasisVariable( Phi[:,0:K].transpose() ) 
    return L2Basis

  def makeBasisViaGeneralizedSvd(self,stateVariable,K=0,tol=0):
    if (K  == 0 and tol == 0):
      print('Error, must specify either K or tol')
      sys.exit()
    if (K  != 0 and tol != 0):
      print('Error, can only specify one of K and tol')
      sys.exit()

    
    MQ = np.einsum('ijk,...jk->...ik',self.Msqrt,stateVariable.qConserved*self.weighting[None,:,None,None])
    nSamplesU = np.shape(MQ)[0]
    MQ = np.reshape(MQ,(nSamplesU,self.nSpace))
    U,sigma,_ = np.linalg.svd(MQ.transpose(),full_matrices=False)
    U = U[:,0:K]
    U = stateVariable.unflattenVariable(U.transpose())
    Phi = np.einsum('ijk,...jk->...ik',self.MsqrtInv,U)
    Phi = Phi*(1./ self.weighting[None,:,None,None] )
    self.sigma = sigma
    rel_energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    if (K == 0):
      K = np.size(rel_energy[rel_energy < tol]) + 1
      L2Basis = createL2BasisVariable( stateVariable.flattenVariable(Phi[:,:] )) 
    else:
      L2Basis = createL2BasisVariable( stateVariable.flattenVariable(Phi[:,:] )) 
    return L2Basis 


