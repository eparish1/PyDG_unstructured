import numpy as np
data = np.load('npsol0.npz')
N = np.size(data['U'])
snapshots = np.zeros((N,0))
for i in range(0,1000,5):
  print('Loading snapshot ' + str(i))
  data = np.load('npsol' + str(i) + '.npz')
  snapshots = np.append(snapshots,data['U'].flatten()[:,None],axis=1)


U,s,_ = np.linalg.svd(snapshots,full_matrices=False)
rel_e = np.cumsum(s**2)/np.sum(s**2)
K = np.size(rel_e[rel_e<0.99999])
Phi = U[:,0:K]

np.savez('pod_basis',Phi=Phi,rel_e=rel_e)
