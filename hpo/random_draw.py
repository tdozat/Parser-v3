import numpy as np
import numpy.linalg as la
import scipy.stats as sps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 200
d = 3
B = sps.truncnorm.rvs(0,1,loc=.25, scale=.5,size=[100,d])
D = B
for i in range(1, n+1):
  accepted = False
  added = False
  attempts = 0
  best = None
  best_p = 0
  while not accepted:
    b = np.random.rand(d)
    if not added:
      D = np.concatenate([D, b.reshape([1,-1])])
      added = True
    C = B
    mins = []
    maxes = []
    for j in range(d):
      drange = 1/np.power(i+1, 1/d)
      lower_bound = b[j]-.5*drange
      upper_bound = b[j]+.5*drange
      dmin = max(0, lower_bound - max(0, upper_bound-1))
      dmax = min(1, upper_bound - min(0, lower_bound))
      mins.append(dmin)
      maxes.append(dmax)
      C = C[np.where(np.greater(C[:,j], mins[-1]) * np.less(C[:,j], maxes[-1]))]
    n = len(C)
    if n == 0:
      accepted = True
    else:
      mins = np.array(mins)
      maxes = np.array(maxes)
      volume = np.prod(maxes-mins)
      lamda = (i+1)*volume
      p = sps.poisson.sf(n, lamda)
      accepted = sps.bernoulli.rvs(p)
      if not accepted:
        attempts += 1
        print('{:d}) Vol: {:.2f}\tP: {:.2e}\tAttempts: {:d}\tStatus: {}'.format(i, volume*(i+1), p, attempts, 'REJECTED'))
        if p > best_p:
          best = b
          best_p = best_p
        if attempts == 15:
          accepted = True
          b = best
  B = np.concatenate([B, b.reshape([1,-1])])

fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(121, projection='3d')
ax.scatter(D[:,0], D[:,1], D[:,2])
ax = fig.add_subplot(122, projection='3d')
ax.scatter(B[:,0], B[:,1], B[:,2])
plt.show()



#n = 100
#d = 2
#A = np.random.uniform(-1,1, size=[n,d])
#B = A[:1]
#D = A[:1]
#for i in range(1,n+1):
#  b = np.random.uniform(-1,1, size=d)
#  C = np.array(B)
#  D = np.concatenate([D, b.reshape([1,-1])])
#  for j in range(d):
#    c1 = np.array(b)
#    c1[j] = 1
#    c2 = np.array(b)
#    c2[j] = -1
#    C = np.concatenate([C, c1.reshape([1,-1]), c2.reshape([1,-1])])
#  C_centered = C - b
#  dists = np.sqrt(np.sum(C_centered**2, axis=1))
#  C_norm = C_centered / dists.reshape([-1,1])
#  current = b
#  bests = [C[np.argmin(dists)]]
#  C_norm[np.argmin(dists)] = 0
#  for j in range(d):
#    current = np.mean(bests, axis=0)
#    current_centered = current - b
#    current_norm = current_centered / np.sqrt(current_centered.dot(current_centered)+1e-12)
#    coses = C_norm.dot(current_norm)
#    where = np.where(C_norm.dot(current_norm) < 0)
#    #fig, ax = plt.subplots(1,1)
#    #ax.scatter([b[0]], [b[1]], label='random point', alpha=.5)
#    #ax.scatter([current[0]], [current[1]], label='current', alpha=.5)
#    #ax.scatter(C[where][:,0], C[where][:,1], label='possible nexts', alpha=.5)
#    #ax.set_xlim([-1,1])
#    #ax.set_ylim([-1,1])
#    #ax.legend()
#    #plt.show()
#    bests.append(C[where][np.argmin(dists[where])])
#    C_norm[where[0][np.argmin(dists[where])]] = 0
#  best = np.mean(bests, axis=0)
#  #best = np.mean([best, b], axis=0)
#  B = np.concatenate([B, best.reshape([1,-1])])
#  
#fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
#ax1.scatter(B[:,0], B[:,1], alpha=.5)
#ax1.set_xlim([-1,1])
#ax1.set_ylim([-1,1])
#ax2.scatter(D[:,0], D[:,1], alpha=.5)
#ax2.set_xlim([-1,1])
#ax2.set_ylim([-1,1])
#plt.show()
  
  
#  BTB = B.T.dot(B)
#  centered_B = B - np.mean(B, axis=0, keepdims=True)
#  BTB = centered_B.T.dot(centered_B)
#  for j in range(100):
#    bbT = np.outer(b,b)
#    bTb = np.dot(b,b)
#    base = (bbT + BTB - I*(i-1)/3)
#    loss = np.sum(base**2)/2
#    grad = base.dot(b)
#    hess = base + I*bTb + (1-I)*bbT
#    # clean up hessian
#    evals, evecs = la.eig(hess)
#    evals = np.abs(evals)
#    hess_ = (evecs*evals).dot(evecs.T)
#    #if j % 100 == 0:
#    #  print(j, loss)
#    b -= .1*la.inv(hess_).dot(grad)
#    #b -= .2*grad
#    b = np.clip(b, -1,1)
#  B = np.concatenate([B, b.reshape([1,-1])])
#
#print(np.cov(A, rowvar=False))
#print(np.cov(B, rowvar=False))
#
