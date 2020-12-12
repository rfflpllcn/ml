"""
the marcenko-pastur distribution vs the empirical pdf of a random correlation with signal

- The eigenvalues and eigenvectors of a random correlation matrix are calculated.
- Using the Kernel Density Estimate algorithm a kernel of the eigenvalues is estimated.
- the empirical KDE is plotted against the Marcenko-Pastur pdf

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_histogram
from src.pca import getPCA
from src.marcenko_pastur import mpPDF, findMaxEval
from src.pdf import fitKDE
from src.risk import getRndCov, cov2corr

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

alpha, nCols, nFact, q = .995, 1000, 100, 10
cov = np.cov(np.random.normal(size=(nCols * q, nCols)), rowvar=0)
cov = alpha * cov + (1 - alpha) * getRndCov(nCols, nFact)  # noise+signal
corr0 = cov2corr(cov)
eVal0, eVec0 = getPCA(corr0)

eMax0, var0 = findMaxEval(np.diag(eVal0), q, .01)
nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)

print(nFacts0)
pdf0 = mpPDF(1., q=q, pts=nCols)
# pdf1 = np.histogram(np.diag(eVal0), bins=100)

# pdf1 = fitKDE(np.diag(eVal0), bWidth=.01)  # empirical pdf


plt.figure()
plt.plot(list(pdf0.index), pdf0, 'b')
plt.hist(np.diag(eVal0), density=True, bins=100)
plt.legend(['Marcenko-Pastur', 'Empirical'])
plt.ylabel('prob[\lambda]')
plt.xlabel('\lambda')
plt.show()
