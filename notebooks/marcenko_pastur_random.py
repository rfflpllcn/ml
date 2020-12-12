"""
the marcenko-pastur distribution vs the empirical pdf of a random correlation

- The eigenvalues and eigenvectors of a random correlation matrix are calculated.
- Using the Kernel Density Estimate algorithm a kernel of the eigenvalues is estimated.
- the empirical KDE is plotted against the Marcenko-Pastur pdf

"""

import numpy as np
import matplotlib.pyplot as plt

from src.pca import getPCA
from src.marcenko_pastur import mpPDF
from src.pdf import fitKDE

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

T, N = 10000, 1000
x = np.random.normal(size=(T, N))
eVal0, eVec0 = getPCA(np.corrcoef(x, rowvar=0))
pdf0 = mpPDF(1., q=x.shape[0] / float(x.shape[1]), pts=N)
pdf1 = fitKDE(np.diag(eVal0), bWidth=.01)  # empirical pdf

plt.figure()
plt.plot(np.diag(eVal0), pdf0, 'b', np.diag(eVal0), pdf1, 'y--')
plt.legend(['Marcenko-Pastur', 'Empirical:KDE'])
plt.ylabel('prob[\lambda]')
plt.xlabel('\lambda')
plt.show()
