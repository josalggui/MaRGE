import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

input = 'Vol163/RAREprotocols_T1_SAG_Left.2023.10.13.17.15.10.332.mat'
rawData = sio.loadmat(input)
img3D = rawData['imgReconTyger2D_Red'][0]
print(img3D.shape)
print(np.max(img3D))

plt.figure()
plt.imshow(img3D[0,:,:])
plt.show()