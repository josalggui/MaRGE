import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

input = '44L.mat'
rawData = sio.loadmat(input)
img3D = rawData['imgReconTyger2D'][0]
print(img3D.shape)
print(np.max(img3D))

plt.figure()
plt.imshow(img3D[0,:,:])
plt.show()