import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

input = '/home/teresa/Tyger_Nov24/scripts/44L.mat'
rawData = sio.loadmat(input)
img3D = rawData['imgReconTyger'][0]
print(img3D.shape)
print(np.max(img3D))

plt.figure(figsize = (5,5), dpi=240)
gs1 = gridspec.GridSpec(5,5)
gs1.update(wspace=0.020, hspace=0.020) # set the spacing between axes.

for i in range(24):
    ii = i + 2
    if i > img3D.shape[0]:
        break
    ax1 = plt.subplot(gs1[i])
    plt.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    imgAux = img3D[int(ii),:,:]
    ax1.imshow(imgAux,cmap='gray')

plt.show()