#Subject, Experiment y Scan se llaman como el nombre entero de la secuencia
#Si un metadata se encuentra vacío [], no se guarda ese field por defecto
#Desde el XML en XNAT se pueden ver los fields
#BW se ha cambiado a bw, XNAT no emplea bien las mayúsculas parece ser

#Cada archivo le cuesta 26 segundos para subirse

# -*- coding: utf-8 -*-
import xnat
import os
import scipy.io
from pyxnat import Interface
import imageio
import nibabel as nib
import numpy as np
from pylab import imsave
import shutil
import sys

#Si queremos añadir FECHA
#subject.demographics.date = '2019-01-01'

#Conexión XNAT
session = xnat.connect('http://localhost', user='admin', password='admin')
interface = Interface(server='http://localhost', user='admin', password='admin') #Para SNAPSHOTS

#PROYECTO tiene que existir en XNAT
project = session.projects['physioMRI']
project_snapshot = interface.select.project('physioMRI').insert() #Para SNAPSHOTS

#Seteamos el directorio en la carpeta de datos
os.chdir("/home/physioMRI/git_repos/PhysioMRI_GUI/experiments/acquisitions/2021.10.13/2021.10.13.21.20.33") #SELECCIONAR CARPETA DONDE ESTÉN LOS ARCHIVOS


#1-Accedemos a los files y guardamos sus nombres
files = os.listdir()

for i in files:
    if i.find('nii') != -1:
        nii = i
    elif i.find('mat') != -1:
        mat = i 
    
    #Si no encuentra .nii o .mat se termina el código
    if i.find('nii') == -1 and i.find('mat') == -1:
        print("Falta el archivo .mat o .nii") #################################MESSAGE FUNCTION#####################################
        sys.exit()

#Eliminamos .nii y .mat de los nombres   
filename = nii.split('.nii')[0]
filename = filename.replace(".", "-")

s_label = filename
e_label = filename

#2-subject
subject = session.classes.SubjectData(parent=project, label=s_label)

#3-experiment
experiment = session.classes.MrSessionData(parent=subject, label=filename)

#4-scan
scan = session.classes.MrScanData(parent=experiment, id=filename, type='MRI')

#5-Resource y subir NIFTI
#Actualizamos el NIFTI si ya existe
try:
    #XNATResponseError (status 409)
    resource_NIFTI = session.classes.ResourceCatalog(parent=scan, label='NIFTI', type='nii')
    resource_NIFTI.upload(nii, nii)
except:
    resource_NIFTI = scan.resources['NIFTI']
    resource_NIFTI.upload(nii, nii)
    
#6-Resource y subir .mat   
try:
    #XNATResponseError (status 409)
    resource_MAT = session.classes.ResourceCatalog(parent=scan, label='MAT', type='mat')
    resource_MAT.upload(mat, mat)
except:
    resource_MAT = scan.resources['MAT']
    resource_MAT.upload(mat, mat)

#7-METADATA
metadata = scipy.io.loadmat(mat)

metadata["bw"] = metadata.pop("BW")

for key in metadata.keys():
    experiment.fields[key] = metadata.get(key)

# =============================================================================
# SNAPSHOTS
# =============================================================================
NIFTI_img = nib.load(nii)
data = NIFTI_img.get_fdata()

#Creamos directorio para guardar imágenes dentro de cada carpeta y será destruido 
#Comprobamos si existe el directorio por error y se borra para crearlo de nuevo
if os.path.isdir("SNAPSHOTS") == False:
    os.mkdir("SNAPSHOTS")
    os.chdir("SNAPSHOTS")
else:
    shutil.rmtree("SNAPSHOTS")
    os.mkdir("SNAPSHOTS")
    os.chdir("SNAPSHOTS")


if data.ndim != 3:
    print("Se espera un número de 3 dimensiones") #################################MESSAGE FUNCTION#####################################
    sys.exit()

#Dimensión donde se encuentra 10
shape=np.array(data.shape)
for i in list(range(data.ndim)):
    if shape[i] == min(shape):
        dim_10 = i
    

images = []

for i in list(range(9)): #El número de slices siempre es 10
    if dim_10 == 0:
        data_1 = np.array(data[i,:,:])
    elif dim_10 == 1: 
        data_1 = np.array(data[:,i,:])
    elif dim_10 == 2: 
        data_1 = np.array(data[:,:,i])
        
    imsave(str(i) + '.gif', data_1)
    
    images.append(imageio.imread(str(i) + '.gif'))

imageio.mimsave(filename + '.gif', images)

#Subir snapshot   
subject_snapshot = project_snapshot.subject(filename)

experiment = subject_snapshot.experiment(filename)

experiment.scan(filename).resource('SNAPSHOTS').file(filename + '.gif').insert(filename + '.gif',
                                                     format="GIF",
                                                     content = "ORIGINAL",
                                                     tags = None)

experiment.scan(filename).resource('SNAPSHOTS').file(filename + '_t.gif').insert(filename + '.gif',
                                                     format="GIF",
                                                     content = "THUMBNAIL",
                                                     tags = None)

os.chdir("../")
shutil.rmtree("SNAPSHOTS")    
# =============================================================================
# FIN SNAPSHOTS
# =============================================================================

#Volvemos a directorios
os.chdir("../")

#elapsed_time_fl = (time.time() - start)
#print(elapsed_time_fl)


session.disconnect()
interface.disconnect()

