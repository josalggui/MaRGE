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
    
def upload(direct):
    #Subject, Experiment y Scan se llaman como el nombre entero de la secuencia
    #Si un metadata se encuentra vacío [], no se guarda ese field por defecto
    #Desde el XML en XNAT se pueden ver los fields
    #BW se ha cambiado a bw, XNAT no emplea bien las mayúsculas parece ser
    
    #Cada archivo le cuesta 26 segundos para subirse

    #Conexión XNAT
    session = xnat.connect('http://localhost', user='admin', password='admin')
    interface = Interface(server='http://localhost', user='admin', password='admin') #Para SNAPSHOTS
    print("Connected to XNAT with API")
    
    #PROYECTO tiene que existir en XNAT
    project = session.projects['physio']
    project_snapshot = interface.select.project('physio').insert() #Para SNAPSHOTS
    
    #Seteamos el directorio en la carpeta de datos
    os.chdir(direct)
    
    #1-Accedemos a los files y guardamos sus nombres
    files = os.listdir()
    nii = 0
    mat = 0
    for i in files:
        if i.find('nii') != -1:
            nii = i
        elif i.find('mat') != -1:
            mat = i 
        
  
    #Si solo hay .mat, se sube solo .mat
    if nii == 0 and mat != 0:
        print("Only .mat file detected")
        
        #Eliminamos .nii y .mat de los nombres   
        filename = mat.split('.mat')[0]
        filename_date = mat.split('.mat')[0]
        filename = filename.replace(".", "-")
        
        s_label = filename
        
        #2-subject
        subject = session.classes.SubjectData(parent=project, label=s_label)
                
        #3-experiment
        experiment = session.classes.MrSessionData(parent=subject, label=filename)
        #Si queremos añadir FECHA
        array_date = filename_date.split('.')[1:4]
        date = array_date[0] + "-" + array_date[1] + "-" + array_date[2]
        experiment.date = date
        
        #4-scan
        scan = session.classes.MrScanData(parent=experiment, id=filename, type='MRI')
        
        print("Subject, Experiment and Scan created")
        
        #6-Resource y subir .mat   
        try:
            #XNATResponseError (status 409)
            resource_MAT = session.classes.ResourceCatalog(parent=scan, label='MAT', type='mat')
            resource_MAT.upload(mat, mat)
            print(".mat uploaded")
        except:
            resource_MAT = scan.resources['MAT']
            resource_MAT.upload(mat, mat)
            print(".mat already exists")    

        #7-METADATA
        metadata = scipy.io.loadmat(mat)
        
        if 'BW' in metadata.keys():
            metadata["bw"] = metadata.pop("BW")
        
        del metadata["image3D"]
        del metadata["kSpace3D"]
        del metadata["dataFull"]
        del metadata["imgFull"]
        del metadata["sampled"]
        del metadata["__header__"]
        del metadata["__version__"]
        del metadata["__globals__"]
        del metadata["phGradients"]
#        del metadata["average"]
#        del metadata["rawdata"]
        
        for key in metadata.keys():
            experiment.fields[key] = metadata.get(key)

        
        session.disconnect()
        interface.disconnect()
        
    elif nii != 0 and mat != 0:
        print(".mat and .nii files detected")

        #Eliminamos .nii y .mat de los nombres   
        filename = nii.split('.nii')[0]
        filename_date = nii.split('.nii')[0]
        filename = filename.replace(".", "-")
        
        s_label = filename.split('-')[0]
        exp_label = filename.split('-')[1]
        
        #2-subject
        subject = session.classes.SubjectData(parent=project, label=s_label)

        #3-experiment
        experiment = session.classes.MrSessionData(parent=subject, label=exp_label)

        #Si queremos añadir FECHA
        array_date = filename_date.split('.')[2:5]
        date = array_date[0] + "-" + array_date[1] + "-" + array_date[2]
        print(date)

        experiment.date = date
        
        #4-scan
        scan = session.classes.MrScanData(parent=experiment, id=filename, type='MRI')
        
        print("Subject, Experiment and Scan created")
        
        #5-Resource y subir NIFTI
        #Actualizamos el NIFTI si ya existe
        try:
            #XNATResponseError (status 409)
            resource_NIFTI = session.classes.ResourceCatalog(parent=scan, label='NIFTI', type='nii')
            resource_NIFTI.upload(nii, nii)
            print("NIFTI uploaded")
        except:
            resource_NIFTI = scan.resources['NIFTI']
            resource_NIFTI.upload(nii, nii)
            print("NIFTI already exists")
        
        #6-Resource y subir .mat   
        try:
            #XNATResponseError (status 409)
            resource_MAT = session.classes.ResourceCatalog(parent=scan, label='MAT', type='mat')
            resource_MAT.upload(mat, mat)
            print(".mat uploaded")
        except:
            resource_MAT = scan.resources['MAT']
            resource_MAT.upload(mat, mat)
            print(".mat already exists")
            
        #7-METADATA
        metadata = scipy.io.loadmat(mat)
        
        if 'BW' in metadata.keys():
            metadata["bw"] = metadata.pop("BW")
        
        del metadata["image3D"]
        del metadata["kSpace3D"]
        del metadata["dataFull"]
        del metadata["imgFull"]
        del metadata["sampled"]
        del metadata["__header__"]
        del metadata["__version__"]
        del metadata["__globals__"]
        del metadata["phGradients"]
        
        for key in metadata.keys():
            experiment.fields[key] = metadata.get(key)
        
        print("metadata uploaded")
        
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
        else:
        
            #Dimensión donde se encuentra el min
            shape=np.array(data.shape)
            for i in list(range(data.ndim)):
                if shape[i] == min(shape):
                    dim_10 = i
                
            
            images = []
            
            for i in list(range(min(shape))): 
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
            subject_snapshot = project_snapshot.subject(s_label)
            
            experiment = subject_snapshot.experiment(exp_label)
            
            experiment.scan(filename).resource('SNAPSHOTS').file(filename+ '.gif').insert(filename + '.gif',
                                                                 format="GIF",
                                                                 content = "ORIGINAL",
                                                                 tags = None)
            
            experiment.scan(filename).resource('SNAPSHOTS').file(filename+ '_t.gif').insert(filename + '.gif',
                                                                 format="GIF",
                                                                 content = "THUMBNAIL",
                                                                 tags = None)
            
            os.chdir("../")
            shutil.rmtree("SNAPSHOTS")
            print("Snapshots uploaded")
        # =============================================================================
        # FIN SNAPSHOTS
        # =============================================================================
        
        session.disconnect()
        interface.disconnect()

