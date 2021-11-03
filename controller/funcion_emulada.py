# -*- coding: utf-8 -*-

#Este script emula la llama de la funcion desde adquisition controller
#Primero debe comprobar si XNAT está lanzado, si no lanzarlo
#Esperar a que XNAT esté correctamente lanzado antes de entrar a la pipeline UPLOAD
#Se le debe pasar como argumento la ruta de archivos


#Se le debe pasar como argumento la ruta de archivos (FALTA)

#Si queremos parar XNAT: sudo docker stop `sudo docker ps -q`
    
import subprocess
import os
import requests
import time
import sys
import socket
import urllib
from sys import exit

#PARA NO LIAR EL PATH
true_path = os.getcwd()

#SOCKET
a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
location = ("0.0.0.0", 80) #Address, Port
result_of_check = a_socket.connect_ex(location)

if result_of_check == 0:
    print("XNAT is currently running")
    a_socket.close()
    
else:
    print("XNAT is not running, launching XNAT")
    
    os.chdir("/home/physioMRI/xnat-docker-compose")
    
    try:
        bash_docker_start = "sudo service docker start"
        p_start = subprocess.Popen(bash_docker_start.split())
        
        bash_XNAT_start = "sudo docker-compose up -d"
        p_start = subprocess.Popen(bash_XNAT_start.split())
        
        timeout = time.time() + 60*3   #3 minutes from now 
        
        a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result_of_check = a_socket.connect_ex(location)
        while True:
            a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result_of_check = a_socket.connect_ex(location)

            if result_of_check == 0 or time.time() > timeout:
                break
            
        a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result_of_check = a_socket.connect_ex(location)
        if result_of_check == 0:
            print("XNAT has been successfully launched")
        else:
            print("There was a problem launching XNAT")
            a_socket.close()
            os.chdir(true_path)
            exit
    except:
        print("There was an error launching XNAT")
        a_socket.close()
        os.chdir(true_path)
        exit

#Comprobación directa de la web (Puede dar problemas)
try:
    print("One last check for XNAT http://localhost/")
    url = "http://localhost/"
    status_code = urllib. request. urlopen(url).getcode()
    website_is_up = status_code == 200
    print("XNAT is running correctly")
except:
    print("There was an error launching XNAT, the port is open but the web is not working")
    os.chdir(true_path)
    exit
    
os.chdir(true_path)

from upload_SINGLE_nifti_XNATpy_2 import upload    

direct="/home/physioMRI/git_repos/PhysioMRI_GUI/experiments/acquisitions/2021.10.13/2021.10.13.21.20.33"


try:
    direct="/home/physioMRI/git_repos/PhysioMRI_GUI/experiments/acquisitions/2021.10.21/2021.10.21.14.45.25"
    upload(direct)
    print("1")
    direct="/home/physioMRI/git_repos/PhysioMRI_GUI/experiments/acquisitions/2021.10.21/2021.10.21.14.16.34"
    upload(direct)
    print("2")
    direct="/home/physioMRI/git_repos/PhysioMRI_GUI/experiments/acquisitions/2021.10.21/2021.10.21.14.44.43"
    upload(direct)
    print("3")
except:
    print("There was a problem with the upload")
    
os.chdir(true_path)



