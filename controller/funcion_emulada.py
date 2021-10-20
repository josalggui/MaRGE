# -*- coding: utf-8 -*-

#Este script emula la llama de la funcion desde adquisition controller
#Primero debe comprobar si XNAT está lanzado, si no lanzarlo
#Esperar a que XNAT esté correctamente lanzado antes de entrar a la pipeline UPLOAD
#Se le debe pasar como argumento la ruta de archivos
import subprocess
import os

os.chdir("/home/physioMRI/xnat-docker-compose")
bash_start = "sudo docker-compose up -d"
p_start = subprocess.Popen(bash_start.split())

#bash_stop = "sudo docker stop e3f42fd9eb1a"
#e3f42fd9eb1a
#3d98fcc011f5
#p_stop = subprocess.Popen("sudo docker stop `sudo docker ps -q`")
