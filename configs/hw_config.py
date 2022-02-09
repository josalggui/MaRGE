#Gradients equivalency

Gx_factor =   5.0 # mT/m/V
Gy_factor =   8.0 # mT/m/V
Gz_factor =   7.0 # mT/m/V

#Gradinets pulse parameters
slewRate = 20 #20us/A*(5V/A)^-1 NO TENGO CLARAS LAS UNIDADES, COMPROBAR!
stepsRate = 4 #steps/A #Tengo que convertirlo a unidades de la RP
gamma = 42.56e6             #Gyromagnetic ratio in Hz/T
chageOcra1RP = 5            #Ocra1 to RP conversion factor A/V
blkTime = 10                #Blanking time of Barthel's RFPA
gradDelay = 9               # Gradient amplifier delay (us)
oversamplingFactor = 6
