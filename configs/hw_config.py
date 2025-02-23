import numpy as np

# Config file for Physio MRI scanner at MRILab, i3M, CSIC, Spain.

# Note: I write Ocra1 Units as o.u.
# Ocra1 gain = 10 V/o.u.
# AE Techrom transductance. Unbalanced input: 5 A/V. Balanced input: 10 A/V.
# From Ocra1 to current: 50 A/o.u.
# X axis: 25 mT/m/o.u., 0.5 mT/m/A, 2.5 mT/m/V
# Y axis: 40 mT/m/o.u., 0.8 mT/m/A, 4.0 mT/m/V
# Z axis: 35 mT/m/o.u., 0.7 mT/m/A, 3.5 mT/m/V

#gFactor = np.array([0.025, 0.040, 0.035])*2 # (X, Y, Z) in T/m/o.u. GX VIEJO
gFactor = np.array([0.035, 0.040, 0.035])*2 # (X, Y, Z) in T/m/o.u. GX NUEVO
max_grad = 80  # mT/m
max_slew_rate = 50e-3  # mT/m/ms
grad_raster_time = 25e-6  # s
grad_rise_time = 400e-6 # s, time for gradient ramps
grad_steps = 16 # steps to gradient ramps
gammaB = 42.57747892e6 # Hz/T, Gyromagnetic ratio
blkTime = 15 # us, blanking time of Barthel's RFPA
gradDelay = 9 # Gradient amplifier delay (us)
oversamplingFactor = 5 # Rx oversampling
maxRdPoints = 2**17 # Maximum number of points to be acquired by the red pitaya
maxOrders = 2**14*6 # Maximun number of orders to be processed by the red pitaya
deadTime = 400 # us, RF coil dead time
b1Efficiency = np.pi/(1.0*15) # rads / (a.u. * us)
larmorFreq = 3.053482675254997 # MHz
cic_delay_points = 3 # to account for signal delay from red pitaya due to cic filter
addRdPoints = 10 # to account for wrong first points after decimation (50)
scanner_name = "Demo"
antenna_dict = {}
reference_time = 100  # us
fov = [18.0, 16.0, 16.0] # cm
dfov = [0.0, 0.0, 0.0] # mm

# To change the original reference system. Check in MaRGE wiki for more info.
rotations = []
dfovs = []
fovs = []

bash_path = "gnome-terminal"
rp_ip_list = []
rp_version = "rp-122"
adcFactor = 13.788
lnaGain = 45 # dB
temperature = 293 # k
shimming_factor = 1e-4
rf_min_gain = 35
rf_max_gain = 50


# Arduinos
ard_sn_autotuning = '4423431343435131E180'
ard_sn_interlock = '242353133363518050E1'
ard_sn_attenuator = '242353133363518050E2'