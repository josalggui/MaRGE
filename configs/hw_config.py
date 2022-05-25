# Config file for Physio MRI scanner at MRILab, i3M, CSIC, Spain.

# Note: I write Ocra1 Units as o.u.
# Ocra1 gain = 10 V/o.u.
# AE Techrom transductance 5 A/V
# From Ocra1 to current: 50 A/o.u.
# X axis: 25 mT/m/o.u., 0.5 mT/m/A, 2.5 mT/m/V
# Y axis: 40 mT/m/o.u., 0.8 mT/m/A, 4.0 mT/m/V
# Z axis: 35 mT/m/o.u., 0.7 mT/m/A, 3.5 mT/m/V

gFactor = [0.025, 0.040, 0.035] # (X, Y, Z) in T/m/o.u.
slewRate = 1000 # us/o.u., slew rate for gradient rises
stepsRate = 200 # steps/o.u., steps rate for gradient rises
gammaB = 42.56e6 # Hz/T, Gyromagnetic ratio
blkTime = 15 # us, blanking time of Barthel's RFPA
gradDelay = 9 # Gradient amplifier delay (us)
oversamplingFactor = 6 # Rx oversampling
maxRdPoints = 2**12 # Maximum number of points to be acquired by the red pitaya
maxOrders = 2**14 # Maximun number of orders to be processed by the red pitaya
deadTime = 400 # us, RF coil dead time

