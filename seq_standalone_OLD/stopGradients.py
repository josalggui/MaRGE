
import sys
sys.path.append('../marcos_client')
import experiment as expt
import numpy as np

def stopGradients(init_gpa= False):
    expt.add_flodict({
                        'grad_vx': (np.array([20]),np.array([0]) ), 
                        'grad_vy': (np.array([20]),np.array([0]) ), 
                        'grad_vz': (np.array([20]),np.array([0]) ),
             })
             
             
if __name__ == "__main__":
    stopGradients()
