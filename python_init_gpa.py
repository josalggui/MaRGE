import sys
sys.path.append('/home/physioMRI/gitHub_repos/marcos_client')

import experiment as ex


expt = ex.Experiment(init_gpa=True)
expt.run()
expt._del_()
