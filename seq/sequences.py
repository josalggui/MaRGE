"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""

import seq.rare as rare
import seq.rare_pp as rare_pp
import seq.rareProtocols as rareProtocols
import seq.rareProtocolsTest as rareProtocolsTest
import seq.gre3d as gre3d
import seq.gre1d as gre1d
import seq.petra as petra
import seq.spds as spds
import seq.fid as fid
import seq.FIDandNoise as FIDandNoise
import seq.rabiFlops as rabiFlops
import seq.B1calibration as B1calibration
import seq.cpmg as tse
import seq.eddycurrents as eddycurrents
import seq.larmor as larmor
import seq.larmor_pypulseq as larmor_pypulseq
import seq.inversionRecovery as inversionRecovery
import seq.noise as noise
import seq.shimmingSweep as shimming
import seq.sweepImage as sweep
import seq.autoTuning as autoTuning
import seq.localizer as localizer
import seq.larmor_raw as larmor_raw
import seq.mse as mse
import seq.pulseq_reader as pulseq_reader
import seq.fix_gain as fix_gain
import seq.mse_pp as mse_pp
import seq.mse_pp_jma as mse_jma
import seq.rare_t2prep_pp as rare_t2prep_pp

"""
Definition of default sequences
"""
defaultsequences = {
    'Larmor': larmor.Larmor(),
    'MSE_jma': mse_jma.MSE(),
    'RAREprotocols': rareProtocols.RAREProtocols(),
    'RAREprotocolsTest': rareProtocolsTest.RAREProtocolsTest(),
    'RARE': rare.RARE(),
    'RarePyPulseq': rare_pp.RarePyPulseq(),
    'RARE_T2prep_pp': rare_t2prep_pp.RARE_T2prep_pp(),
    'PulseqReader': pulseq_reader.PulseqReader(),
    'Noise': noise.Noise(),
    'RabiFlops': rabiFlops.RabiFlops(),
    'Shimming': shimming.ShimmingSweep(),
    'AutoTuning': autoTuning.AutoTuning(),
    'SPDS': spds.spds(),
    'FixGain': fix_gain.FixGain(),
    'Localizer': localizer.Localizer(),
    'GRE3D': gre3d.GRE3D(),
    'GRE1D': gre1d.GRE1D(),
    'PETRA': petra.PETRA(),
    'FID': fid.FID(),
    'FIDandNoise': FIDandNoise.FIDandNoise(),
    'B1calibration': B1calibration.B1calibration(),
    'TSE': tse.TSE(),
    'EDDYCURRENTS': eddycurrents.EDDYCURRENTS(),
    'InversionRecovery': inversionRecovery.InversionRecovery(),
    'SWEEP': sweep.SWEEP(),
    'Larmor Raw': larmor_raw.LarmorRaw(),
    'MSE': mse.MSE(),
    'MSE_PyPulseq': mse_pp.MSE(),
    'Larmor PyPulseq': larmor_pypulseq.LarmorPyPulseq(),
}