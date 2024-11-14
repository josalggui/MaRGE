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
# import seq.haste as haste
import seq.gre3d as gre3d
import seq.gre1d as gre1d
import seq.petra as petra
import seq.fid as fid
import seq.FIDandNoise as FIDandNoise
import seq.rabiFlops as rabiFlops
import seq.B1calibration as B1calibration
import seq.cpmg as tse
import seq.eddycurrents as eddycurrents
import seq.larmor as larmor
import seq.larmor_pypulseq as larmor_pypulseq
import seq.inversionRecovery as inversionRecovery
# import seq.ADCdelayTest as ADCdelayTest
import seq.noise as noise
import seq.shimmingSweep as shimming
import seq.testSE as testSE
# import seq.sliceSelection as sliceSelection
import seq.sweepImage as sweep
import seq.autoTuning as autoTuning
import seq.localizer as localizer
# import seq.MRID as mrid
import seq.tsePrescan as tsePrescan
# import seq.PETRAphysio as PETRAphysio
import seq.larmor_raw as larmor_raw
import seq.mse as mse
import seq.pulseq_reader as pulseq_reader
import seq.fix_gain as fix_gain
import seq.mse_pp as mse_pp
import seq.mse_pp_jma as mse_jma
import seq.rare_t2prep_pp as rare_t2prep_pp

class RARE(rare.RARE):
    def __init__(self): super(RARE, self).__init__()

# class PETRAphysio(PETRAphysio.PETRAphysio):
#     def __init__(self): super(PETRAphysio, self).__init__()

class TSEPrescan(tsePrescan.TSEPRE):
    def __init__(self): super(TSEPrescan, self).__init__()

class RAREProtocols(rareProtocols.RAREProtocols):
    def __init__(self): super(RAREProtocols, self).__init__()

class RAREProtocolsTest(rareProtocolsTest.RAREProtocolsTest):
    def __init__(self): super(RAREProtocolsTest, self).__init__()

class testSE(testSE.testSE):
    def __init__(self): super(testSE, self).__init__()

class GRE3D(gre3d.GRE3D):
    def __init__(self): super(GRE3D, self).__init__()

class GRE1D(gre1d.GRE1D):
    def __init__(self): super(GRE1D, self).__init__()

class PETRA(petra.PETRA):
    def __init__(self): super(PETRA, self).__init__()

# class HASTE(haste.HASTE):
#     def __init__(self): super(HASTE, self).__init__()

class FID(fid.FID):
    def __init__(self): super(FID, self).__init__()

# class MRID(mrid.MRID):
#     def __init__(self): super(MRID, self).__init__()

class FIDandNoise(FIDandNoise.FIDandNoise):
    def __init__(self): super(FIDandNoise, self).__init__()

class RabiFlops(rabiFlops.RabiFlops):
    def __init__(self): super(RabiFlops, self).__init__()

class B1calibration(B1calibration.B1calibration):
    def __init__(self): super(B1calibration, self).__init__()

class Larmor(larmor.Larmor):
    def __init__(self): super(Larmor, self).__init__()

class Noise(noise.Noise):
    def __init__(self): super(Noise, self).__init__()

class TSE(tse.TSE):
    def __init__(self): super(TSE, self).__init__()

class EDDYCURRENTS(eddycurrents.EDDYCURRENTS):
    def __init__(self): super(EDDYCURRENTS, self).__init__()

class IR(inversionRecovery.InversionRecovery):
    def __init__(self): super(IR, self).__init__()

# class ADCtest(ADCdelayTest.ADCdelayTest):
#     def __init__(self): super(ADCtest, self).__init__()

class Shimming(shimming.ShimmingSweep):
    def __init__(self): super(Shimming, self).__init__()

# class SliceSelection(sliceSelection.SliceSelection):
#     def __init__(self): super(SliceSelection, self).__init__()

class SWEEP(sweep.SweepImage):
    def __init__(self): super(SWEEP, self).__init__()

class AutoTuning(autoTuning.AutoTuning):
    def __init__(self): super(AutoTuning, self).__init__()

class Localizer(localizer.Localizer):
    def __init__(self): super(Localizer, self).__init__()

"""
Definition of default sequences
"""
defaultsequences = {
    'Larmor': Larmor(),
    'MSE_jma': mse_jma.MSE(),
    'RAREprotocols': RAREProtocols(),
    'RAREprotocolsTest': RAREProtocolsTest(),
    'RARE': RARE(),
    'RARE_pp': rare_pp.RARE_pp(),
    'RARE_T2prep_pp': rare_t2prep_pp.RARE_T2prep_pp(),
    'PulseqReader': pulseq_reader.PulseqReader(),
    'Noise': Noise(),
    'RabiFlops': RabiFlops(),
    'Shimming': Shimming(),
    'AutoTuning': AutoTuning(),
    'FixGain': fix_gain.FixGain(),
    'TSE_prescan': TSEPrescan(),
    'Localizer': Localizer(),
    'GRE3D': GRE3D(),
    'GRE1D': GRE1D(),
    'PETRA': PETRA(),
    # 'HASTE': HASTE(),
    # 'AutoTuning': AutoTuning(),
    'FID': FID(),
    'FIDandNoise': FIDandNoise(),
    'B1calibration': B1calibration(),
    'TSE': TSE(),
    'EDDYCURRENTS': EDDYCURRENTS(),
    'InversionRecovery': IR(),
    # 'ADCtest': ADCtest(),
    'SWEEP': SWEEP(),
    'testSE': testSE(),
    # 'PETRAphysio': PETRAphysio(),
    'Larmor Raw': larmor_raw.LarmorRaw(),
    'MSE': mse.MSE(),
    'MSE_PyPulseq': mse_pp.MSE(),
    'Larmor PyPulseq': larmor_pypulseq.LarmorPyPulseq(),
}