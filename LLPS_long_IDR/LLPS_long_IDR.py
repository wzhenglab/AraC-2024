import sys
sys.path.append('../openfree/')

import mdtraj as md
import openfree
import util
import time
import openmm
import numpy as np
import openmm.app as app
from tqdm import tqdm
from openmm import unit
import matplotlib.pyplot as plt
from biopandas.pdb import PandasPdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", help="system name", type=str)
parser.add_argument("T", help="Temperature", type=float)
args = parser.parse_args()
name = args.name
T = args.T

print(name, T)

seq1 = 'MRKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFGYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYKASMSKGPGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGYVDMAEAQNDPLLPGYSFNAHLVAGLTPIEANGYLDFFIDRPLGMKGYILNLTIRGQGVVKNQGREFVCRPGDILLFPPGEIHHYGRHPEAREWYHQWVYFRPRAYWHEWLNWPSIFANTGFFRPDEAHQPHFSDLFGQIINAGQGEGRYSELLAINLLEQLLLRRMEAINESLHPPMDNRVREACQYISDHLADSNFDIASVAQHVCLSPSRLSHLFRQQLGISVLSWREDQRISQAKLLLSTTRMPIATVGRNVGFDDQLYFSRVFKKCTGASPSEFRAGCEEKVNDVAVKLS'
# GFP 1-238, RLP20: 241-407, AraC: 410-701
seq2 = 'MRKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFGYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYKASMASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGNYGQDQSSMSSGGGSGGGYGNQDQSGGGGSGGYGQQDRGVDMAEAQNDPLLPGYSFNAHLVAGLTPIEANGYLDFFIDRPLGMKGYILNLTIRGQGVVKNQGREFVCRPGDILLFPPGEIHHYGRHPEAREWYHQWVYFRPRAYWHEWLNWPSIFANTGFFRPDEAHQPHFSDLFGQIINAGQGEGRYSELLAINLLEQLLLRRMEAINESLHPPMDNRVREACQYISDHLADSNFDIASVAQHVCLSPSRLSHLFRQQLGISVLSWREDQRISQAKLLLSTTRMPIATVGRNVGFDDQLYFSRVFKKCTGASPSEFRAGCEEKVNDVAVKLS'
# GFP 1-238, FUS N-terminal: 241-454, AraC: 457-748
print(len(seq1), len(seq2))

if name=='RLP20':
    seq = seq1[240:407]
    print(len(seq))
elif name=='FUSn':
    seq = seq2[240:454]
    print(len(seq))

seq_100 = []
for i in range(100):
    seq_100.append(seq)

system, topology = openfree.system_topology_define_fromseq(seq_100)
system.setDefaultPeriodicBoxVectors((16,0,0),(0,16,0),(0,0,300))

openfree.add_backbone_bonds(system, topology)
#openfree.add_backbone_custom_angles(system, topology)
#e_d_list = openfree.add_backbone_custom_dihedrals(system, topology)
yukawa_eps, yukawa_kappa, HIS_charge = util.gen_params_electro(temp=T, pH=7.5, ionic=0.237)
print(yukawa_eps, yukawa_kappa, HIS_charge)
openfree.add_electrostatics(system, topology, yukawa_eps, yukawa_kappa, HIS_charge, periodic_cutoff=True, exclusion='1-2')
openfree.add_LJs(system, topology, LJs_eps=0.2*4.184, rc=2., periodic_cutoff=True, exclusion='1-2')

xyz = md.load(name + '_20ns_200K_LLPS_NPT_100.pdb').xyz[0]
xyz -= np.mean(xyz, axis=0)
xyz[:,0] += 8
xyz[:,1] += 8
xyz[:,2] += 150

integrator = openmm.LangevinIntegrator(T*unit.kelvin, 0.01/unit.picosecond, 0.01*unit.picosecond)
platform = openmm.Platform.getPlatformByName('CUDA')
#properties = {'CudaDeviceIndex': str(GPU),}
simulation = app.Simulation(topology, system, integrator, platform)#, properties)
simulation.context.setPositions(xyz)
system.usesPeriodicBoundaryConditions()

simulation.minimizeEnergy()

n_save = 1000
n_step = 100000
tag = 'LLPS_long'

length = str(int(n_save*n_step*0.01/1000)) + 'ns'
simulation.reporters.append(app.PDBReporter('./' + name + '_' + length + '_' + str(int(T)) + 'K_' + tag + '.pdb',\
                                            n_save*n_step))
simulation.reporters.append(app.DCDReporter('./' + name + '_' + length + '_' + str(int(T)) + 'K_' + tag + '.dcd',\
                                            int(n_step)))#, enforcePeriodicBox=False))
simulation.reporters.append(app.StateDataReporter('./' + name + '_' + length + '_' + str(int(T)) + 'K_' + tag + '_log' + '.dat',\
                                                  n_step, step=True, potentialEnergy=True, temperature=True))

f = open("./" + name + "_" + length + "_" + str(int(T)) + "K_" + tag + "_energy_decomposition" + ".dat", "w")

for i in tqdm(range(n_save)):
    '''
    if i < 2000:
        integrator.setTemperature((500. - (200.*(i/1000.))) * unit.kelvin)
    else:
        integrator.setTemperature(300.* unit.kelvin)
    '''
    #print(i+1, time.asctime(time.localtime(time.time())))
    #sys.stdout.flush()
    f.write(str(i+1)+' '+str(time.asctime(time.localtime(time.time())))+'\n')
    simulation.step(n_step)
    for j in range(6):
        f.write(str(simulation.context.getState(getEnergy=True, groups={j}).getPotentialEnergy())+'\n')
    
    f.write('\n')
    
f.close()
print('Simulation Finished!') 
