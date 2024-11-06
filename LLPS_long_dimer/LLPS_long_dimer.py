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
parser.add_argument("eps", help="epsilon of inter-molecular contact", type=float)
args = parser.parse_args()
name = args.name
eps = args.eps
T = 310.
n_copy = 100

print(name, T, eps)

seq1 = 'MRKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFGYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYKASMSKGPGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGRGDSPYSGYVDMAEAQNDPLLPGYSFNAHLVAGLTPIEANGYLDFFIDRPLGMKGYILNLTIRGQGVVKNQGREFVCRPGDILLFPPGEIHHYGRHPEAREWYHQWVYFRPRAYWHEWLNWPSIFANTGFFRPDEAHQPHFSDLFGQIINAGQGEGRYSELLAINLLEQLLLRRMEAINESLHPPMDNRVREACQYISDHLADSNFDIASVAQHVCLSPSRLSHLFRQQLGISVLSWREDQRISQAKLLLSTTRMPIATVGRNVGFDDQLYFSRVFKKCTGASPSEFRAGCEEKVNDVAVKLS'
# GFP 1-238, RLP20: 241-407, AraC: 410-701
seq2 = 'MRKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFGYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYKASMASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGNYGQDQSSMSSGGGSGGGYGNQDQSGGGGSGGYGQQDRGVDMAEAQNDPLLPGYSFNAHLVAGLTPIEANGYLDFFIDRPLGMKGYILNLTIRGQGVVKNQGREFVCRPGDILLFPPGEIHHYGRHPEAREWYHQWVYFRPRAYWHEWLNWPSIFANTGFFRPDEAHQPHFSDLFGQIINAGQGEGRYSELLAINLLEQLLLRRMEAINESLHPPMDNRVREACQYISDHLADSNFDIASVAQHVCLSPSRLSHLFRQQLGISVLSWREDQRISQAKLLLSTTRMPIATVGRNVGFDDQLYFSRVFKKCTGASPSEFRAGCEEKVNDVAVKLS'
# GFP 1-238, FUS N-terminal: 241-454, AraC: 457-748

sigmas = {'MET':0.618, 'GLY':0.45,  'LYS':0.636, 'THR':0.562, 'ARG':0.656,\
          'ALA':0.504, 'ASP':0.558, 'GLU':0.592, 'TYR':0.646, 'VAL':0.586,\
          'LEU':0.618, 'GLN':0.602, 'TRP':0.678, 'PHE':0.636, 'SER':0.518,\
          'HIS':0.608, 'ASN':0.568, 'PRO':0.556, 'CYS':0.548, 'ILE':0.618}

if name=='RLP20':
    seq = seq1[240:407]
    print(len(seq))
elif name=='FUSn':
    seq = seq2[240:454]
    print(len(seq))
elif name=='RLP20-AraC':
    seq = seq1[240:]
    print(len(seq))
    sec1 = list(range(175, 336))
    sec2 = list(range(343, 450))
elif name=='FUSn-AraC':
    seq = seq2[240:]
    print(len(seq))
    sec1 = list(range(222, 383))
    sec2 = list(range(390, 497))

seq_list = [seq for i in range(n_copy)]

system, topology = openfree.system_topology_define_fromseq(seq_list)
system.setDefaultPeriodicBoxVectors((21,0,0),(0,21,0),(0,0,300))
openfree.add_backbone_bonds(system, topology)

pdb1 = md.load('2arc.pdb')
pdb1_ca = pdb1.atom_slice(pdb1.top.select('name CA'))

angle_indeces1 = [[i, i+1, i+2] for i in range(len(sec1)-2)]
angle_indeces1 = angle_indeces1 + [[i+len(sec1), i+1+len(sec1), i+2+len(sec1)] for i in range(len(sec1)-2)]
dihedral_indeces1 = [[i, i+1, i+2, i+3] for i in range(len(sec1)-3)]
dihedral_indeces1 = dihedral_indeces1 + [[i+len(sec1), i+1+len(sec1), i+2+len(sec1), i+3+len(sec1)] for i in range(len(sec1)-3)]
intra_contact_indeces1 = []
intra_contact_cutoff1 = []
inter_contact_indeces1 = []
inter_contact_cutoff1 = []
for i in range(len(sec1)):
    for j in range(i+2, len(sec1)):
        intra_contact_indeces1.append([i,j])
        intra_contact_indeces1.append([i+len(sec1),j+len(sec1)])
        intra_contact_cutoff1.append((sigmas[pdb1_ca.top.residue(i).name]+sigmas[pdb1_ca.top.residue(j).name])/2)

for i in range(len(sec1)):
    for j in range(len(sec1)):
        inter_contact_indeces1.append([i,j+len(sec1)])
        inter_contact_cutoff1.append((sigmas[pdb1_ca.top.residue(i).name]+sigmas[pdb1_ca.top.residue(j+len(sec1)).name])/2)

angles1 = md.compute_angles(pdb1_ca, angle_indeces1)
dihedrals1 = md.compute_dihedrals(pdb1_ca, dihedral_indeces1)
intra_contacts1 = md.compute_contacts(pdb1_ca, intra_contact_indeces1)[0]
inter_contacts1 = md.compute_contacts(pdb1_ca, inter_contact_indeces1)[0]

angles1_mean = (angles1[0][0:159] + angles1[0][159::]) / 2
dihedrals1_diff = dihedrals1[0][0:158] - dihedrals1[0][158::]
dihedrals1_mean = (dihedrals1[0][0:158] + dihedrals1[0][158::]) / 2
dihedrals1_mean[np.where(dihedrals1_diff<-3)] += np.pi
intra_contacts1_mean = (intra_contacts1[0][0::2] + intra_contacts1[0][1::2]) / 2

pdb2 = md.load('2k9s.pdb')
pdb2_ca = pdb2.atom_slice(pdb2.top.select('name CA'))

angle_indeces2 = [[i, i+1, i+2] for i in range(len(sec2)-2)]
dihedral_indeces2 = [[i, i+1, i+2, i+3] for i in range(len(sec2)-3)]
intra_contact_indeces2 = []
intra_contact_cutoff2 = []
for i in range(len(sec2)):
    for j in range(i+2, len(sec2)):
        intra_contact_indeces2.append([i,j])
        intra_contact_cutoff2.append((sigmas[pdb2_ca.top.residue(i).name]+sigmas[pdb2_ca.top.residue(j).name])/2)

angles2 = md.compute_angles(pdb2_ca, angle_indeces2)[0]
dihedrals2 = md.compute_dihedrals(pdb2_ca, dihedral_indeces2)[0]
intra_contacts2 = md.compute_contacts(pdb2_ca, intra_contact_indeces2)[0][0]

angle_indeces = []
dihedral_indeces = []
intra_contacts = []
contact_cutoff = 1.5

all_intra_contact = np.concatenate([intra_contacts1_mean, intra_contacts2])
all_intra_contact_cutoff = np.concatenate([intra_contact_cutoff1, intra_contact_cutoff2])
selected_intra_contact = all_intra_contact[np.where(all_intra_contact-contact_cutoff*all_intra_contact_cutoff<0)].tolist()
selected_inter_contact = inter_contacts1[0,:][np.where(inter_contacts1[0,:]-contact_cutoff*np.array(inter_contact_cutoff1)<0)].tolist()

selected_intra_contact_copy = []
selected_inter_contact_copy = []
min1 = np.min(sec1)
min2 = np.min(sec2)
n_res = len(seq)

for i in range(n_copy):
    angle_indeces += [[i*n_res+min1+j, i*n_res+min1+1+j, i*n_res+min1+2+j] for j in range(len(sec1)-2)]
    angle_indeces += [[i*n_res+min2+j, i*n_res+min2+1+j, i*n_res+min2+2+j] for j in range(len(sec2)-2)]
    dihedral_indeces += [[i*n_res+min1+j, i*n_res+min1+1+j, i*n_res+min1+2+j, i*n_res+min1+3+j] for j in range(len(sec1)-3)]
    dihedral_indeces += [[i*n_res+min2+j, i*n_res+min2+1+j, i*n_res+min2+2+j, i*n_res+min2+3+j] for j in range(len(sec2)-3)]
    
    intra_contact_per = []
    for j in range(len(sec1)):
        for k in range(j+2, len(sec1)):
            intra_contact_per.append([i*n_res+min1+j, i*n_res+min1+k])

    for j in range(len(sec2)):
        for k in range(j+2, len(sec2)):
            intra_contact_per.append([i*n_res+min2+j, i*n_res+min2+k])

    intra_contacts += np.array(intra_contact_per)[np.where(all_intra_contact-contact_cutoff*all_intra_contact_cutoff<0)].tolist()
    selected_intra_contact_copy += selected_intra_contact

inter_contacts = []
for i_chain in range(0, n_copy, 2):
    j_chain = i_chain + 1
    inter_contact_per = []
    for i in range(len(sec1)):
        for j in range(len(sec1)):
            inter_contact_per.append([i+i_chain*n_res+min1, j+j_chain*n_res+min1])

    inter_contacts += np.array(inter_contact_per)[np.where(inter_contacts1[0,:]-contact_cutoff*np.array(inter_contact_cutoff1)<0)].tolist()
    selected_inter_contact_copy += selected_inter_contact

#openfree.add_backbone_angles(system, np.tile(np.concatenate([angles1_mean, angles2]), n_copy), angle_indeces)
#openfree.add_backbone_dihedrals(system, np.tile(np.concatenate([dihedrals1_mean, dihedrals2])-np.pi, n_copy), dihedral_indeces)
openfree.add_contacts(system, selected_intra_contact_copy, intra_contacts, contact_eps=2*4.184)
openfree.add_contacts(system, selected_inter_contact_copy, inter_contacts, contact_eps=eps*4.184)
print(len(selected_intra_contact_copy), len(selected_inter_contact_copy))
print(len(selected_intra_contact_copy+selected_inter_contact_copy), len(intra_contacts+inter_contacts))
print(len(angle_indeces), len(dihedral_indeces))

yukawa_eps, yukawa_kappa, HIS_charge = util.gen_params_electro(temp=T, pH=7.5, ionic=0.237)
print(yukawa_eps, yukawa_kappa, HIS_charge)

non_bonded_exclusion = []
'''
for i in range(n_copy):    
    for j in range(len(sec1)-3):
        for k in range(j+2, j+4):
            non_bonded_exclusion.append([i*n_res+min1+j, i*n_res+min1+k])
        if j==len(sec1)-4:
            non_bonded_exclusion.append([i*n_res+min1+j+1, i*n_res+min1+j+3])

    for j in range(len(sec2)-3):
        for k in range(j+2, j+4):
            non_bonded_exclusion.append([i*n_res+min2+j, i*n_res+min2+k])
        if j==len(sec2)-4:
            non_bonded_exclusion.append([i*n_res+min2+j+1, i*n_res+min2+j+3])

print(len(non_bonded_exclusion))
'''
non_bonded_exclusion += intra_contacts+inter_contacts

openfree.add_electrostatics(system, topology, yukawa_eps, yukawa_kappa, HIS_charge, periodic_cutoff=True, exclusion=non_bonded_exclusion)
openfree.add_LJs(system, topology, LJs_eps=0.2*4.184, rc=2., periodic_cutoff=True, exclusion=non_bonded_exclusion)

xyz = md.load(name + '_20ns_200K_LLPS_NPT_100_test.pdb').xyz[0]
xyz -= np.mean(xyz, axis=0)
xyz[:,0] += 10.5
xyz[:,1] += 10.5
xyz[:,2] += 150

integrator = openmm.LangevinIntegrator(T*unit.kelvin, 0.01/unit.picosecond, 0.01*unit.picosecond)
platform = openmm.Platform.getPlatformByName('CUDA')
#properties = {'CudaDeviceIndex': str(GPU),}
simulation = app.Simulation(topology, system, integrator, platform)#, properties)
simulation.context.setPositions(xyz)
system.usesPeriodicBoundaryConditions()

simulation.minimizeEnergy()

n_save = 2000
n_step = 100000
tag = 'LLPS_long_eps' + str(eps)
length = str(int(n_save*n_step*0.01/1000)) + 'ns'
simulation.reporters.append(app.PDBReporter('./' + name + '_' + length + '_' + str(int(T)) + 'K_' + tag + '.pdb',\
                                            n_save*n_step))
simulation.reporters.append(app.DCDReporter('./' + name + '_' + length + '_' + str(int(T)) + 'K_' + tag + '.dcd',\
                                            int(n_step)))#, enforcePeriodicBox=False))
simulation.reporters.append(app.StateDataReporter('./' + name + '_' + length + '_' + str(int(T)) + 'K_' + tag + '_log' + '.dat',\
                                                  n_step, step=True, potentialEnergy=True, temperature=True))

f = open("./" + name + "_" + length + "_" + str(int(T)) + "K_" + tag + "_energy_decomposition" + ".dat", "w")

for i in tqdm(range(n_save)):
    f.write(str(i+1)+' '+str(time.asctime(time.localtime(time.time())))+'\n')
    simulation.step(n_step)
    for j in range(7):
        f.write(str(simulation.context.getState(getEnergy=True, groups={j}).getPotentialEnergy())+'\n')
    
    f.write('\n')
    
f.close()
print('Simulation Finished!') 
