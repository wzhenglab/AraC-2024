import mdtraj as md
import numpy as np
from copy import deepcopy

# Coarse-graining pdb to Ca mapping
#--------------------------------
def CA_CG(input_pdb_file, save_file):

    pdb = md.load(input_pdb_file)
    n_residues = pdb.top.n_residues
    CA_indices = pdb.top.select('name CA')
    if CA_indices.shape[0] !=  n_residues:
        print('Error! Ca atom number does not match residue number!')
    else:
        CA_pdb = pdb.atom_slice(CA_indices)
        CA_pdb.save_pdb(save_file)

    return CA_pdb

# Generate the parameters for electrostatic interation
#--------------------------------
def gen_params_electro(temp, pH, ionic):

    kT = 8.3145*temp*1e-3

    # Set the charge on HIS based on the pH of the protein solution
    HIS_charge = 1. / ( 1 + 10 ** (pH - 6) )

    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321 / T + 233.76 - 0.9297 * T + 0.1417 * 1e-2 * T ** 2 - 0.8292 * 1e-6 * T ** 3
    epsw = fepsw(temp)
    lB = 1.6021766 ** 2 / (4 * np.pi * 8.854188 * epsw) * 6.022 * 1000 / kT
    yukawa_eps = np.sqrt(lB * kT)

    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8 * np.pi * lB * ionic * 6.022 / 10)
    
    return yukawa_eps, yukawa_kappa, HIS_charge

# Make multiple replicas of one molecule
#--------------------------------
def replica(input_pdb_file, save_file, displacement, n_replicas):

    pdb = md.load(input_pdb_file)
    pdb_c = deepcopy(pdb)
    for i in range(1, n_replicas):
        pdb_c.xyz += displacement
        pdb = pdb.stack(pdb_c)

    pdb.save_pdb(save_file)

    return pdb

# Generate crosslink pairs accroding to the input locations
#--------------------------------
def xlk_location2pair(topology, locations):

    xlk_pairs = []
    chain_list = list(topology.chains())
    n_chains = len(list(topology.chains()))

    acc_n_atoms_i = 0
    for i in range(n_chains):
        atom_list_i = list(chain_list[i].atoms())
        acc_n_atoms_j = acc_n_atoms_i + len(atom_list_i)
        for j in range(i+1, n_chains):
            atom_list_j = list(chain_list[j].atoms())
            for atom_i in atom_list_i:
                for atom_j in atom_list_j:
                    if int(atom_i.index - acc_n_atoms_i)==int(atom_j.index - acc_n_atoms_j) and\
                       int(atom_i.index - acc_n_atoms_i) in locations:
                        xlk_pairs.append((atom_i.index, atom_j.index))

            acc_n_atoms_j += len(atom_list_j)

        acc_n_atoms_i += len(atom_list_i)

    return xlk_pairs