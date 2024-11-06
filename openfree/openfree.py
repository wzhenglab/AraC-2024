import openmm
import openmm.app as app
from openmm import unit
import mdtraj as md
import numpy as np

# General Constants Definition
#--------------------------------
seq1to3={'R':'ARG','H':'HIS','K':'LYS','D':'ASP','E':'GLU',\
         'S':'SER','T':'THR','N':'ASN','Q':'GLN','C':'CYS',\
         'U':'SEC','G':'GLY','P':'PRO','A':'ALA','V':'VAL',\
         'I':'ILE','L':'LEU','M':'MET','F':'PHE','Y':'TYR',\
         'W':'TRP'}
         
residues = ['MET', 'GLY', 'LYS', 'THR', 'ARG', 'ALA', 'ASP', 'GLU', 'TYR', 'VAL',\
            'LEU', 'GLN', 'TRP', 'PHE', 'SER', 'HIS', 'ASN', 'PRO', 'CYS', 'ILE']

masses = {'MET':131.199997, 'GLY':57.049999,  'LYS':128.199997, 'THR':101.099998, 'ARG':156.199997,\
          'ALA':71.080002,  'ASP':115.099998, 'GLU':129.100006, 'TYR':163.199997, 'VAL':99.070000,\
          'LEU':113.199997, 'GLN':128.100006, 'TRP':186.199997, 'PHE':147.199997, 'SER':87.080002,\
          'HIS':137.100006, 'ASN':114.099998, 'PRO':97.120003,  'CYS':103.099998, 'ILE':113.199997}

lambdas = {'MET':0.5308481134337497, 'GLY':0.7058843733666401, 'LYS':0.1790211738990582, 'THR':0.3713162976273964,\
           'ARG':0.7307624767517166, 'ALA':0.2743297969040348, 'ASP':0.0416040480605567, 'GLU':0.0006935460962935,\
           'TYR':0.9774611449343455, 'VAL':0.2083769608174481, 'LEU':0.6440005007782226, 'GLN':0.3934318551056041,\
           'TRP':0.9893764740371644, 'PHE':0.8672358982062975, 'SER':0.4625416811611541, 'HIS':0.4663667290557992,\
           'ASN':0.4255859009787713, 'PRO':0.3593126576364644, 'CYS':0.5615435099141777, 'ILE':0.5423623610671892}

sigmas = {'MET':0.618, 'GLY':0.45,  'LYS':0.636, 'THR':0.562, 'ARG':0.656,\
          'ALA':0.504, 'ASP':0.558, 'GLU':0.592, 'TYR':0.646, 'VAL':0.586,\
          'LEU':0.618, 'GLN':0.602, 'TRP':0.678, 'PHE':0.636, 'SER':0.518,\
          'HIS':0.608, 'ASN':0.568, 'PRO':0.556, 'CYS':0.548, 'ILE':0.618}

eps_ds = {'MET':-1.6,  'GLY':0.65,  'LYS':-0.95, 'THR':-0.3,  'ARG':-1.37,\
          'ALA':-2.59, 'ASP':-0.8,  'GLU':-1.8,  'TYR':-0.68, 'VAL':-0.75,\
          'LEU':-2.05, 'GLN':-1.25, 'TRP':-1.15, 'PHE':-0.68, 'SER':-0.69,\
          'HIS':0.8,   'ASN':-0.42, 'PRO':3.7,   'CYS':-0.15, 'ILE':-1.39}

charges = {'LYS':1., 'ARG':1., 'ASP':-1., 'GLU':-1., 'HIS':0.}


# System and Topology Definition
#--------------------------------
def system_topology_define(pdb_file):

    new_system = openmm.System()
    new_topology = app.topology.Topology()

    molecule = md.load(pdb_file)

    for i in range(molecule.top.n_chains):
        new_chain = new_topology.addChain()
        for j in range(molecule.top.chain(i).n_residues):
            resname = molecule.top.chain(i).residue(j).name
            if j==0:
                new_system.addParticle(masses[resname] + 2.)
            elif j==molecule.top.chain(i).n_residues-1:
                new_system.addParticle(masses[resname] + 16.)
            else:
                new_system.addParticle(masses[resname])

            # Here we use only one bead for each residue by default
            atom_name = molecule.top.chain(i).residue(j).atom(0).name
            atom_element = molecule.top.chain(i).residue(j).atom(0).element
            new_residue = new_topology.addResidue(resname, new_chain)
            new_topology.addAtom(atom_name, atom_element, new_residue)


    return new_system, new_topology

# System and Topology Definition from only sequence
#--------------------------------
def system_topology_define_fromseq(seq_list):
    # The input seq_list should be a list of strings containing all the chains including proteins and other molecules

    new_system = openmm.System()
    new_topology = app.topology.Topology()

    for seq in seq_list:
        new_chain = new_topology.addChain()
        for j, res in enumerate(seq):
            if res in list(seq1to3.keys()):
                resname = seq1to3[res]
                if j==0:
                    new_system.addParticle(masses[resname] + 2.)
                elif j==len(seq)-1:
                    new_system.addParticle(masses[resname] + 16.)
                else:
                    new_system.addParticle(masses[resname])

                atom_name = 'CA'
                atom_element = openmm.app.element.Element.getBySymbol('C')
                new_residue = new_topology.addResidue(resname, new_chain)
                new_topology.addAtom(atom_name, atom_element, new_residue)

            else:
                print('Wrong letter for amino acid')

    return new_system, new_topology

# Add Bond Potential as well as in the Toppology
#--------------------------------
def add_backbone_bonds(system, topology, bond_r0=0.382, bond_K=4184.):

    backbone_bonds = openmm.HarmonicBondForce()

    for chain in topology.chains():
        atom_list = list(chain.atoms())
        for i in range(1, len(atom_list)):
            topology.addBond(atom_list[i-1], atom_list[i])
            backbone_bonds.addBond(atom_list[i-1].index, atom_list[i].index, bond_r0, bond_K)

    backbone_bonds.setForceGroup(0) # Harmonic bond interaction is set as interaction 0
    system.addForce(backbone_bonds)

    return backbone_bonds

# Add Harmonic Angle Force (Since typically only one angular force will be added, here we use same Force Group Number)
#--------------------------------
def add_backbone_angles(system, angle_theta0, angle_list, angle_K=2*4.184):

    backbone_angles = openmm.HarmonicAngleForce()

    for i, angle in enumerate(angle_list):
        backbone_angles.addAngle(angle[0], angle[1], angle[2], angle_theta0[i], angle_K)

    backbone_angles.setForceGroup(1) # Harmonic angle interaction is set as interaction 1
    system.addForce(backbone_angles)

    return backbone_angles

# Add Periodic Torsion Force (Since typically only one dihedral force will be added, here we use same Force Group Number)
#--------------------------------
def add_backbone_dihedrals(system, dihedral_theta0, dihedral_list, dihedral_K=0.5*4.184):

    backbone_dihedrals = openmm.PeriodicTorsionForce()

    for i, dihedral in enumerate(dihedral_list):
        backbone_dihedrals.addTorsion(dihedral[0], dihedral[1], dihedral[2], dihedral[3], 1, dihedral_theta0[i], dihedral_K)

    backbone_dihedrals.setForceGroup(2) # Periodic Torsion interaction is set as interaction 2
    system.addForce(backbone_dihedrals)

    return backbone_dihedrals

# Add Custom Angle Potential
#--------------------------------
def add_backbone_custom_angles(system, topology):

    backbone_custom_angles = openmm.CustomAngleForce("-log(exp(-gamma*(k_a*(theta-theta_a)^2+e_a))+exp(-gamma*(k_b*(theta-theta_b)^2)))/gamma")

    backbone_custom_angles.addGlobalParameter("gamma", 0.0239)
    backbone_custom_angles.addGlobalParameter("k_a", 445.1776)
    backbone_custom_angles.addGlobalParameter("theta_a", 1.60)
    backbone_custom_angles.addGlobalParameter("e_a", 17.9912)
    backbone_custom_angles.addGlobalParameter("k_b", 110.0392)
    backbone_custom_angles.addGlobalParameter("theta_b", 2.27)

    for chain in topology.chains():
        atom_list = list(chain.atoms())
        for i in range(2, len(atom_list)):
            backbone_custom_angles.addAngle(atom_list[i-2].index, atom_list[i-1].index, atom_list[i].index)

    
    backbone_custom_angles.setForceGroup(1) # Custom angle interaction is set as interaction 1
    system.addForce(backbone_custom_angles)

    return

# Add Custom Dihedral Potential
#--------------------------------
def add_backbone_custom_dihedrals(system, topology, delta_e_d=None):

    U_dihed_a = "exp(-k_a1*(theta-theta_a1)^2-e_d)+exp(-k_a2*(theta-theta_a2)^4+e_0)+exp(-k_a2*(theta-theta_a2+6.2832)^4+e_0)"
    U_dihed_b = "exp(-k_b1*(theta-theta_b1)^2+e_1+e_d)+exp(-k_b1*(theta-theta_b1-6.2832)^2+e_1+e_d)+\
                 exp(-k_b2*(theta-theta_b2)^4+e_2)+exp(-k_b2*(theta-theta_b2-6.2832)^4+e_2)"
    
    backbone_custom_dihedrals = openmm.CustomTorsionForce( "-4.184*log(" + U_dihed_a + "+" + U_dihed_b + ")")

    backbone_custom_dihedrals.addGlobalParameter("k_a1", 11.4)
    backbone_custom_dihedrals.addGlobalParameter("k_a2", 0.15)
    backbone_custom_dihedrals.addGlobalParameter("theta_a1", 0.90)
    backbone_custom_dihedrals.addGlobalParameter("theta_a2", 1.02)
    backbone_custom_dihedrals.addGlobalParameter("e_0", 0.27)
    backbone_custom_dihedrals.addGlobalParameter("k_b1", 1.08)
    backbone_custom_dihedrals.addGlobalParameter("k_b2", 0.65)
    backbone_custom_dihedrals.addGlobalParameter("theta_b1", -1.55)
    backbone_custom_dihedrals.addGlobalParameter("theta_b2", -2.50)
    backbone_custom_dihedrals.addGlobalParameter("e_1", 0.14)
    backbone_custom_dihedrals.addGlobalParameter("e_2", 0.40)
    backbone_custom_dihedrals.addPerTorsionParameter("e_d")
    
    e_d_list = []

    if delta_e_d is not None:
        if len(delta_e_d)!=len(list(topology.chains())):
            print("Error! Input delta_e_d dimension does not match the number of chains!")
            return

    for n, chain in enumerate(list(topology.chains())):
        atom_list = list(chain.atoms())
        if delta_e_d is not None:
            if len(delta_e_d[n])!=len(atom_list)-3:
                print("Error! Input delta_e_d dimension does not match the number of dihedrals for chain {}!".foramt(n))
                return
            
        for i in range(3, len(atom_list)):
            if i==3:
                e_d = (eps_ds[atom_list[i-3].residue.name] + eps_ds[atom_list[i].residue.name] + eps_ds[atom_list[i+1].residue.name]) / 3.
            elif i==len(atom_list)-1:
                e_d = (eps_ds[atom_list[i-4].residue.name] + eps_ds[atom_list[i-3].residue.name] + eps_ds[atom_list[i].residue.name]) / 3.
            else:
                e_d = (eps_ds[atom_list[i-4].residue.name] + eps_ds[atom_list[i-3].residue.name] +\
                       eps_ds[atom_list[i].residue.name]   + eps_ds[atom_list[i+1].residue.name]) / 4.
                
            if delta_e_d is not None:
                e_d = e_d + delta_e_d[n][i-3]
                
            backbone_custom_dihedrals.addTorsion(atom_list[i-3].index, atom_list[i-2].index, atom_list[i-1].index, atom_list[i].index, [e_d])
            e_d_list.append(e_d)
    
    backbone_custom_dihedrals.setForceGroup(2) # Custom dihedral interaction is set as interaction 2
    system.addForce(backbone_custom_dihedrals)

    return e_d_list

# Add CALVADOS2 Electrostatic Potential
#--------------------------------
def add_electrostatics(system, topology, yukawa_eps, yukawa_kappa, HIS_charge, periodic_cutoff=False, exclusion='1-2'):

    electrostatics = openmm.CustomNonbondedForce("q*(exp(-kappa*r)/r-shift); q=q1*q2")

    electrostatics.addGlobalParameter("kappa" ,yukawa_kappa/unit.nanometer)
    electrostatics.addGlobalParameter("shift", np.exp(-yukawa_kappa*4.0)/4.0/unit.nanometer)
    electrostatics.addPerParticleParameter("q")

    electrostatics.setCutoffDistance(4*unit.nanometer)
    if periodic_cutoff:
        electrostatics.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)

    # Assign charges
    charges['HIS'] += HIS_charge
    for chain in topology.chains():
        atom_list = list(chain.atoms())
        for i in range(len(atom_list)):
            if atom_list[i].residue.name in list(charges.keys()):
                charge = charges[atom_list[i].residue.name]
            else:
                charge = 0.

            if i==0:
                electrostatics.addParticle([(charge + 1.)*yukawa_eps])
            elif i==len(atom_list)-1:
                electrostatics.addParticle([(charge - 1.)*yukawa_eps])
            else:
                electrostatics.addParticle([charge*yukawa_eps])


    # Add exclusions for atoms within bonded interactions
    if exclusion not in ['1-2', '1-3', '1-4'] and not isinstance(exclusion, list):
        print('Wrong Exclusion Mode! Using 1-2 Exclusion')
        exclusion = '1-2'
    for chain in topology.chains():
        atom_list = list(chain.atoms())
        for i in range(1, len(atom_list)):
            electrostatics.addExclusion(atom_list[i-1].index, atom_list[i].index)
            if i > 1 and exclusion in ['1-3', '1-4']:
                electrostatics.addExclusion(atom_list[i-2].index, atom_list[i].index)
            if i > 2 and exclusion=='1-4':
                electrostatics.addExclusion(atom_list[i-3].index, atom_list[i].index)    

    if isinstance(exclusion, list):
        for pair in exclusion:
            electrostatics.addExclusion(pair[0], pair[1])

    electrostatics.setForceGroup(3) # Electrostatic interaction is set as interaction 3
    system.addForce(electrostatics)
    return

# Add CALVADOS2 LJs Potential
#--------------------------------
def add_LJs(system, topology, LJs_eps, rc, periodic_cutoff=False, exclusion='1-2'):

    LJs_expression = "select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))"
    LJs = openmm.CustomNonbondedForce(LJs_expression + "; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6")

    LJs.addGlobalParameter("eps" ,LJs_eps*unit.kilojoules_per_mole)
    LJs.addGlobalParameter("rc", rc*unit.nanometer)
    LJs.addPerParticleParameter("s")
    LJs.addPerParticleParameter("l")

    LJs.setCutoffDistance(rc*unit.nanometer)
    if periodic_cutoff:
        LJs.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)

    # Assign atom parameters
    for chain in topology.chains():
        atom_list = list(chain.atoms())
        for i in range(len(atom_list)):
            LJs.addParticle([sigmas[atom_list[i].residue.name]*unit.nanometer, lambdas[atom_list[i].residue.name]*unit.dimensionless])

    # Add exclusions for atoms within bonded interactions
    if exclusion not in ['1-2', '1-3', '1-4'] and not isinstance(exclusion, list):
        print('Wrong Exclusion Mode! Using 1-2 Exclusion')
        exclusion = '1-2'
    for chain in topology.chains():
        atom_list = list(chain.atoms())
        for i in range(1, len(atom_list)):
            LJs.addExclusion(atom_list[i-1].index, atom_list[i].index)
            if i > 1 and exclusion in ['1-3', '1-4']:
                LJs.addExclusion(atom_list[i-2].index, atom_list[i].index)
            if i > 2 and exclusion=='1-4':
                LJs.addExclusion(atom_list[i-3].index, atom_list[i].index)    

    if isinstance(exclusion, list):
        for pair in exclusion:
            LJs.addExclusion(pair[0], pair[1])

    LJs.setForceGroup(4) # Electrostatic interaction is set as interaction 4
    system.addForce(LJs)
    return

# Add Crosslink Potential
#--------------------------------
def add_crosslinks(system, k, A, r_x, xlk_list):

    crosslinks = openmm.CustomBondForce("-k_x*exp(-A_x*(r-r_x)^2)")

    crosslinks.addGlobalParameter("k_x", k*unit.kilojoules_per_mole)
    crosslinks.addGlobalParameter("A_x", A/(unit.nanometer*unit.nanometer))
    crosslinks.addGlobalParameter("r_x", r_x*unit.nanometer)

    # Add crosslink bonds
    for xlk in xlk_list:
        crosslinks.addBond(xlk[0], xlk[1])
   
    crosslinks.setForceGroup(5) # Intermolecular crosslink interaction is set as interaction 5
    system.addForce(crosslinks)
    return

# Add Contact Potential
#--------------------------------
def add_contacts(system, r_contact, contact_list, contact_eps=4.184):

    contacts = openmm.CustomBondForce("eps_con*(5*(r_con/r)^12-6*(r_con/r)^10)")

    contacts.addPerBondParameter("eps_con")
    contacts.addPerBondParameter("r_con")

    # Add crosslink bonds
    for i, contact in enumerate(contact_list):
        if isinstance(contact_eps, float):
            contacts.addBond(contact[0], contact[1], [contact_eps*unit.kilojoules_per_mole, r_contact[i]*unit.nanometer])
        elif isinstance(contact_eps, list):
            contacts.addBond(contact[0], contact[1], [contact_eps[i]*unit.kilojoules_per_mole, r_contact[i]*unit.nanometer])
        else:
            print('Wrong parameter type!')
   
    contacts.setForceGroup(6) # Contact interaction is set as interaction 6
    system.addForce(contacts)
    return

# Add Exclusive Volume Potential
#--------------------------------
def add_exclusive(system, exclusion_list, exclusion_eps=4.184):

    exclusive = openmm.CustomBondForce("eps_exclu*(r_exclu/r)^12")

    exclusive.addGlobalParameter("eps_exclu", exclusion_eps*unit.kilojoules_per_mole)
    exclusive.addGlobalParameter("r_exclu", 0.4*unit.nanometer)

    # Add excluion
    for pair in exclusion_list:
        exclusive.addBond(pair[0], pair[1])
   
    exclusive.setForceGroup(7) # Exclusive interaction is set as interaction 7
    system.addForce(exclusive)
    return