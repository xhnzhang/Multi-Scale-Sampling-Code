# Multiscale sampling (MSS) with VASP and LAMMPS
# Xiaohong Zhang
# Getman Research Group
# Mar 10, 2019
import sys
import numpy as np 
from collections import OrderedDict 
import pandas as pd


def readSol(sol_name):
	if sol_name.upper() == "TIP3PCHARMM":
		return TIP3PCHARMM()	
	elif sol_name.upper() == "TIP3PPRICE":
		return TIP3PPrice()
	elif sol_name.upper() == "TIP4P2005":
		return TIP4P2005()
	elif sol_name.upper() == "METHANOL":
		return Methanol()
	else:
		print("\nERROR: solvent type not defined.\n")


class TIP3PCHARMM:
	def __init__(self):
		self.name = 'TIP3PCHARMM'
		self.total_atom = 3
		self.total_atom_type = 2
		self.atom = pd.DataFrame({'mass': [15.9994, 1.008],								  
								  'epsilon': [0.1521, 0.0460],
								  'sigma': [3.1507, 0.4000],
								  'charge': [-0.834, 0.417]},
								  index = ['O_H', 'H_O'])
		# self.atom.index = ['O', 'H'] # add row name
		self.total_bond_type = 1
		self.bond = pd.DataFrame({'k': [450.0], 'r0':[0.9572]})
		self.bond.index = ['O-H']
		self.total_angle_type = 1
		self.angle = pd.DataFrame({'k': [55.0], 'theta': [104.52]})
		self.angle.index = ['H-O-H']
		self.total_dihedral_type = 0
		self.dihedral = pd.DataFrame({'k': [[]]}, index = [])




class TIP3PPrice:
	''' LAMMPS PriceTIP3P water model '''
	def __init__(self):
		self.name = 'TIP3PPrice'
		self.total_atom = 3
		self.total_atom_type = 2
		self.atom = pd.DataFrame({'mass': [15.9994, 1.008],								  
								  'epsilon': [0.102, 0.0],
								  'sigma': [3.188, 0.0],
								  'charge': [-0.830, 0.415]},
								  index = ['O_H', 'H_O'])
		# self.atom.index = ['O', 'H'] # add row name
		self.total_bond_type = 1
		self.bond = pd.DataFrame({'k': [450.0], 'r0':[0.9572]})
		self.bond.index = ['O-H']
		self.total_angle_type = 1
		self.angle = pd.DataFrame({'k': [55.0], 'theta': [104.52]})
		self.angle.index = ['H-O-H']
		self.total_dihedral_type = 0
		self.dihedral = pd.DataFrame({'k': [[]]}, index = [])

#https://lammps.sandia.gov/threads/msg18731.html
class TIP4P2005:
	def __init__(self):
		self.name = 'TIP4P2005'
		self.total_atom = 3
		self.total_atom_type = 2
		self.atom = pd.DataFrame({'mass': [15.9994, 1.008],								  
								  'epsilon': [0.1852, 0.0],
								  'sigma': [3.1589, 0.0],
								  'charge': [-1.1128, 0.5564]},
								  index = ['O_H', 'H_O'])
		# self.atom.index = ['O', 'H'] # add row name
		self.total_bond_type = 1
		self.bond = pd.DataFrame({'k': [1000000.0], 'r0':[0.9572]}) # rigid TIP4P
		self.bond.index = ['O-H']
		self.total_angle_type = 1
		self.angle = pd.DataFrame({'k': [1000000.0], 'theta': [104.52]}) # rigid TIP4P
		self.angle.index = ['H-O-H']
		self.total_dihedral_type = 0	
		self.dihedral = pd.DataFrame({'k': [[]]}, index = [])

class Methanol:
	''' OPLS methanol model '''
	def __init__(self):
		self.name = 'Methanol'
		self.total_atom = 6
		self.total_atom_type = 4
		self.atom = pd.DataFrame({'mass': [12.0107, 15.9994, 1.008, 1.008],						  
								  'epsilon': [0.066, 0.17, 0.03, 0.0],
								  'sigma': [3.50, 3.12, 2.5, 0.0],
								  'charge': [0.145, -0.683, 0.04, 0.418]},
								   index = ['C_O', 'O_C', 'H_C', 'H_O'])
		# self.atom.index = ['C', 'O', 'H_C', 'H_O'] # add row name
		self.total_bond_type = 3
		self.bond = pd.DataFrame({'k': [320.0, 340.0, 553.0], 
								  'r0':[1.41, 1.09, 0.945]},
								  index = ['C-O', 'C-H', 'O-H'])
		# self.bond.index = ['C-O', 'C-H', 'O-H']
		self.total_angle_type = 3
		self.angle = pd.DataFrame({'k': [33.0, 35.0, 55.0], 
								   'theta': [107.8, 109.5, 108.5]},
								   index = ['H-C-H', 'H-C-O', 'C-O-H'])
		# self.angle.index = ['H-C-H', 'H-C-O', 'C-O-H']	
		self.total_dihedral_type = 1	
		# self.dihedral = pd.DataFrame({'k1':[0.0], 'k2':[0.0], 'k3':[0.45], 'k4':[0.0]},
		self.dihedral = pd.DataFrame({'k': [[0.0, 0.0, 0.45, 0.0]]},
									  index = ['H-C-O-H'])





