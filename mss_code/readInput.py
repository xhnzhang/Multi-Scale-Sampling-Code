# Multiscale sampling (MSS) with VASP and LAMMPS
# Xiaohong Zhang
# Getman Research Group
# Mar 5, 2020
import sys,os
import pandas as pd
import solvent

class ReadInput(object):


	def __init__(self, poscar_file, mss_input):
		self.readPOSCAR(poscar_file)
		self.readMSSinput(mss_input)
		self.groupAtom()



	def readPOSCAR(self, poscar_file):
		""" read VASP POSCAR/CONTCAR file """
		self.elem = {} # vac or solvated poscar atoms (pt+ads+h2o), set in readPOSCAR()
		self.cell_vec = [] # set in readposcar()
		self.old_coords = [] # set in readposcar()		

		with open(poscar_file) as f:
			flag = False
			for i, line in enumerate(f):
				if len(line.strip().split()) == 0:
					break
				if i == 0:
					title = line.strip().split() # better make this title as the element list
				elif i == 1:
					self.multiplier = float(line.strip().split()[0])
				elif 2 <= i  < 5:
					self.cell_vec.append([float(j) for j in line.strip().split()])
				elif i == 5:               
					if line.strip().split()[0].isdigit(): # check if provide element name
						self.elem['elem'] = title 
						self.elem['num'] = [int(j) for j in line.strip().split()]
					else:
						self.elem['elem'] = line.strip().split()
						self.elem['num'] = [int(j) for j in next(f).strip().split()]

					pattern = next(f).strip().split()[0]
					if pattern.lower().startswith('s'):# check if this line is Selective Dyanmics
						self.coord_type = next(f).strip().split()[0] 
					else:
						self.coord_type = pattern
					if self.coord_type[0].lower() not in ['c', 'd']: # check if cartesian or direct 
						sys.exit('\nERROR READING POSCAR: please check coordinate type\n') 
					flag = True
					####################################
				elif flag and len(line.strip().split()) > 0:
					self.old_coords.append(line.strip().split()) 


	def readMSSinput(self, mss_input):
		""" read MSS modeling input file """
		self.elem_surface = {'elem':[], 'num':[]}
		self.elem_ads = {'elem':[], 'num':[]}
		self.elem_sol = {'elem':[], 'num':[]}
		self.atom = pd.DataFrame() # all atom params, set in readMSSinput()
		self.bcoeff = pd.DataFrame()  # bond coeff, set in readMSSinput()
		self.acoeff = pd.DataFrame()  # angle coeff, set in readMSSinput()
		self.dcoeff = pd.DataFrame()  # dihedral coeff, set in readMSSinput()		


		with open(mss_input) as f:
			## 190508 add the group information. group: surface, adsorbate,...,water,..,methanol....
			vac = [None]*8   ### all vacuum atom information [‘group’，'name','elem','num','mass','epsLJ','sigLJ','charge']

			for line in f:
				if line.startswith('Number of different types of fluid molecules'):
					sol_type = int(line.strip().split()[-1])
					self.solvent = pd.DataFrame(index = range(sol_type)) 
					for i in range(sol_type):
						info = next(f).strip().split()
						self.solvent.loc[i,'name'] = info[-1].split('/')[-1].split('.')[0]
						self.solvent.loc[i,'num'] = int(info[0])
				elif line.startswith('user'):
					self.user = line.strip().split()[1]
				elif line.startswith('Adsorbate_name'):
					self.ads_name = line.strip().split()[1]
				elif line.startswith('LMPS_temperature'):
					self.temperature = float(line.strip().split()[1])
				elif line.startswith('LMPS_pressure'):
					self.pressure = float(line.strip().split()[1])
				## make sure Total_vac_types line in MSS_input is before Surface_atom_type
				elif line.startswith('Total_vac_types'):
					self.vac_type = int(line.strip().split()[1])  # total number of vac atom types
				elif line.startswith('Surface_atom_type'):
					self.surface_type = [int(i) for i in line.strip().split()[1:]] # a list [1]
					group = []
					for i in range(self.vac_type):
						if (i+1) in self.surface_type:
							group.append('surface')
						else:
							group.append('adsorbate')
					vac[0] = group
				elif line.startswith('Water_type') and line.strip().split()[1].lower() != 'none':
					self.water_type = line.strip().split()[1] 
				elif line.startswith('Frame_num'):
					self.frame_num = int(line.strip().split()[1])
				elif line.startswith('Atom_element'):
					## 190509 modify self.atom['name'] with full element_name
					_elem = line.strip().split()[1:]
					vac[1] = [i+'_'+j for i,j in zip(_elem, vac[0])]
					vac[2] = _elem
               
				elif line.startswith('Atom_number'):
					vac[3] = [int(i) for i in line.strip().split()[1:]] # vacuum element number
				elif line.startswith('Atom_mass'):
					vac[4] = [float(i) for i in line.strip().split()[1:]] # vacuum element mass
				elif line.startswith('LJ_epsilon'):
					vac[5] = [float(i) for i in line.strip().split()[1:]] # vacuum element eps
				elif line.startswith('LJ_sigma'):
					vac[6] = [float(i) for i in line.strip().split()[1:]] # vacuum element sigma     
				

		#[‘group’，'name','elem','num','mass','epsLJ','sigLJ','charge']
		vac[7] = [1000]*self.vac_type

		sol = [[] for i in range(8)] #[‘group’，'name','elem','num','mass','epsLJ','sigLJ','charge']

		bd, agl, dhd = ( [[] for i in range(5)] for j in range(3))

		for i in range(len(self.solvent)):  # i loop twice with water and methanol
			name = self.solvent.loc[i, 'name']

			_model = self.water_type if name == 'water' else name
			self.solvent.loc[i,'obj'] = solvent.readSol(_model)
			s = self.solvent.loc[i,'obj']
			self.solvent.loc[i,'model'] = s.name

			for j, key in enumerate([[name]*s.total_atom_type, [i+'_'+s.name for i in s.atom.index], s.atom.index,
                                 [1000]*s.total_atom_type, s.atom['mass'], 
                                 s.atom['epsilon'], s.atom['sigma'], s.atom['charge']]): 
				sol[j].extend(key)  ##all solvent atom information

			for j, key in enumerate([[name]*s.total_bond_type, [s.name]*s.total_bond_type, list(s.bond.index), 
                                   s.bond['k'], s.bond['r0']]):
				bd[j].extend(key)
			for j, key in enumerate([[name]*s.total_angle_type, [s.name]*s.total_angle_type, list(s.angle.index), 
                                   s.angle['k'], s.angle['theta']]):
				agl[j].extend(key)           
			for j, key in enumerate([[name]*s.total_dihedral_type, [s.name]*s.total_dihedral_type, list(s.dihedral.index), 
                                   s.dihedral['k']]):
				dhd[j].extend(key)

		# print(self.solvent)

		#### fill atom params df information
		for (i, key) in enumerate(['group', 'name','elem','num','mass','epsLJ','sigLJ','charge']):
			self.atom[key] = vac[i] + sol[i]

		### add another param: atom type. Now the C bonded H is not updated yet, will update in the calcBond()
		###'name','elem','num','mass','epsLJ','sigLJ','charge','type']
		for i in range(len(self.atom)):
			self.atom.loc[i,'type'] = i+1
		# print(self.atom)
		#### fill bcoeff, acoeff, dcoeff df information
		for (i, key) in enumerate(['name','model','type','k','r0']):
			self.bcoeff[key] = bd[i]
		for (i, key) in enumerate(['name','model','type','k','theta']):
			self.acoeff[key] = agl[i]      
		for (i, key) in enumerate(['name','model','type','k']):
			self.dcoeff[key] = dhd[i]

		# print(self.bcoeff)
		# print(self.acoeff)
		# print(self.dcoeff)

		# print(self.atom)
		## Run partial system vasp optimization
		for index, row in self.atom.iterrows():
			e = row['elem'].split('_')[0]
			n = int(row['num'])
			if (index+1) in self.surface_type:
				if not e in self.elem_surface['elem']:
					self.elem_surface['elem'].append(e)
					self.elem_surface['num'].append(n)
				else:
					self.elem_surface['num'][-1] += n # repeated elements in poscar, eg., H_C, H_O

			elif (index+1) <= self.vac_type:
				if not e in self.elem_ads['elem']:
					self.elem_ads['elem'].append(e)
					self.elem_ads['num'].append(n)
				else:
					self.elem_ads['num'][-1] += n # repeated elements in poscar, eg., H_C, H_O
		_l = len(self.surface_type)
		for i, _e in enumerate(self.elem['elem'][_l:]):
			if _e not in self.elem_ads['elem']:
				print('no')
				self.elem_sol['elem'].append(_e)
				self.elem_sol['num'].append(self.elem['num'][_l+i])
			else:
				diff = self.elem['num'][_l+i] - self.elem_ads['num'][self.elem_ads['elem'].index(_e)]
				if diff > 0:
					self.elem_sol['elem'].append(_e)
					self.elem_sol['num'].append(diff)

		# print(self.elem)
		# print(self.elem_surface)
		# print(self.elem_ads)
		# print(self.elem_sol)





	def groupAtom(self):
		self.group = pd.DataFrame(columns=['name','type']) # group atoms to use in LMPS input and later VASP, set in self.groupAtom()


		solvent_type = list(range(self.vac_type+1, len(self.atom)+1))
		slab_type = list(range(1, self.vac_type+1))
		ads_type = [i for i in slab_type if i not in self.surface_type]
		# print(slab_type)
		# print(self.surface_type)
		
		self.group.loc[len(self.group)] = ['surface', ' '.join(map(str,self.surface_type))]
		self.group.loc[len(self.group)] = ['adsorbate', ' '.join(map(str,ads_type))]
		self.group.loc[len(self.group)] = ['solvent', ' '.join(map(str,solvent_type))]
		self.group.loc[len(self.group)] = ['slab', ' '.join(map(str,slab_type))]

		 
		_start = self.vac_type
		for i in range(len(self.solvent)):  # i loop twice with water and methanol
			s = self.solvent.loc[i,'obj']
			### Add each solvent atom type, cannot just add list
			_list = list(range(_start+1, _start+1+s.total_atom_type))
			self.group.loc[len(self.group)] = [self.solvent.loc[i,'name'], ' '.join(map(str, _list))]
			# print(list(range(_start+1, _start+1+s.total_atom_type))) #[6, 7] \n [8, 9, 10, 11]
			_start += s.total_atom_type 

		Osol, Hsol, Oads, Hads = ([] for i in range(4))
		# for i in range(len(self.atom)):
		for i in range(len(self.atom)):
			r = self.atom.iloc[i]
			_type = int(r['type'])
			if r['elem'].split('_')[0].upper() in ['O', 'N']:
				if _type in ads_type:
					Oads.append(_type)        
				elif _type in  solvent_type:
					Osol.append(_type)
			elif r['elem'].split('_')[0].upper() == 'H':
				if r['elem'].split('_')[1].upper() != 'C':
					if _type in ads_type:
						Hads.append(_type)
					elif _type in solvent_type:              
 						Hsol.append(_type)

		self.group.loc[len(self.group)] = ['Oads', ' '.join(map(str,Oads))]
		self.group.loc[len(self.group)] = ['Hads', ' '.join(map(str,Hads))]                  
		self.group.loc[len(self.group)] = ['Osol', ' '.join(map(str,Osol))]
		self.group.loc[len(self.group)] = ['Hsol', ' '.join(map(str,Hsol))]


	# below function is used in multiple derived classes
	def createFolder(self, dirname):
		try:
			if not os.path.exists(dirname):
				os.makedirs(dirname)
		except OSError:
			print('Error: Creating directory. ' +  dirname)







