# Multiscale sampling (MSS) with VASP and LAMMPS
# Xiaohong Zhang
# Getman Research Group
# Apr 14, 2019
import os, sys
import pandas as pd
import numpy as np
from collections import OrderedDict
import math
import fnmatch
from vasp import VASP



class LmpsToVasp(VASP):
	""" convert LAMMPS file to VASP file """
	def __init__(self, curr_dir):
		poscar_file = os.path.join(curr_dir, '2-lammps/0-add-sol/POSCAR_0.POSCAR') # solvated poscar file
		mss_input = os.path.join(curr_dir, 'input/master_input.txt')
		lmps_input = os.path.join(curr_dir, '2-lammps/prod/input.prod')
		for f in os.listdir(os.path.join(curr_dir, '2-lammps/prod')):
			if fnmatch.fnmatch(f, 'dump.*.lammpstrj'):
				lmps_dump = os.path.join(curr_dir, '2-lammps/prod/', f)

		super().__init__(poscar_file, mss_input)
		self.equil_run = 2000000  # remove first 2 ns
		self.shift_z = 14.0 # add more vacuum space in z direction for vasp simulation 
		self.frames = [] # store e.g. 10 frames into list, include both title and coords
		self.cell_vec = [[0]*3 for i in range(3)]  ## from dump file generated POSCAR with unshifted cell vector
		self.elem_type_list = [] #all element type list for a frame (same for all frames)
		# self.indices is the atom index used for calc HB, this indices cannot be obtained from vac POSCAR or
		# after Jeremy's solvent added, b/c re-ordered the coordinates from POSCAR to LAMMPS
		self.indices = {'surface':[],'adsorbate':[],'solvent':[],
						'Oads':[],'Hads':[],'Osol':[],'Hsol':[]} #need ads list for rigid ads

		## cannot move below statement to above
		self.readLmps(lmps_input, lmps_dump)
		self.setElemInfo() # must call after read lammps




	def readLmps(self, lmps_input, lmps_dump):
		""" Extract 10 frames from LAMMPS dump trajectory of production run """
		with open(lmps_input) as f:
			for line in f:
				if "name string" in line:
					self.name = line.strip().split()[3]
				if "dumpFreq equal" in line:
					self.dump_freq = int(line.strip().split()[3])
				if "thermFreq equal" in line:
					self.therm_freq = int(line.strip().split()[3])					
				if "runStep equal" in line:
					self.final_step = int(line.strip().split()[3])

		frame_interval = (self.final_step - self.equil_run) / self.frame_num   # 300000
		with open(lmps_dump, 'r') as f:
			dump_vec = []
			for i, line in enumerate(f):
				if i == 3:
					self.total_atom = int(line.strip().split()[0]) # 211
					## self.total_atom+9 is lineno for each frame
					self.frame_line = self.total_atom + 9 # 220
					## skip the equilibrium time
					start_line = self.frame_line*(self.equil_run+frame_interval*0.5)/self.dump_freq
					# print(start_line)
				elif 5 <= i  <= 7 :
					dump_vec.append([float(j) for j in line.strip().split()])
					if len(dump_vec) == 3:
						self.setCellVec(dump_vec) 
				elif i > 7 and start_line <= i < start_line+self.frame_line:
					self.frames.append(line.strip().split())
					### read to the last line of this frame
					if i == start_line+self.frame_line - 1:
						start_line += self.frame_line*(frame_interval/self.dump_freq)
						# print(start_line)




	def setCellVec(self, dump_vec):
		""" convert lammps dump raw cell format to poscar cell format """
				xlo_bound = dump_vec[0][0]
		xhi_bound = dump_vec[0][1]
		ylo_bound = dump_vec[1][0]
		yhi_bound = dump_vec[1][1]
		zlo_bound = dump_vec[2][0]
		zhi_bound = dump_vec[2][1]
		xy = dump_vec[0][2] if len(dump_vec[0]) > 2 else 0.0
		xz = dump_vec[1][2] if len(dump_vec[1]) > 2 else 0.0
		yz = dump_vec[2][2] if len(dump_vec[2]) > 2 else 0.0
		
		xlo = xlo_bound - min(0.0,xy,xz,xy+xz)
		xhi = xhi_bound - max(0.0,xy,xz,xy+xz)
		ylo = ylo_bound - min(0.0,yz)
		yhi = yhi_bound - max(0.0,yz)
		zlo = zlo_bound
		zhi = zhi_bound		
		self.cell_vec = [[xhi-xlo,0,0],
						[xy,yhi-ylo,0],
						[xz,yz,zhi-zlo]]



	def setElemInfo(self):
		
		"""only read the first of the 10 selected frames. 
		get the information of each atom element and number.
		Also get a list of each element 
		"""
		frame = self.frames[0:self.frame_line]
		for i in range(9, len(frame)):
			_type = int(frame[i][2])
			elem = self.atom.loc[_type-1, 'elem'].split('_')[0] # inheritant self.atom from VASP(ReadInput)
			self.elem_type_list.append(elem)  ## total of 211 elem type

		coords = self.reorderCoords(frame) 
		self.setIndices(coords) # use coords rather than frame




	def reorderCoords(self, frame):
		""" process one frame at a time, get all info of GROUPED elements
		must call self.getElemInfo(frame) at first in genVASP() function
		to define self.elem_type_list.
		Group coords according to their element type. Previous dump file O,H,H, 
		Now all O are together, all H are together.
		frame is a list of list that is extracted from dump, including header
		"""

		# reordered according to Pt C O H
		coords = pd.DataFrame(columns=['id','mol','type','xs','ys','zs','opt']) #optimization
		#### loop over Pt, C, O,H in order to group the same elements together
		for elem in self.elem['elem']: # loop over elem type 
			for i in range(9, len(frame)):
				## all frames have same order, thus instead of get elem type for each
				## frame, here use the general elem_type_list
				if self.elem_type_list[i-9] == elem: 
					# set the default selective dynamics as F
					coords.loc[len(coords)] = [frame[i][0],frame[i][1],frame[i][2],
											   frame[i][3],frame[i][4],frame[i][5],'F   F   F']
		return coords



	def setIndices(self, coords):
		"""get all group information"""
		# self.indices = {'Osol':[],'Hsol':[],'Oads':[],'Hads':[],'adsorbate':[]}
		# only need to be called once
		surface = [int(i) for i in self.group[self.group['name']=='surface']['type'].item().split()] # 1  
		adsorbate = [int(i) for i in self.group[self.group['name']=='adsorbate']['type'].item().split()] # [2, 3, 4, 5]
		solvent = [int(i) for i in self.group[self.group['name']=='solvent']['type'].item().split()] # [6, 7, 8, 9, 10, 11]
		Oads = [int(i) for i in self.group[self.group['name']=='Oads']['type'].item().split()] # 3
		Hads = [int(i) for i in self.group[self.group['name']=='Hads']['type'].item().split()] # 5
		Osol = [int(i) for i in self.group[self.group['name']=='Osol']['type'].item().split()] # 6 9
		Hsol = [int(i) for i in self.group[self.group['name']=='Hsol']['type'].item().split()] # 7 11

		# print (surface)		
		# print (adsorbate)
		# print (solvent)
		# print (Oads)
		# print (Hads)
		# print (Osol)
		# print (Hsol)


		for i in range(len(coords)):
			_type = int(coords.loc[i,'type'])
			if _type in surface:
				self.indices['surface'].append(i)
			if _type in adsorbate:
				self.indices['adsorbate'].append(i)
			if _type in solvent:
				self.indices['solvent'].append(i)
			if _type in Oads:
				self.indices['Oads'].append(i) 
			if _type in Hads:
				self.indices['Hads'].append(i)
			if _type in Osol:
				self.indices['Osol'].append(i) # coords start from 0
			if _type in Hsol:
				self.indices['Hsol'].append(i)
		# print ('surface  ', self.indices['surface'])
		# print ('\nadsorbate  ', self.indices['adsorbate'])
		# print ('\nsolvent  ', self.indices['solvent'])
		# print ('\nOads  ', self.indices['Oads'])
		# print ('\nHads  ', self.indices['Hads'])
		# print ('\nOsol  ', self.indices['Osol'])
		# print ('\nHsol  ', self.indices['Hsol'])

	def calcHB(self, coords):
		
		""" calculate the HB between adsorbates and solvent molecules 
			process one frame's coords at one time
		"""
		relax_sol = {'sol_acc':[], 'sol_don':[]} # index starts from 0

		for i in self.indices['Oads']: # all index in coords is based on start from 0
			p_Oads = [float(coords.loc[i,'xs']), float(coords.loc[i,'ys']), float(coords.loc[i,'zs'])]
			# work for different adsorbate coverage
			mol_ads_type = coords.loc[i,'mol']
			mol_ads = coords.loc[coords['mol'] == mol_ads_type].index.tolist() # index for a whole molecule 
			# print(mol_ads) #int,[27, 28, 29, 45, 46, 47, 90, 91, 92, 93, 94, 95, 96]

			for j in self.indices['Osol']:
				p_Osol = [float(coords.loc[j,'xs']), float(coords.loc[j,'ys']), float(coords.loc[j,'zs'])]
				d_oo = self.getDist(p_Oads, p_Osol)
				# print("%d\t%d: %f" % (i,j,dist))
				if d_oo > 0.0 and d_oo <= 3.5:					
					mol_sol_type = coords.loc[j,'mol']
					mol_sol = coords.loc[coords['mol'] == mol_sol_type].index.tolist() # index for a whole molecule,start 0

					######################################################
					##### Calc HB-ed H2O mol when solvent as acceptor (ads as donor) #####
					for k in mol_ads:
						if k in self.indices['Hads']: # 95，96
							p_Hads = [float(coords.loc[k,'xs']), float(coords.loc[k,'ys']), float(coords.loc[k,'zs'])]
							d_hyd = self.getDist(p_Oads, p_Hads)  # doner H and its bonded O/N
							if d_hyd <= 1.2:
								d_oh = self.getDist(p_Osol, p_Hads) # dist of O...H, i.e., HB
								if d_oh <= 2.5:
									a_ooh = self.getAngle(d_oo, d_hyd, d_oh)
									a_oho = self.getAngle(d_hyd, d_oh, d_oo)
									if a_ooh <= 30 and a_oho >= 120:
										relax_sol['sol_acc'].append(mol_sol)

					######################################################
					##### Calc HB-ed H2O mol when solvent as donor #####
					for k in mol_sol: # e.g., H2O has two H
						if k in self.indices['Hsol']: # 95，96
							p_Hsol = [float(coords.loc[k,'xs']), float(coords.loc[k,'ys']), float(coords.loc[k,'zs'])]
							d_hyd = self.getDist(p_Osol, p_Hsol)  # hydroxyl group, i.e., doner H and its bonded O/N
							if d_hyd <= 1.2:
								d_oh = self.getDist(p_Oads, p_Hsol) # dist of O...H, i.e., HB
								if d_oh <= 2.5:
									a_ooh = self.getAngle(d_oo, d_hyd, d_oh)
									a_oho = self.getAngle(d_hyd, d_oh, d_oo)
									if a_ooh <= 30 and a_oho >= 120:
										relax_sol['sol_don'].append(mol_sol)

		return relax_sol





	def fracToCartesian(self, p):
		xa = self.cell_vec[0][0]
		xb = self.cell_vec[0][1]
		xc = self.cell_vec[0][2]
		ya = self.cell_vec[1][0]
		yb = self.cell_vec[1][1]
		yc = self.cell_vec[1][2]
		za = self.cell_vec[2][0]
		zb = self.cell_vec[2][1]
		zc = self.cell_vec[2][2]
		m = 1.0 # self.multiplier
	
		###must return array in order to use lambda in getDist()
		cp = m * np.array( [p[0]*xa + p[1]*ya + p[2]*za, 
		                    p[0]*xb + p[1]*yb + p[2]*zb, 
		                    p[0]*xc + p[1]*yc + p[2]*zc] )
		return cp



	def getDist(self, p0, p1):
		""" test all images """
		x = p1[0]
		y = p1[1]
		z = p1[2]
		images = np.array([[x,y,z],     [x+1,y,z],      [x-1,y,z],
		                   [x,y+1,z],   [x+1,y+1,z],    [x-1,y+1,z],
		                   [x,y-1,z],   [x+1,y-1,z],    [x-1,y-1,z]]) 

		return min(np.sqrt(sum(map(lambda x:x*x, self.fracToCartesian(p0-p)))) for p in images)



	def getAngle(self, d1, d2, d3): # Note: square of the distances
		return math.degrees(math.acos((d1**2 + d2**2 - d3**2)/(2.0*d1*d2)))




	def genVASP(self, frame_idx, model='full', jobtype='geomopt', queue='workq', version='cpu'):
		""" if generate 10 VASP folders with index "file_id"
			model choices: full, sol, ads
			jobtype choices: singpt, geomopt, cineb, dimer
			queue choices: workq, curium
			version choices: cpu, gpu, neb
		"""
		# frame: list of list. including the frame header. frame is the raw data
		if model not in ['full', 'surface', 'ads', 'sol']:
			sys.exit('\nERROR: wrong poscar model\n') 


		# print(type(frame_idx))
		frame = self.frames[self.frame_line*frame_idx:self.frame_line*(frame_idx+1)] 
		jobname = self.ads_name + '_' + str(frame_idx) + '_' + model

		if model == 'full':
			path = os.path.join('./3-vasp-eint/', str(frame_idx))
			coords = self.reorderCoords(frame)
			relax_sol = self.calcHB(coords) # return a dict
			contcar_file = ''
		else:
			coords = pd.DataFrame()
			relax_sol = {}
			contcar_file = os.path.join('./3-vasp-eint/', str(frame_idx), 'CONTCAR')
			if model == 'surface':
				path = os.path.join('./3-vasp-eint/', str(frame_idx)+'p')
			elif model == 'ads':
				path = os.path.join('./3-vasp-eint/', str(frame_idx)+'a')
			elif model == 'sol':
				path = os.path.join('./3-vasp-eint/', str(frame_idx)+'s')

		self.createFolder(path)
		self.writePOSCAR(path, coords, relax_sol, self.cell_vec, self.shift_z, 
						 self.indices, contcar_file, model, jobtype)
		self.writeINCAR(path, model, jobtype)
		self.writeKPOINTS(path)
		self.writePOTCAR(path, model)
		self.writeSubVASP(path, jobname, queue, version)	#subVasp no need model b/c jobname includes model			



def main():

	curr_dir = sys.argv[1]
	vasp = LmpsToVasp(curr_dir)

	"""command line model choices: full, surface, ads, sol"""
	model = sys.argv[2]  # CHANGE TO ARGPARSE LATER

	if model == 'full':
		for i in range(vasp.frame_num):
			vasp.genVASP(frame_idx=i, model='full', jobtype='geomopt', queue='workq', version='cpu')

	else:
		idx = int(sys.argv[3])
		vasp.genVASP(frame_idx=idx, model=model, jobtype='singpt', queue='workq', version='cpu')		



if __name__ == '__main__':
	main()






