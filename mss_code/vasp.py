# Multiscale sampling (MSS) with VASP and LAMMPS
# Xiaohong Zhang
# Getman Research Group
# Mar 5, 2020


import os
import pandas as pd
from readInput import ReadInput


class VASP(ReadInput):
	## derive from ReadInput when used for generating VASP files from vacuum POSCAR

	def __init__(self, poscar_file, mss_input):
		super().__init__(poscar_file, mss_input) 
		self.setD2Correction() # set D2 correction used in VASP INCAR file

	def setD2Correction(self):
		""" 
		Grimme D2 correction in VASP
		"""
		self.d2 = pd.DataFrame(columns = ['elem', 'vdw_c6', 'vdw_r0', 'rwigs'])
		self.d2.loc[len(self.d2)] = ['Pt', '42.440', '1.750', '1.300']
		self.d2.loc[len(self.d2)] = ['C',   '1.750', '1.452', '0.770']
		self.d2.loc[len(self.d2)] = ['O',   '0.700', '1.342', '0.730']
		self.d2.loc[len(self.d2)] = ['H',   '0.140', '1.002', '0.320']
		self.d2.loc[len(self.d2)] = ['N',   '1.230', '1.397', '0.750']


	# vac2VASP doesn't need writePOSCAR()
	def writePOSCAR(self, path, coords, relax_sol, # coords and relax_sol to generate full poscar file
					cell_vec, shift_z, indices, # cell_vec read from dump file
					contcar_file, # to generate partial system
					model='full', jobtype='geomopt'):
		""" 
		elem is a dict 
		use coords information of one frame, and the information of
		relaxed adsorbated to write POSCAR used for VASP
		### only work for orthoganal z direction, if triclinic, modify this code

		200227 update, only checked for geomopt and single point, if run
		other simulation e.g. neb, need to modify code
		"""

		### 200320 write partial system vasp files

		if model in ['surface', 'ads', 'sol']:
			contcar_coords = []
			with open(contcar_file) as inf:
				for i, line in enumerate(inf):
					if len(line.strip().split()) == 0:
						break
					if i >= 8:
						if i == 8 and line.strip().split()[0].isalpha(): # whether this line is "direct" or "cartesian"
							contcar_coords.append(next(inf).strip().split())
						else:
							contcar_coords.append(line.strip().split())


		_coord = []
		with open(os.path.join(path, 'POSCAR'), 'w') as f:

			if model == 'full':
				elem = self.elem['elem']
				num = self.elem['num']

			elif model == 'surface':
				elem = self.elem_surface['elem']
				num = self.elem_surface['num']
				for idx in indices['surface']:
					_coord.append(contcar_coords[idx][0:3])
			elif model == 'ads':
				elem = self.elem_surface['elem'] + self.elem_ads['elem']
				num = self.elem_surface['num'] + self.elem_ads['num']
				for idx in (indices['surface'] + indices['adsorbate']):
					_coord.append(contcar_coords[idx][0:3])
			elif model == 'sol':
				elem = self.elem_surface['elem'] + self.elem_sol['elem']
				num = self.elem_surface['num'] + self.elem_sol['num']
				for idx in indices['surface'] + indices['solvent']:
					_coord.append(contcar_coords[idx][0:3])				
		
			# write tile and also used for element
			for e in elem:
				f.write('{} '.format(e))
			f.write('\n  {}\n'.format('1.0')) # write multiplier
			### below is new cell_vac from dump file, not from original poscar file
			f.write('{:>12.8f} {:>12.8f} {:>12.8f}\n'.format(cell_vec[0][0],cell_vec[0][1],cell_vec[0][2]))
			f.write('{:>12.8f} {:>12.8f} {:>12.8f}\n'.format(cell_vec[1][0],cell_vec[1][1],cell_vec[1][2]))
			f.write('{:>12.8f} {:>12.8f} {:>12.8f}\n'.format(cell_vec[2][0],cell_vec[2][1],cell_vec[2][2]+shift_z))				
			# write elem numbers
			for n in num:
				f.write('{} '.format(n))
			f.write('\nSelective dynamics\n')
			f.write('Direct\n')

			if model == 'full':
				# convert relax_sol dict value into list
				relax_sol_list = [i for key in relax_sol for val in relax_sol[key] for i in val]
				# print(relax_list)
				for i in range(len(coords)):
					x = float(coords.loc[i,'xs'])
					y = float(coords.loc[i,'ys'])
					z = float(coords.loc[i,'zs'])*cell_vec[2][2]/(cell_vec[2][2]+shift_z)
	
					if jobtype == 'geomopt' and (i in indices['adsorbate'] or i in relax_sol_list):
						# relax adsorbate and hydrogen bonded solvent molecules
							opt = 'T   T   T'
					else:
						opt = coords.loc[i,'opt'] # 200117 check ad.xlsx for coords information
					f.write('{:>12.8f} {:>12.8f} {:>12.8f}  {}\n'.format(x,y,z,opt))
			else: # other models
				f.write('   F   F   F\n'.join(['   '.join(i[0:3]) for i in _coord]))
				f.write('   F   F   F')


	def writeINCAR(self, path, model='full', jobtype='geomopt', images='1'):
		"""
		please modify the code if you do NEB or DIMER
		"""
		if jobtype == 'singpt':
			nelm = '1000'
			nsw  = '0'
			ediff = '1e-5'
			ediffg = '-0.03'
			potim = '0.35'
			ibrion = '1'
		elif jobtype == 'geomopt':
			nelm = '100'
			nsw  = '500'
			ediff = '1e-5'
			ediffg = '-0.03'
			potim = '0.35'
			ibrion = '1'			
		elif jobtype == 'cineb':
			nelm = '100'
			nsw  = '100'
			ediff = '1e-5'
			ediffg = '-0.5'
			potim = '0.35'
			ibrion = '1'			
		elif jobtype == 'dimer':
			nelm = '100'
			nsw  = '100'
			ediff = '1e-7'
			ediffg = '-0.03'
			potim = '0.0'
			ibrion = '3'	

		if model in ['full', 'vac']:
			elem = self.elem['elem']
		elif model == 'surface':
			elem = self.elem_surface['elem']
		elif model == 'ads':
			elem = self.elem_surface['elem'] + self.elem_ads['elem']
		elif model == 'sol':
			elem = self.elem_surface['elem'] + self.elem_sol['elem']

		with open(os.path.join(path, 'INCAR'), 'w') as f:
			f.write('{:<8} = {}\n'.format('System', self.ads_name))
			f.write('{:<8} = 1\n'.format('NWRITE'))			
			f.write('{:<8} = .FALSE.\n'.format('LWAVE'))
			f.write('{:<8} = .FALSE.\n'.format('LCHARG'))
			f.write('{:<8} = .FALSE.\n'.format('LVTOT'))
			f.write('\nElectronic Relaxation\n')
			f.write('{:<8} = 400\n'.format('ENCUT'))
			f.write('{:<8} = Fast\n'.format('ALGO'))
			f.write('{:<8} = 0\n'.format('ISMEAR'))
			f.write('{:<8} = 0.100\n'.format('SIGMA'))
			f.write('{:<8} = Accurate\n'.format('PREC'))
			f.write('{:<8} = Auto\n'.format('LREAL'))
			f.write('{:<8} = '.format('ROPT'))
			for i in range(len(elem)):
				f.write('2e-4 ')

			f.write('\n{:<8} = 0\n'.format('ISTART'))
			f.write('{:<8} = {}\n'.format('NELM', nelm))
			f.write('{:<8} = -8\n'.format('NELMDL'))
			f.write('{:<8} = {}\n'.format('EDIFF', ediff))
			f.write('{:<8} = 1\n'.format('ISPIN'))

			f.write('\nIonic Relaxation\n')
			f.write('{:<8} = {}\n'.format('NSW', nsw))
			f.write('{:<8} = 2\n'.format('ISIF'))
			f.write('{:<8} = {}\n'.format('IBRION', ibrion))
			f.write('{:<8} = 10\n'.format('NFREE'))
			f.write('{:<8} = {}\n'.format('POTIM', potim))
			f.write('{:<8} = {}\n'.format('EDIFFG', ediffg))

			f.write('\nDispersion\n')
			f.write('{:<8} = .TRUE.\n'.format('LVDW'))
			f.write('{:<8} = '.format('VDW_C6'))
			for e in elem:
				#200117, get pd value rather than series object
				f.write(self.d2[self.d2['elem']==e]['vdw_c6'].item() + ' ')
			f.write('\n{:<8} = '.format('VDW_R0'))
			for e in elem:
				f.write(self.d2[self.d2['elem']==e]['vdw_r0'].item() + ' ')			
			f.write('\n\nDensity of States\n')
			f.write('{:<8} = '.format('RWIGS'))
			for e in elem:
				f.write(self.d2[self.d2['elem']==e]['rwigs'].item() + ' ')	

			f.write('\n')
			if jobtype in ['singpt', 'geomopt']:
				f.write('\nParallel\n')
				f.write('{:<8} = 4\n'.format('NPAR'))
				f.write('{:<8} = .TRUE.\n'.format('LPLANE'))
				f.write('{:<8} = 10\n'.format('NSIM'))
			elif jobtype == 'cineb':
				f.write('\nClimbing Image-Nudged Elastic Band\n')
				f.write('{:<8} = {}\n'.format('IMAGES', images))
				f.write('{:<8} = .TRUE.\n'.format('LCLIMB'))
			elif jobtype == 'dimer':
				f.write('\n{:<8} = 2\n'.format('ICHAIN'))
				f.write('{:<8} = 2\n'.format('IOPT'))
				f.write('\nDimer Method\n')
				f.write('{:<8} = 5e-3\n'.format('DdR'))
				f.write('{:<8} = 1\n'.format('DRotMax'))
				f.write('{:<8} = 0.01\n'.format('DFNMin'))
				f.write('{:<8} = 1.0\n'.format('DFNMax'))
			



			
	def writeKPOINTS(self, path, kpts=[7,7,1]):
		with open(os.path.join(path, 'KPOINTS'), 'w') as f:
			f.write('Automatic\n')
			f.write('0\n')
			f.write('Gamma\n')
			f.write('{} {} {}\n'.format(kpts[0],kpts[1],kpts[2]))
			f.write('0. 0. 0.\n')



	def writePOTCAR(self, path, model='full'):
		if model in ['full', 'vac']:
			elem = self.elem['elem']
		elif model == 'surface':
			elem = self.elem_surface['elem']
		elif model == 'ads':
			elem = self.elem_surface['elem'] + self.elem_ads['elem']
		elif model == 'sol':
			elem = self.elem_surface['elem'] + self.elem_sol['elem']

		for e in elem:
			if not os.path.isfile('/curium/VASP/PPs/120423/PBE/'+e+'/POTCAR'):
				sys.exit('\nERROR: Incorrect element symbol: ' + e + '\n')

		with open(os.path.join(path, 'POTCAR'), 'w') as f:
			for e in elem:
				with open('/curium/VASP/PPs/120423/PBE/'+e+'/POTCAR', 'r') as inf:
					f.write(inf.read()) # delete previous 'r'



	def writeSubVASP(self, path, jobname, queue, version, images='1'):
		""" 
		user is the username on Palmetto, e.g., xiaohoz
		jobname=adsorbate_name from master_input.txt + loop number

		######### need to modify code if run neb or dimer
		"""
		if queue == 'workq':
			walltime = '72:00:00'
		elif queue == 'curium':
			walltime = '336:00:00'
		else:
			sys.exit('\nERROR: Incorrect queue choice.\n')

		if version == 'cpu':
			resource = 'select=1:ncpus=20:mpiprocs=20:mem=120gb,'
			module = 'intel/16.0 openmpi/1.10.3'
			exe_path = '/curium/VASP/vasp.5.4.4-cpu-multi/vasp_std'
		elif version == 'gpu':
			resource = 'select=1:ncpus=20:mpiprocs=20:ngpus=2:gpu_model=k40:mem=120gb,'
			module = 'intel/16.0 cuda-toolkit/7.5.18'
			exe_path = '/curium/VASP-GPU5.4.1.05Feb16/executable/vasp_gpu_ph12p'
		elif version == 'neb':
			resource = 'select=' + images + ':ncpus=20:mpiprocs=20:chip_type=e5-2680v3:mem=120gb,'
			module = 'intel/13.0 openmpi/1.8.1'
			exe_path = '/curium/SOFTWARE/executables/vaspph10'
		else:
			sys.exit('\nERROR: Incorrect VASP version choice.\n')


		with open(os.path.join(path, 'subvasp.sh'), 'w') as f:
			f.write('#!/bin/bash\n')
			f.write('#PBS -N {}\n'.format(jobname))			
			f.write('#PBS -l {}walltime={}\n'.format(resource, walltime))
			f.write('#PBS -q {}\n'.format(queue))
			f.write('#PBS -j oe\n')
			f.write('#PBS -m abe\n')
			f.write('#PBS -M {}@g.clemson.edu\n'.format(self.user))

			f.write('\necho "START ---------------------"\n')
			f.write('qstat -xf $PBS_JOBID\n')
			f.write('\nmodule purge\n')
			f.write('export MODULEPATH=$MODULEPATH:/software/experimental\n')
			f.write('module add {}\n'.format(module))
			f.write('export OMP_NUM_THREADS=1\n')

			if version == 'gpu':
				f.write('\nmkdir /tmp/nvidia-mps\n')
				f.write('export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps\n')
				f.write('mkdir /tmp/nvidia-log\n')
				f.write('export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log\n')
				f.write('nvidia-cuda-mps-control -d\n')


			f.write('\ncd $PBS_O_WORKDIR\n')
			f.write('mpirun -n 20 {}\n'.format(exe_path))
			f.write('\nrm -f core.*\n')
			f.write('echo "FINISH --------------------"\n')




