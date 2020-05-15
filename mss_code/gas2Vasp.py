# Multiscale sampling (MSS) with VASP and LAMMPS
# Xiaohong Zhang
# Getman Research Group
# Mar 10, 2020
import os, sys
import shutil
from vasp import VASP




class GasToVASP(VASP):

	def __init__(self, curr_dir):
		poscar_file = os.path.join(curr_dir, 'input/POSCAR')
		mss_input = os.path.join(curr_dir, 'input/master_input.txt') # gas phase poscar
		super().__init__(poscar_file, mss_input)
		self.poscar_file = poscar_file
		self.path = os.path.join(curr_dir, '1-vasp-vac/')

	def genVASP(self, model='vac', jobtype='geomopt', queue='workq', version='cpu'):
		self.createFolder(self.path)
		## use the original poscar, do not need to write new poscar
		jobname = self.ads_name + '_' + model
		self.writeINCAR(self.path, model, jobtype)
		self.writeKPOINTS(self.path)
		self.writePOTCAR(self.path, model)
		self.writeSubVASP(self.path, jobname, queue, version)	#subVasp no need model b/c jobname includes model			



def main():

	curr_dir = sys.argv[1]
	gas = GasToVASP(curr_dir) 
	gas.genVASP(model='vac', jobtype='geomopt', queue='workq', version='cpu')
	shutil.copy(gas.poscar_file, os.path.join(curr_dir, '1-vasp-vac/POSCAR'))


if __name__ == '__main__':
	main()


