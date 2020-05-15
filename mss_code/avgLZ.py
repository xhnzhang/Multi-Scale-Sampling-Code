# Multiscale sampling (MSS) with VASP and LAMMPS
# Xiaohong Zhang
# Getman Research Group
# Mar 18, 2020
import sys,os
import math



def avgLZ(log_file):

	with open(log_file) as f:
		lz = [] # lammps npt simulation to get lz
		for line in f:
			if line.startswith('Step TotEng'):
				break
		for line in f:
			if line.startswith('Loop'):
				break
			lz.append(line.strip().split()[-1]) # last column
	lz = [float(i) for i in lz]
	return round(sum(lz[2000:])/len(lz[2000:]), 6)


def main():
	
	curr_dir = sys.argv[1]
	for f in os.listdir(os.path.join(curr_dir, '2-lammps/equil')):
		if f.startswith('log') and not f.endswith('.lammps'):
			log_file = os.path.join(curr_dir, '2-lammps/equil', f)
	avg_lz = avgLZ(log_file)
	print(avg_lz)


if __name__== '__main__':
	main()















