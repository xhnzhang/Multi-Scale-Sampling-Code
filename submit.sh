#!/bin/bash
# usage: nohup bash submit.sh > log.txt &

cwd="$(pwd)"
############## step 1.1 create files to run gas phase vasp ##############
module purge
module add anaconda3/4.2.0 gcc/7.1.0 # needs python3.6



############## step 1.2 run gas phase vasp optimiztation ##############
python /home/xiaohoz/bin/mss_code/gas2Vasp.py ${cwd} # qsub inside python code
cd ./1-vasp-vac/
qsub subvasp.sh
############## check if vasp job converged ######################
sleep 10m
converged=$(vaspgeom |  grep 'converged' | wc -l)
while [ "$converged" -eq 0 ]; do
 	sleep 10m
 	converged=$(vaspgeom |  grep 'converged' | wc -l)
done 


############## step 1.3 run spe to generate chargecar ##############
mkdir -p ./singpt_for_charge/

cp CONTCAR ./singpt_for_charge/POSCAR
cp INCAR KPOINTS POTCAR subvasp.sh ./singpt_for_charge/
cd ./singpt_for_charge/
sed -i 's/T   T   T/F   F   F/g' POSCAR  
sed -i '/^LCHARG/s/\S*$/.TRUE./'  INCAR
sed -i '5iLAECHG   = .TRUE.' INCAR
sed -i '/^NELM /s/\S*$/1000/'  INCAR
sed -i '/^NSW/s/\S*$/0/'  INCAR

qsub subvasp.sh
############# check if vasp job finished ######################
while [ ! -f ./*.o[0-9]* ]; do sleep 10m; done


############## step 1.4 run ddec and get ***charges.xyz file ##############
mkdir -p ./ddec/
mv AECCAR0 AECCAR2 CHGCAR POTCAR ./ddec/
cp /home/xiaohoz/bin/DDEC/job_control.txt /home/xiaohoz/bin/DDEC/subDDEC.sh ./ddec/
cd ./ddec/
qsub subDDEC.sh
############# check if ddec job finished ######################
while [ ! -f ./*.o[0-9]* ]; do sleep 5m; done
cd ${cwd}


############## step 2.1 run jeremy's code to add solvent ##############
mkdir -p ./2-lammps/0-add-sol/
cp ./input/master_input.txt ./2-lammps/0-add-sol/  # in order to read master_input.txt 
cp ./1-vasp-vac/CONTCAR ./2-lammps/0-add-sol/POSCAR
cd ./2-lammps/0-add-sol/
## note: make sure surface atom# is correct, e.g., vac atom Pt C O H, then surface atom# is 4
{ /home/xiaohoz/bin/mcpliq/mcpliq; } > out.txt 2>&1
rm -f core.*
cd ${cwd}



############## step 2.2 run mss code to generate LAMMPS input files ##############
python /home/xiaohoz/bin/mss_code/vasp2Lmps.py ${cwd}

############## step 2.3 run lammps  ##############
cd ./2-lammps/equil/
qsub sublammps.sh
############# check if equil finished ######################
while [ ! -f ./*.o[0-9]* ]; do sleep 10m; done
cd ${cwd}

#call python code to average lz, redirect 
lz=$( (python /home/xiaohoz/bin/mss_code/avgLZ.py ${cwd}) 2>&1)

############# modify lz for prod and ti ######################
sed -i "/zhi/s/[^ ]*[^ ]/${lz}/2"  ./2-lammps/prod/data.*
sed -i "/zhi/s/[^ ]*[^ ]/${lz}/2"  ./2-lammps/ti/data.*
cd ./2-lammps/ti/
qsub sublammps.sh  # must go inside folder to submit
cd ${cwd}
cd ./2-lammps/prod/
qsub sublammps.sh  # need to wait until finish
############# check if equil finished ######################
while [ ! -f ./*.o[0-9]* ]; do sleep 10m; done
cd ${cwd}

############## step 3 run vasp  ##############
python /home/xiaohoz/bin/mss_code/lmps2Vasp.py ${cwd} full
dir_arr=($(find "${cwd}/3-vasp-eint" -mindepth 1 -type d))
for dir in ${dir_arr[@]}; do cd ${dir}; qsub subvasp.sh; cd ${cwd}; done

remain_dir_arr=("${dir_arr[@]}")
while [ -n "$dir_arr" ]; do # while not empty 
	for dir in ${dir_arr[@]}; do cd ${dir};
		i=$(awk -F/ '{print $NF}' <<< ${dir})
		converged=$(vaspgeom | grep 'converged' | wc -l)
		if [ "$converged" -eq 1 ]; then 
			#call python to generate partial folders, pass folder index $i to generate ${i}a ${i}s#
			cd ${cwd}
			if [ $i -eq 0 ]; then # run surface vasp
				python /home/xiaohoz/bin/mss_code/lmps2Vasp.py ${cwd} surface $i
				cd ${cwd}/3-vasp-eint/${i}p/
				qsub subvasp.sh
				cd ${cwd}
			fi

			python /home/xiaohoz/bin/mss_code/lmps2Vasp.py ${cwd} ads $i
			python /home/xiaohoz/bin/mss_code/lmps2Vasp.py ${cwd} sol $i
			cd ${cwd}/3-vasp-eint/${i}a/
			qsub subvasp.sh
			cd ${cwd}/3-vasp-eint/${i}s/
			qsub subvasp.sh
			remain_dir_arr=( "${remain_dir_arr[@]/${dir}}" )
		fi
	done
	dir_arr=("${remain_dir_arr[@]}")

	if [ "$dir_arr" ]; then # if still full job running, keep checking
		sleep 30m
	fi
done

		


