# Multiscale sampling (MSS) with VASP and LAMMPS
# Xiaohong Zhang
# Getman Research Group
# Mar 10, 2019

import sys, os
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from readInput import ReadInput



class VaspToLmps(ReadInput):
   """
   read POSCAR/CONTCAR and extract information
   """
   def __init__(self, curr_dir):
      poscar_file = os.path.join(curr_dir, '2-lammps/0-add-sol/POSCAR_0.POSCAR') # solvated poscar file
      mss_input = os.path.join(curr_dir, 'input/master_input.txt') 
      ddec_file = os.path.join(curr_dir, '1-vasp-vac/singpt_for_charge/ddec/DDEC6_even_tempered_net_atomic_charges.xyz')
      super().__init__(poscar_file, mss_input) # solvated poscar file
      self.curr_dir = curr_dir
      self.coords = pd.DataFrame()  # new reordered coords, set in reorderCoords()
      ### process from coords
      self.bonds = [] #set in getBondedList(self)
      self.angles= [] #set in getBondedList(self)
      self.dihedrals = [] #set in getBondedList(self)
      self.ddec = [] # set in self.readDDEC()
      self.readDDEC(ddec_file)
      self.getBondedList()



   def readDDEC(self, ddec_file):

      with open(ddec_file) as f:
         for i, line in enumerate(f):
            if i == 0:
               num = int(line.strip().split()[0])
            elif 2 <= i  < num+2:
               charge = float(line.strip().split()[-1])
               self.ddec.append(charge)




   def reorderCoords(self):
      temp_vac_coord = [] #store for temporary vac atom
      temp_sol_coord = [] #store for temporary solvent
      cum_vac_atom = np.cumsum(list(self.atom['num'][0:self.vac_type]))   ## [27 30 33 38 40]   
      cum_atom = np.cumsum(self.elem['num'])  ## [ 27  45  90 211]
      count = 0   # match pt-ads vac atom type with info in master_input.txt file
      
      for i in range(len(self.old_coords)):
         x = float(self.old_coords[i][0])   # use float in case fracToCartesian() is not called
         y = float(self.old_coords[i][1])
         z = float(self.old_coords[i][2])
         mol_type = int(self.old_coords[i][-2][1:])  # split #1
         group_type = int(self.old_coords[i][-1])

         if group_type == -1:   # vac atom
            count += 1
            # print('count: ', count)
            # print(np.where(count <= cum_vac_atom))
            idx = min(np.where(count <= cum_vac_atom)[0])
            temp_vac_coord.append([x, y, z, mol_type, group_type, list(self.atom['type'])[idx]])
         else: # sol atom
            start_type = self.vac_type  # index from 0. index solvent atom type, need initialize here
            for j in range(group_type):       # if multi solvent, add index after last sol
               start_type += self.solvent.loc[j,'obj'].total_atom_type
            idx = min(np.where((i+1) <= cum_atom)[0])  

            if self.solvent.loc[group_type, 'name'] == 'water':
               if self.elem['elem'][idx] == 'O':
                  sol_atom_type = start_type + 1
               else:
                  sol_atom_type = start_type + 2
            ############ MODIFY BELOW FOR NON CxHyOz solvent ############
            else: #non-water solvent
               if self.elem['elem'][idx] == 'C':
                  sol_atom_type = start_type + 1
               elif self.elem['elem'][idx] == 'O':
                  sol_atom_type = start_type + 2
               elif self.elem['elem'][idx] == 'H':
                  sol_atom_type = start_type + 3 # decide later for H_C or H_O
 
            ############ try not add atom name and elem in the coords
            temp_sol_coord.append([x, y, z, mol_type, group_type, sol_atom_type]) #['x','y','z','mol','grp','type']
      
      # print("self.elem\n")
      # print(self.elem) #{'elem': ['Pt', 'C', 'O', 'H'], 'num': [27, 18, 45, 121]}
      # print("\n")
      # for elem in self.elem['elem']:
      #    print(elem)


      df_vac = pd.DataFrame(temp_vac_coord)
      df_sol = pd.DataFrame(temp_sol_coord)
      # sort by group, then molecule, than element
      df_sol.sort_values(by=[4, 3, 5], inplace=True) #columns = ['x','y','z','mol','grp','type']
      self.coords = pd.concat([df_vac, df_sol]).reset_index(drop=True)
      self.coords.columns = ['x','y','z','mol','grp','type']    # molcule_tag, slab/solvent_group
      ######## add elem and name to self.coords according to the atom type
      for i in range(len(self.coords)): #['x','y','z','mol','grp','type','elem','name']
         self.coords.loc[i,'elem'] = self.atom.loc[self.coords.loc[i,'type']-1, 'elem']
         self.coords.loc[i,'name'] = self.atom.loc[self.coords.loc[i,'type']-1, 'name']
      # print(self.coords)

      mol = 1 # molecule tag start from 1
      for i in range(len(self.coords)):
         _type = self.coords.loc[i,'type']
         ### surface atom group
         if _type in self.surface_type:
            self.coords.loc[i,'mol'] = _type
         ### adsorbate group
         elif _type <= self.vac_type:
            self.coords.loc[i,'mol'] = len(self.surface_type)+1
         ### each solvent molecule group
         else:
            self.coords.loc[i,'mol'] += len(self.surface_type)+2









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
      m = self.multiplier

      ###must return array in order to use lambda in getDist()?
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

      #*********************************
      # future update: consider for cartesian POSCAR as input
      return min(np.sqrt(sum(map(lambda x:x*x, self.fracToCartesian(p0-p)))) for p in images)



   ### called inside getBondedList(self)
   def calcBonds(self, coords, s, all_bonds, bond_type):
      """calculate bonds within a molecule """
      bonds = []
      for pair in itertools.combinations(coords.index, 2):  ### 【140，141】
         e0 = self.coords.loc[pair[0],'elem'].split('_')[0]
         e1 = self.coords.loc[pair[1],'elem'].split('_')[0]
         bond = [e0, e1]

         # print(bond)
         if set(bond) in all_bonds:   ### all_bonds is  a list of bonds within one mol
            bidx = all_bonds.index(set(bond))
            p0 = np.array(self.coords.iloc[pair[0]][0:3])  ## iloc can slice integer row/col
            p1 = np.array(self.coords.iloc[pair[1]][0:3]) ## type of pandas series       
            dist = self.getDist(p0, p1)
            if dist < s.bond.iloc[bidx]['r0']*1.2:  ## cannot do s.bond.loc[bidx,'r0'] . OK
               #LAMMPS atom index starts from 1
               bond_name = bond[0] + '-' + bond[1] + '_' + s.name
               bonds.append([pair[0]+1, pair[1]+1, bond_name, bond_type+1+bidx]) 

               ###### if H is bonded to O, then update new index
               if set(bond) == {'O', 'H'}:
                  self.coords.loc[pair[1],'type'] += 1 # new index for H_O   
                  self.coords.loc[pair[1],'elem'] = self.atom.loc[self.coords.loc[pair[1],'type']-1, 'elem'] # new index for H_O
                  self.coords.loc[pair[1],'name'] = self.atom.loc[self.coords.loc[pair[1],'type']-1, 'name'] # new index for H_O
      # print(self.coords.iloc[:][112:172])
      return bonds


   ### called inside getBondedList(self)
   def calAngles(self, bonds, s, all_angles, angle_type):
      """calculate angles within a molecule """
      angles = []
      for pair in itertools.combinations(bonds, 2):
         k = list(set(pair[0][0:2]) & set(pair[1][0:2]))
         if len(k) > 0:
            b0 = [l for l in pair[0][0:2] if l not in k][0]
            b1 = k[0]   # common bond atom
            b2 = [l for l in pair[1][0:2] if l not in k][0]
            e0 = self.coords.loc[b0-1,'elem'].split('_')[0]
            e1 = self.coords.loc[b1-1,'elem'].split('_')[0]
            e2 = self.coords.loc[b2-1,'elem'].split('_')[0]           
            angle = [e0, e1, e2]
            
            if angle in all_angles:
               aidx = all_angles.index(angle)
               angle_name = e0 + '-' + e1 + '-' + e2 + '_' + s.name
               angles.append([b0, b1, b2, angle_name, angle_type+1+aidx])
            elif list(reversed(angle)) in all_angles:
               aidx = all_angles.index(list(reversed(angle)))
               angle_name = e2 + '-' + e1 + '-' + e0 + '_' + s.name
               angles.append([b2, b1, b0, angle_name, angle_type+1+aidx])

      return angles
   

   ### called inside getBondedList(self)
   def calDihedrals(self, angles, s, all_dihedrals, dihedral_type):
      """calculate dihedrals within a molecule """
      dihedrals = []
      for pair in itertools.combinations(angles, 2):
         flag = False
         if pair[0][1:3] == pair[1][:2]:
            b0 = pair[0][0]
            b1 = pair[0][1]
            b2 = pair[0][2]
            b3 = pair[1][2]  
            flag = True   
         elif pair[0][:2] == pair[1][1:3]:
            b0 = pair[1][0]
            b1 = pair[1][1]
            b2 = pair[1][2]
            b3 = pair[0][2]    
            flag = True     
         elif pair[0][1:3] == list(reversed(pair[1][1:3])):
            b0 = pair[0][0]
            b1 = pair[0][1]
            b2 = pair[0][2]
            b3 = pair[1][0]
            flag = True
         elif pair[0][:2] == list(reversed(pair[1][:2])):
            b0 = pair[0][2]
            b1 = pair[0][1]
            b2 = pair[0][0]
            b3 = pair[1][2]
            flag = True

         if flag:
            e0 = self.coords.loc[b0-1,'elem'].split('_')[0]
            e1 = self.coords.loc[b1-1,'elem'].split('_')[0]
            e2 = self.coords.loc[b2-1,'elem'].split('_')[0]
            e3 = self.coords.loc[b3-1,'elem'].split('_')[0]
            dihedral = [e0, e1, e2, e3]
   
            if dihedral in all_dihedrals:
               didx = all_dihedrals.index(dihedral)
               dihedral_name = e0 + '-' + e1 + '-' + e2 + '-' + e3 + '_' + s.name
               dihedrals.append([b0, b1, b2, b3, dihedral_name, dihedral_type+1+didx])
            elif list(reversed(dihedral)) in all_dihedrals:
               didx = all_dihedrals.index(list(reversed(dihedral)))
               dihedral_name = e3 + '-' + e2 + '-' + e1 + '-' + e0 + '_' + s.name
               dihedrals.append([b3, b2, b1, b0, dihedral_name, dihedral_type+1+didx])   

      return dihedrals  


   ###called inside writeLmpsData()
   def getBondedList(self):
      self.reorderCoords() ## needed for self.calcBonds, calcAngle, calcDihedrals
      start = sum(list(self.atom['num'])[0:self.vac_type])
      bond_type = 0
      angle_type = 0
      dihedral_type = 0
      bond_list = []
      angle_list = []
      dihedral_list = []
      ## loop over solvent chunk, e.g., first is H2O, second is methanol
      for i in range(len(self.solvent)):
         s = self.solvent.loc[i,'obj']
         step = s.total_atom
         stop = int(start +  self.solvent.loc[i,'num'] * step)  ## OK

         if s.total_bond_type > 0:  ### all bonds within a molecule
            all_bonds = [set(s.bond.index[n].split('-')) for n in range(len(s.bond.index))]
         if s.total_angle_type > 0: ### all angles within a molecule
            all_angles = [s.angle.index[n].split('-') for n in range(len(s.angle.index))]
         if s.total_dihedral_type > 0: ### all dihedrals within a molecule
            all_dihedrals = [s.dihedral.index[n].split('-') for n in range(len(s.dihedral.index))]

         ## accumulate bond and angle types
         for j in range(start, stop, step):
            if self.solvent.loc[i,'name'].lower() == "water":
               #LAMMPS atom index starts from 1
               # print('water j: ', j)
               self.bonds.append([j+1, j+2, 'O-H_'+s.name, bond_type+1])
               self.bonds.append([j+1, j+3, 'O-H_'+s.name, bond_type+1])
               self.angles.append([j+2, j+1, j+3, 'H-O-H_'+s.name, angle_type+1])
            else: 
               #### call bond, angle function           
               bonds = self.calcBonds(self.coords[j:j+step], s, all_bonds, bond_type)
               angles = self.calAngles(bonds, s, all_angles, angle_type)
               dihedrals = self.calDihedrals(angles, s, all_dihedrals, dihedral_type)
               self.bonds.extend(bonds)
               self.angles.extend(angles)
               self.dihedrals.extend(dihedrals)
         ##loop over next solvent
         bond_type += s.total_bond_type
         angle_type += s.total_angle_type
         dihedral_type += s.total_dihedral_type
         start = stop








   def writeSubLmps(self, path, jobtype='prod'):
      """
      2020/2/26 jobtype choices: equil, prod, ti 
      write .pbs script to submit LAMMPS jobs on Palmetto
      Note: lammps version is 31Mar17
      """
      jobname = self.ads_name + '_' + jobtype
      if jobtype == 'equil':
         lmps_input = 'input.equil'
      elif jobtype == 'prod':
         lmps_input =  'input.prod'
      elif jobtype == 'ti':
         lmps_input = 'input.ti'

      with open(os.path.join(path, 'sublammps.sh'), 'w') as f:
         f.write('#!/bin/bash\n')
         f.write('#PBS -N {}\n'.format(jobname))         
         f.write('#PBS -l select=1:ncpus=20:mpiprocs=20:interconnect=fdr:mem=120gb,walltime=72:00:00\n')
         f.write('#PBS -q workq\n')
         f.write('#PBS -j oe\n')  
         f.write('\necho "START ---------------------"\n')
         f.write('qstat -xf $PBS_JOBID\n')
         f.write('\nmodule purge\n')
         f.write('module add gcc/7.1.0 openmpi/1.8.4 fftw/3.3.4-g481\n')   
         f.write('\ncd $PBS_O_WORKDIR\n')
         f.write('mpiexec -n 20 /home/xiaohoz/bin/lmp_mpi < {}\n'.format(lmps_input))
         f.write('\nrm -f core.*\n')
         f.write('echo "FINISH --------------------"\n')


   def writeLmpsData(self, path, jobtype='prod'):
      """ jobtype choices: equil, prod, ti """
      with open(os.path.join(path, 'data.'+self.ads_name), 'w') as f:
         f.write('Created by Multi-Scale-Sampling (MSS) {}\n'.format(datetime.now().strftime('%c')))
         f.write('{} atoms\n'.format(len(self.coords)))  #atom number
         f.write('{} bonds\n'.format(len(self.bonds)))
         f.write('{} angles\n'.format(len(self.angles)))
         f.write('{} dihedrals\n'.format(len(self.dihedrals)))
         f.write('{} impropers\n'.format(0))  ## didn't calc impropers
         f.write('{} atom types\n'.format(len(self.atom)))  #atom types
         f.write('{} bond types\n'.format(len(self.bcoeff)))  #bond types
         f.write('{} angle types\n'.format(len(self.acoeff)))  #angle types
         f.write('{} dihedral types\n'.format(len(self.dcoeff)))  #dihedral types
         f.write('{} improper types\n'.format(0))  #didn't calc improper types

         # box dimensions
         for (i, j) in enumerate(('x', 'y', 'z')):
             f.write('{0:<10.6f} {1:>10.6f} {2}lo {2}hi\n'.format(0, self.multiplier*self.cell_vec[i][i], j))
         # box tilt
         f.write('{:<10.6f} {:>10.6f} {:>10.6f} xy xz yz\n'.format(self.multiplier*self.cell_vec[1][0],
                                                            self.multiplier*self.cell_vec[2][0],
                                                            self.multiplier*self.cell_vec[2][1]))
         # 200227 update: if run ti, do not write pair coeffs in data file, b/c input file has hybrid pair coeffs
         if jobtype in ['equil', 'prod']:
            f.write('\nPair Coeffs\n\n')
            for i in range(len(self.atom)):   #'name','elem','num','mass','epsLJ','sigLJ','charge'
               f.write('{:<3} {:>8.4f} {:>8.4f}\t# {}\n'.format(i+1, self.atom.loc[i,'epsLJ'],
                                                                  self.atom.loc[i,'sigLJ'],
                                                               self.atom.loc[i,'name']))
         if (len(self.bcoeff) > 0):
            f.write('\nBond Coeffs\n\n')
            for i in range(len(self.bcoeff)):
               f.write('{:<3} {:>8.4f} {:>8.4f}\t# {}\n'.format(i+1, self.bcoeff.loc[i,'k'],
                                                               self.bcoeff.loc[i,'r0'],
                                 self.bcoeff.loc[i,'type']+'_'+self.bcoeff.loc[i,'name']))
         if (len(self.acoeff) > 0):     
            f.write('\nAngle Coeffs\n\n')
            for i in range(len(self.acoeff)):
               f.write('{:<3} {:>8.4f} {:>8.4f}\t# {}\n'.format(i+1, self.acoeff.loc[i,'k'],
                                                                  self.acoeff.loc[i,'theta'],
                                 self.acoeff.loc[i,'type']+'_'+self.acoeff.loc[i,'name']))
         if (len(self.dcoeff) > 0):      
            f.write('\nDihedral Coeffs\n\n')
            for i in range(len(self.dcoeff)):
               f.write('{:<3} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}\t# {}\n'.format(i+1, self.dcoeff.loc[i,'k'][0],
                                                                    self.dcoeff.loc[i,'k'][1],
                                                                    self.dcoeff.loc[i,'k'][2],
                                                                    self.dcoeff.loc[i,'k'][3],
                                 self.dcoeff.loc[i,'type']+'_'+self.dcoeff.loc[i,'name'])) 
         #['name','elem','num','mass','epsLJ','sigLJ','charge','type']
         f.write('\nMasses\n\n')
         for i in range(len(self.atom)):
            f.write('{:<3} {:>8.4f}\t# {}\n'.format(i+1, self.atom.loc[i,'mass'], 
                                                       self.atom.loc[i,'name']))
         f.write('\nAtoms\n\n')
         for i in range(len(self.coords)):
            r = self.coords.loc[i]  
            if i < len(self.ddec): #### replace vac with ddec charge
               charge = self.ddec[i]
            else:
               charge = self.atom.loc[r['type']-1,'charge']
            #### self.coords: 'x','y','z','mol','grp','type','elem','name']
            #### LAMMPS data file: idx, mld_grp, atom_type, charge, x, y, z, # atom        
            pos =  self.fracToCartesian([r['x'], r[['y']], r['z']])
            f.write('{:<6} {} {:>4} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}\t# {}\n'.format(i+1,  
                  int(r['mol']), int(r['type']), charge, pos[0][0], pos[1][0], pos[2][0], r['name']))    

         if (len(self.bonds) > 0):
            f.write('\nBonds\n\n')
            for i in range(len(self.bonds)):
               f.write('{} {} {} {}\t# {}\n'.format(i+1, self.bonds[i][-1], self.bonds[i][0], 
                                                   self.bonds[i][1], self.bonds[i][-2]))
         if (len(self.angles) > 0):
            f.write('\nAngles\n\n')      
            for i in range(len(self.angles)):
               f.write('{} {} {} {} {}\t# {}\n'.format(i+1, self.angles[i][-1], self.angles[i][0], 
                                       self.angles[i][1], self.angles[i][2], self.angles[i][-2]))
         if (len(self.dihedrals) > 0):
            f.write('\nDihedrals\n\n')              
            for i in range(len(self.dihedrals)):
               f.write('{} {} {} {} {} {}\t# {}\n'.format(i+1, self.dihedrals[i][-1], self.dihedrals[i][0], 
                        self.dihedrals[i][1], self.dihedrals[i][2], self.dihedrals[i][3], self.dihedrals[i][-2]))


   def writeLmpsInputEquil(self, path):
      with open(os.path.join(path, 'input.equil'), 'w') as f:
         f.write('#Created by Multi-Scale-Sampling (MSS) {}\n'.format(datetime.now().strftime('%c')))         
         f.write('{:<20} real\n'.format('units'))
         f.write('{:<20} p p p\n'.format('boundary'))
         f.write('\n#### change variable value as needed in your system ####\n')
         f.write('{:<20} name string {}\n'.format('variable', self.ads_name))
         f.write('{:<20} temp equal {}\n'.format('variable', self.temperature))
         f.write('{:<20} press equal {}\n'.format('variable', self.pressure))
         f.write('{:<20} tcoupl equal {}\n'.format('variable', 100))
         f.write('{:<20} pcoupl equal {}\n'.format('variable', 5000))
         f.write('{:<20} dumpFreq equal {}\n'.format('variable', 10000))
         f.write('{:<20} thermFreq equal {}\n'.format('variable', 1000))
         f.write('{:<20} runStep equal {}\n'.format('variable', 5000000))  

         f.write('\n{:<20} log.${{name}}\n'.format('log'))
         f.write('{:<20} full\n'.format('atom_style'))
         f.write('{:<20} harmonic\n'.format('bond_style'))
         f.write('{:<20} harmonic\n'.format('angle_style'))
         f.write('{:<20} opls\n'.format('dihedral_style'))
         f.write('{:<20} lj/coul 0.0 0.0 0.5\n'.format('special_bonds'))


         #******************************************************************#
         # 200412, need modify for tip4p
         f.write('\n{:<20} lj/cut/coul/long 7.0\n'.format('pair_style'))     
         f.write('{:<20} mix geometric\n'.format('pair_modify'))
         f.write('{:<20} pppm 1.0e-4\n'.format('kspace_style'))         
         #******************************************************************#



         f.write('\n{:<20} data.${{name}}\n'.format('read_data'))
         f.write('{:<20} 1000000 restart.${{name}}.*\n'.format('#restart'))
         f.write('{:<20} restart.${{name}}.*\n\n'.format('#read_restart')) 

         ### 200411 update, arithmetic sigma for TIP3PCHARMM
         for i in list(range(1, len(self.atom)+1)):
            for j in list(range(i+1, len(self.atom)+1)):

               ## 200410 update, CHARMM water model is arithmetic sigma
               if (self.atom.loc[i-1,'name'].split('_')[-1] == 'TIP3PCHARMM' and 
                   self.atom.loc[j-1,'name'].split('_')[-1] == 'TIP3PCHARMM'):
                  pair = self.atom.loc[i-1,'name'] + '--' + self.atom.loc[j-1,'name']
                  f.write('{:<20} {:<2} {:<2} {:<8.4f} {:<8.4f} # {}\n\n'.format(
                                       'pair_coeff', i, j, 0.0836, 1.7753, pair))


         ##### 190508 add all groups #######
         for i in range(len(self.group)):
            f.write('{:<20} {} type {}\n'.format('group', self.group.loc[i,'name'], self.group.loc[i,'type']))

         f.write('\n{:<20} 2.0 bin\n'.format('neighbor'))
         f.write('{:<20} delay 1 every 1 check yes\n'.format('neigh_modify'))
         f.write('{:<20} ${{thermFreq}}\n'.format('thermo'))
         f.write('{:<20} custom step temp fmax fnorm etotal lx ly lz\n'.format('thermo_style'))
         f.write('{:<20} fixSlab slab setforce 0.0 0.0 0.0\n'.format('fix'))
         f.write('{:<20} freezeSlab slab rigid single force * off off off torque * off off off\n'.format('fix'))
         f.write('\nprint "------------ beginning minimization ------------"\n')
         f.write('{:<20} 1.0e-8  1.0e-10 100000 100000\n'.format('minimize'))
         f.write('\nprint "------------ beginning equilibration (const Vol)------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 0.1\n'.format('timestep'))
         f.write('{:<20} solvent create ${{temp}} 343037 rot yes dist gaussian\n'.format('velocity'))
         f.write('{:<20} slab set 0.0 0.0 0.0\n'.format('velocity'))
         f.write('{:<20} 1 solvent nve\n'.format('fix'))
         f.write('{:<20} 2 solvent temp/csvr ${{temp}} ${{temp}} ${{tcoupl}} 52364\n'.format('fix'))
         f.write('{:<20} custom step temp pe etotal press vol lz\n'.format('thermo_style'))
         f.write('{:<20} 500000 # modify as needed\n'.format('run'))
         f.write('\nprint "------------ check energy conservation (only nve)------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 2\n'.format('unfix'))
         f.write('{:<20} 500000 # modify as needed\n'.format('run'))
         f.write('\nprint "------------ beginning npt (use drag) ------------"\n')       
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 0.1\n'.format('timestep'))
         f.write('{:<20} 1\n'.format('unfix'))
         f.write('{:<20} delay 1 every 1 check yes exclude group slab slab\n'.format('neigh_modify'))
         f.write('{:<20} 1 solvent npt temp ${{temp}} ${{temp}} ${{tcoupl}} z ${{press}} ${{press}} ${{pcoupl}} dilate solvent drag 2.0 fixedpoint 0.0 0.0 0.0\n'.format('fix'))
         f.write('{:<20} tempMobile solvent temp\n'.format('compute'))
         f.write('{:<20} pressMobile all pressure tempMobile\n'.format('compute'))
         f.write('{:<20} custom step etotal c_tempMobile c_pressMobile temp press vol lz\n'.format('thermo_style'))
         f.write('{:<20} 1 temp tempMobile\n'.format('fix_modify'))

         #******************************************************************#
         # add fix shake for tip3pprice and tip4p
         #******************************************************************#





         # f.write('\n{:<20} 1 all atom ${{dumpFreq}} dump.${{name}}_eq.lammpstrj\n'.format('dump'))
         f.write('\n{:<20} 1 all custom ${{dumpFreq}} dump.${{name}}_eq.lammpstrj id mol type xs ys zs\n'.format('dump'))
         f.write('{:<20} 1 sort id\n'.format('dump_modify'))
         f.write('{:<20} ${{runStep}}\n'.format('run'))
         f.write('{:<20} data.${{name}}_eq\n'.format('write_data'))
         f.write('{:<20} rst.${{name}}_eq\n'.format('write_restart'))
 

   def writeLmpsInputProd(self, path):
      with open(os.path.join(path, 'input.prod'), 'w') as f:
         f.write('#Created by Multi-Scale-Sampling (MSS) {}\n'.format(datetime.now().strftime('%c')))         
         f.write('{:<20} real\n'.format('units'))
         f.write('{:<20} p p p\n'.format('boundary'))
         f.write('\n#### change variable value as needed in your system ####\n')
         f.write('{:<20} name string {}\n'.format('variable', self.ads_name))
         f.write('{:<20} temp equal {}\n'.format('variable', self.temperature))
         f.write('{:<20} press equal {}\n'.format('variable', self.pressure))
         f.write('{:<20} tcoupl equal {}\n'.format('variable', 100))
         f.write('{:<20} pcoupl equal {}\n'.format('variable', 5000))
         f.write('{:<20} dumpFreq equal {}\n'.format('variable', 10000))
         f.write('{:<20} thermFreq equal {}\n'.format('variable', 1000))
         f.write('{:<20} runStep equal {}\n'.format('variable', 5000000))  
         f.write('\n{:<20} log.${{name}}\n'.format('log'))
         f.write('{:<20} full\n'.format('atom_style'))
         f.write('{:<20} harmonic\n'.format('bond_style'))
         f.write('{:<20} harmonic\n'.format('angle_style'))
         f.write('{:<20} opls\n'.format('dihedral_style'))
         f.write('{:<20} lj/coul 0.0 0.0 0.5\n'.format('special_bonds'))         

         #******************************************************************#
         # 200412, need modify for tip4p         
         f.write('\n{:<20} lj/cut/coul/long 7.0\n'.format('pair_style'))     
         f.write('{:<20} mix geometric\n'.format('pair_modify'))
         f.write('{:<20} pppm 1.0e-4\n'.format('kspace_style'))
         #******************************************************************#

         f.write('\n{:<20} data.${{name}}\n'.format('read_data'))
         f.write('{:<20} 1000000 restart.${{name}}.*\n'.format('#restart'))
         f.write('{:<20} restart.${{name}}.*\n\n'.format('#read_restart'))            


         ### 200411 update, arithmetic sigma for TIP3PCHARMM
         for i in list(range(1, len(self.atom)+1)):
            for j in list(range(i+1, len(self.atom)+1)):

               ## 200410 update, CHARMM water model is arithmetic sigma
               if (self.atom.loc[i-1,'name'].split('_')[-1] == 'TIP3PCHARMM' and 
                   self.atom.loc[j-1,'name'].split('_')[-1] == 'TIP3PCHARMM'):
                  pair = self.atom.loc[i-1,'name'] + '--' + self.atom.loc[j-1,'name']
                  f.write('{:<20} {:<2} {:<2} {:<8.4f} {:<8.4f} # {}\n\n'.format(
                                       'pair_coeff', i, j, 0.0836, 1.7753, pair))

         ##### 190508 add all groups #######
         for i in range(len(self.group)):
            f.write('{:<20} {} type {}\n'.format('group', self.group.loc[i,'name'], self.group.loc[i,'type']))

         f.write('\n{:<20} 2.0 bin\n'.format('neighbor'))
         f.write('{:<20} delay 1 every 1 check yes\n'.format('neigh_modify'))
         f.write('{:<20} ${{thermFreq}}\n'.format('thermo'))
         f.write('{:<20} fixSlab slab setforce 0.0 0.0 0.0\n'.format('fix'))
         f.write('{:<20} freezeSlab slab rigid single force * off off off torque * off off off\n'.format('fix'))

         f.write('\nprint "------------ beginning minimization ------------"\n')
         f.write('{:<20} 1.0e-8  1.0e-10 100000 100000\n'.format('minimize'))
         f.write('\nprint "------------ beginning equilibration (const Vol)------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 0.1\n'.format('timestep'))
         f.write('{:<20} solvent create ${{temp}} 343037 rot yes dist gaussian\n'.format('velocity'))
         f.write('{:<20} slab set 0.0 0.0 0.0\n'.format('velocity'))
         f.write('{:<20} 1 solvent nve\n'.format('fix'))
         f.write('{:<20} 2 solvent temp/csvr ${{temp}} ${{temp}} ${{tcoupl}} 52364\n'.format('fix'))
         f.write('{:<20} custom step temp pe etotal press vol lz\n'.format('thermo_style'))
         f.write('{:<20} 500000 # modify as needed\n'.format('run'))
         f.write('\nprint "------------ check energy conservation (only nve)------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 2\n'.format('unfix'))
         f.write('{:<20} 500000 # modify as needed\n'.format('run'))
         f.write('\nprint "------------ beginning nvt ------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 1000000 restart.${{name}}.*\n'.format('#restart'))
         f.write('{:<20} 1.0\n'.format('timestep'))
         f.write('{:<20} 1\n'.format('unfix'))      
         f.write('{:<20} 1 solvent nvt temp ${{temp}} ${{temp}} ${{tcoupl}}\n'.format('fix'))
         f.write('{:<20} custom step temp  press etotal\n'.format('thermo_style'))
         f.write('\n#### calculate group/group Eint btwn H2O and adsorbates ####\n')
         f.write('{:<20} eint adsorbate group/group solvent kspace yes\n'.format('compute'))
         f.write('{:<20} 6 all ave/time 1000 1 1000 c_eint file eint.${{name}}\n'.format('fix'))

         #******************************************************************#
         # add fix shake for tip3pprice and tip4p
         #******************************************************************#


         # f.write('\n{:<20} 1 all atom ${{dumpFreq}} dump.${{name}}_prod.lammpstrj\n'.format('dump'))
         f.write('\n{:<20} 1 all custom ${{dumpFreq}} dump.${{name}}_prod.lammpstrj id mol type xs ys zs\n'.format('dump'))
         f.write('{:<20} 1 sort id\n'.format('dump_modify'))
         f.write('{:<20} ${{runStep}}\n'.format('run'))
         f.write('{:<20} data.${{name}}_prod\n'.format('write_data'))
         f.write('{:<20} rst.${{name}}_prod\n'.format('write_restart'))
         f.write('{:<20} dump.${{name}}_prod.lammpstrj first 0 every 1000 last ${{runStep}} dump x y z box yes scaled yes\n'.format('#rerun'))


   def writeLmpsInputTI(self, path):
      with open(os.path.join(path, 'input.ti'), 'w') as f:
         f.write('#Created by Multi-Scale-Sampling (MSS) {}\n'.format(datetime.now().strftime('%c')))         
         f.write('{:<20} real\n'.format('units'))
         f.write('{:<20} p p p\n'.format('boundary'))
         f.write('\n#### change variable value as needed in your system ####\n')
         f.write('{:<20} name string {}\n'.format('variable', self.ads_name))
         f.write('{:<20} temp equal {}\n'.format('variable', self.temperature))
         f.write('{:<20} press equal {}\n'.format('variable', self.pressure))
         f.write('{:<20} tcoupl equal {}\n'.format('variable', 100))
         f.write('{:<20} pcoupl equal {}\n'.format('variable', 5000))
         f.write('{:<20} dumpFreq equal {}\n'.format('variable', 100000))
         f.write('{:<20} thermFreq equal {}\n'.format('variable', 10000))
         f.write('{:<20} runStep equal {} # ti01\n'.format('variable', 7350000)) 
         f.write('\n{:<20} log.${{name}}\n'.format('log'))
         f.write('{:<20} full\n'.format('atom_style'))
         f.write('{:<20} harmonic\n'.format('bond_style'))
         f.write('{:<20} harmonic\n'.format('angle_style'))
         f.write('{:<20} opls\n'.format('dihedral_style'))
         f.write('{:<20} lj/coul 0.0 0.0 0.5\n'.format('special_bonds'))
         f.write('\n{:<20} data.${{name}}\n'.format('read_data'))
         f.write('{:<20} 1000000 restart.${{name}}.*\n'.format('#restart'))
         f.write('{:<20} restart.${{name}}.*\n\n'.format('#read_restart')) 

         ##### 190508 add all groups #######
         for i in range(len(self.group)):
            f.write('{:<20} {} type {}\n'.format('group', self.group.loc[i,'name'], self.group.loc[i,'type']))
         sol_ = self.group[self.group['name']=='solvent']['type'].item() # 6 7 8 9 10 11
         solvent_type = [int(i) for i in sol_.split()] # [6, 7, 8, 9, 10, 11]
         slab_ = self.group[self.group['name']=='slab']['type'].item()
         slab_type = [int(i) for i in slab_.split()]
         ads_ = self.group[self.group['name']=='adsorbate']['type'].item()
         ads_type = [int(i) for i in ads_.split()]   

         f.write('\n{:<20} hybrid/overlay lj/cut/coul/long 7.0 &\n'.format('pair_style'))     
         f.write('{:<20} lj/cut/soft 1 0.5 7 &\n'.format(''))
         f.write('{:<20} coul/long/soft 1 0 7\n'.format(''))
         f.write('{:<20} pppm 1.0e-4\n\n'.format('kspace_style'))
      


         for i in list(range(1, len(self.atom)+1)):
            for j in list(range(i, len(self.atom)+1)):

               ## 200410 update, CHARMM water model is arithmetic sigma
               if (self.atom.loc[i-1,'name'].split('_')[-1] == 'TIP3PCHARMM' and 
                   self.atom.loc[j-1,'name'].split('_')[-1] == 'TIP3PCHARMM') and (i != j):   
                  sig = 1.7753
               else:
                  sig = np.round(np.sqrt(self.atom.loc[i-1,'sigLJ']*self.atom.loc[j-1,'sigLJ']), 4)
               eps = np.round(np.sqrt(self.atom.loc[i-1,'epsLJ']*self.atom.loc[j-1,'epsLJ']), 4)
               pair = self.atom.loc[i-1,'name'] + '--' + self.atom.loc[j-1,'name']
               if eps < 0.00001 and sig < 0.00001:
                  sig = 1.0               
               if (i in solvent_type and j in slab_type) or (j in solvent_type and i in slab_type):
                  lam = 1.0 if any(x in [i, j] for x in self.surface_type) else 0.0
                  f.write('{:<12} {:<2} {:<2} {:<18} {:<8.4f} {:<8.4f} {:<6.1f}# {}\n'.format(
                                       'pair_coeff', i, j, 'lj/cut/soft', eps, sig, lam, pair))
                  f.write('{:<12} {:<2} {:<2} {:<18} {:<23.1f} # {}\n'.format(
                                       'pair_coeff', i, j, 'coul/long/soft', 0.0, pair))
               else:
                  f.write('{:<12} {:<2} {:<2} {:<18} {:<8.4f} {:<14.4f} # {}\n'.format(
                                       'pair_coeff', i, j, 'lj/cut/coul/long', eps, sig, pair))
            f.write('\n')
         f.write('{:<20} 2.0 bin\n'.format('neighbor'))
         f.write('{:<20} delay 1 every 1 check yes\n'.format('neigh_modify'))
         f.write('{:<20} ${{thermFreq}}\n'.format('thermo'))
         f.write('{:<20} fixSlab slab setforce 0.0 0.0 0.0\n'.format('fix'))
         f.write('{:<20} freezeSlab slab rigid single force * off off off torque * off off off\n'.format('fix'))


         f.write('\nprint "------------ beginning minimization ------------"\n')
         f.write('{:<20} 1.0e-8  1.0e-10 100000 100000\n'.format('minimize'))
         f.write('\nprint "------------ beginning equilibration (const Vol)------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 0.1\n'.format('timestep'))
         f.write('{:<20} solvent create ${{temp}} 343037 rot yes dist gaussian\n'.format('velocity'))
         f.write('{:<20} slab set 0.0 0.0 0.0\n'.format('velocity'))
         f.write('{:<20} 1 solvent nve\n'.format('fix'))
         f.write('{:<20} 2 solvent temp/csvr ${{temp}} ${{temp}} ${{tcoupl}} 52364\n'.format('fix'))
         f.write('{:<20} custom step temp pe etotal press vol lz\n'.format('thermo_style'))
         f.write('{:<20} 1000000 # modify as needed\n'.format('run'))
         f.write('\nprint "------------ check energy conservation (only nve)------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 2\n'.format('unfix'))
         f.write('{:<20} 500000 # modify as needed\n'.format('run'))
         f.write('\nprint "------------ beginning nvt ------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 1000000 restart.${{name}}.*\n'.format('#restart'))
         f.write('{:<20} 1.0\n'.format('timestep'))
         f.write('{:<20} 1\n'.format('unfix'))      
         f.write('{:<20} 1 solvent nvt temp ${{temp}} ${{temp}} ${{tcoupl}}\n'.format('fix'))
         f.write('{:<20} custom step temp  press etotal\n'.format('thermo_style'))

         f.write('\n{:<20} fullLJ equal 1.0\n'.format('variable'))
         f.write('{:<20} lambda equal ramp(0.0,1.05)  # ti01\n'.format('variable'))
         f.write('{:<20} dlambda equal 0.0001         # ti01\n'.format('variable'))
         f.write('\n################### scale LJ ###################\n')
         f.write('{:<20} ADAPT all adapt/fep 350000 pair lj/cut/soft lambda {}*{} {}*{} v_lambda after yes\n'.
                  format('fix', ads_type[0], ads_type[-1], solvent_type[0], solvent_type[-1]))
         f.write('{:<20} PRINT all print 350000 "adapt lambda = ${{lambda}}"\n'.format('fix'))
         f.write('{:<20} FEP all fep ${{temp}} pair lj/cut/soft lambda {}*{} {}*{} v_dlambda\n'.
                  format('compute', ads_type[0],ads_type[-1], solvent_type[0], solvent_type[-1]))
         f.write('{:<20} FEP all ave/time 1000 250 350000 c_FEP[1] c_FEP[2] file ti011.lmp\n'.format('fix'))
         f.write('{:<20} ${{runStep}}\n'.format('run'))
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} ADAPT\n'.format('unfix'))
         f.write('{:<20} FEP\n'.format('uncompute'))
         f.write('{:<20} FEP\n'.format('unfix'))
         f.write('\n################### keep full LJ ###################\n')
         f.write('{:<20} ADAPT all adapt/fep 0 pair lj/cut/soft lambda {}*{} {}*{} v_fullLJ\n'.
                  format('fix', slab_type[0], slab_type[-1], solvent_type[0], solvent_type[-1]))
         f.write('\n################### scale C ###################\n')
         f.write('{:<20} ADAPT2 all adapt/fep 350000 pair coul/long/soft lambda {}*{} {}*{} v_lambda after yes\n'.
                  format('fix', slab_type[0], slab_type[-1], solvent_type[0], solvent_type[-1]))
         f.write('{:<20} FEP all fep ${{temp}} pair coul/long/soft lambda {}*{} {}*{} v_dlambda\n'.
                  format('compute', slab_type[0], slab_type[-1], solvent_type[0], solvent_type[-1]))
         f.write('{:<20} FEP all ave/time 1000 250 350000 c_FEP[1] c_FEP[2] file ti012.lmp\n'.format('fix'))
         # f.write('\n{:<20} 1 all atom ${{dumpFreq}} dump.${{name}}_prod.lammpstrj\n'.format('dump'))
         f.write('\n{:<20} 1 all custom ${{dumpFreq}} dump.${{name}}_prod.lammpstrj id mol type xs ys zs\n'.format('dump'))

         f.write('{:<20} 1 sort id\n'.format('dump_modify'))
         f.write('{:<20} ${{runStep}}\n'.format('run'))
         f.write('{:<20} data.${{name}}_prod\n'.format('write_data'))
         f.write('{:<20} rst.${{name}}_prod\n'.format('write_restart'))
         f.write('{:<20} dump.${{name}}_prod.lammpstrj first 0 every 1000 last ${{runStep}} dump x y z box yes scaled yes\n'.format('#rerun'))




   def writeLmpsInputSurfaceTI(self, path):
      #### need check scale pairs
      with open(os.path.join(path, 'input.surfaceTI'), 'w') as f:      
         slab_type = list(range(1, self.vac_type+1)) #[1 2 3 4 5]
         if len(slab_type) != len(self.surface_type):
            sys.exit('\nERROR: cannot generate surface TI input file\nPLEASE use clean surface for surface TI input file generation\n')
         solvent_type = list(range(self.vac_type+1, len(self.atom)+1))  #[6 7 8 9 10 11]        
         f.write('#Created by Multi-Scale-Sampling (MSS) {}\n'.format(datetime.now().strftime('%c')))         
         f.write('{:<20} real\n'.format('units'))
         f.write('{:<20} p p p\n'.format('boundary'))
         f.write('\n#### change variable value as needed in your system ####\n')
         f.write('{:<20} name string {}\n'.format('variable', self.ads_name))
         f.write('{:<20} temp equal {}\n'.format('variable', self.temperature))
         f.write('{:<20} press equal {}\n'.format('variable', self.pressure))
         f.write('{:<20} tcoupl equal {}\n'.format('variable', 100))
         f.write('{:<20} pcoupl equal {}\n'.format('variable', 5000))
         f.write('{:<20} dumpFreq equal {}\n'.format('variable', 100000))
         f.write('{:<20} thermFreq equal {}\n'.format('variable', 10000))
         f.write('{:<20} runStep equal {} # ti01\n'.format('variable', 7350000))         
         f.write('\n{:<20} log.${{name}}\n'.format('log'))
         f.write('{:<20} full\n'.format('atom_style'))
         f.write('{:<20} harmonic\n'.format('bond_style'))
         f.write('{:<20} harmonic\n'.format('angle_style'))
         f.write('{:<20} opls\n'.format('dihedral_style'))
         f.write('{:<20} lj/coul 0.0 0.0 0.5\n'.format('special_bonds'))
         f.write('\n{:<20} data.${{name}}\n'.format('read_data'))
         f.write('{:<20} 1000000 restart.${{name}}.*\n'.format('#restart'))
         f.write('{:<20} restart.${{name}}.*\n\n'.format('#read_restart'))             

         ##### 190508 add all groups MODIFY THIS FOR ONLY SURFACE#######
         for i in range(len(self.group)):
            if self.group.loc[i,'name'] == 'slab':
               slab_ = self.group.loc[i,'type']
               slab_type = [int(i) for i in slab_.split()]
               f.write('{:<20} slab type {}\n'.format('group', self.group.loc[i,'type']))
            elif self.group.loc[i,'name'] == 'solvent':
               sol_ = self.group.loc[i,'type']
               f.write('{:<20} solvent type {}\n'.format('group', sol_))
               solvent_type = [int(i) for i in sol_.split()] 


         ###################### need to modify data file pair coeffs #########################
         f.write('\n{:<20} hybrid/overlay lj/cut/coul/long 7.0 &\n'.format('pair_style'))     
         f.write('{:<20} lj/cut/soft 1 0.5 7 &\n'.format(''))
         f.write('{:<20} coul/long/soft 1 0 7\n'.format(''))
         f.write('{:<20} pppm 1.0e-4\n\n'.format('kspace_style'))
         for i in list(range(1, len(self.atom)+1)):
            for j in list(range(i, len(self.atom)+1)):
               eps = np.round(np.sqrt(self.atom.loc[i-1,'epsLJ']*self.atom.loc[j-1,'epsLJ']), 4)
               sig = np.round(np.sqrt(self.atom.loc[i-1,'sigLJ']*self.atom.loc[j-1,'sigLJ']), 4)
               pair = self.atom.loc[i-1,'name'] + '--' + self.atom.loc[j-1,'name']
               if eps < 0.000001 and sig < 0.000001:
                  sig = 1.0
               if (i in solvent_type and j in slab_type) or (j in solvent_type and i in slab_type):
                  lam = 1.0
                  f.write('{:<20} {:<6} {:<6} {:<24} {:<8.4f} {:<8.4f} {:<8.1f}# {}\n'.format(
                                       'pair_coeff', i, j, 'lj/cut/soft', eps, sig, lam, pair))
                  f.write('{:<20} {:<6} {:<6} {:<24} {:<25.4f} # {}\n'.format(
                                       'pair_coeff', i, j, 'coul/long/soft', 0.0, pair))
               else:
                  f.write('{:<20} {:<6} {:<6} {:<24} {:<8.4f} {:<16.4f} # {}\n'.format(
                                       'pair_coeff', i, j, 'lj/cut/coul/long', eps, sig, pair))
            f.write('\n')
         f.write('{:<20} 2.0 bin\n'.format('neighbor'))
         f.write('{:<20} delay 1 every 1 check yes\n'.format('neigh_modify'))
         f.write('{:<20} ${{thermFreq}}\n'.format('thermo'))
         f.write('{:<20} fixSlab slab setforce 0.0 0.0 0.0\n'.format('fix'))
         f.write('{:<20} freezeSlab slab rigid single force * off off off torque * off off off\n'.format('fix'))

         f.write('\nprint "------------ beginning minimization ------------"\n')
         f.write('{:<20} 1.0e-8  1.0e-10 100000 100000\n'.format('minimize'))
         f.write('\nprint "------------ beginning equilibration (const Vol)------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 0.1\n'.format('timestep'))
         f.write('{:<20} solvent create ${{temp}} 343037 rot yes dist gaussian\n'.format('velocity'))
         f.write('{:<20} slab set 0.0 0.0 0.0\n'.format('velocity'))
         f.write('{:<20} 1 solvent nve\n'.format('fix'))
         f.write('{:<20} 2 solvent temp/csvr ${{temp}} ${{temp}} ${{tcoupl}} 52364\n'.format('fix'))
         f.write('{:<20} custom step temp pe etotal press vol lz\n'.format('thermo_style'))
         f.write('{:<20} 500000 # modify as needed\n'.format('run'))
         f.write('\nprint "------------ check energy conservation (only nve)------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 2\n'.format('unfix'))
         f.write('{:<20} 500000 # modify as needed\n'.format('run'))
         f.write('\nprint "------------ beginning nvt ------------"\n')
         f.write('{:<20} 0\n'.format('reset_timestep'))
         f.write('{:<20} 1000000 restart.${{name}}.*\n'.format('#restart'))
         f.write('{:<20} 1.0\n'.format('timestep'))
         f.write('{:<20} 1\n'.format('unfix'))      
         f.write('{:<20} 1 solvent nvt temp ${{temp}} ${{temp}} ${{tcoupl}}\n'.format('fix'))
         f.write('{:<20} custom step temp  press etotal\n'.format('thermo_style'))

         f.write('\n{:<20} fullLJ equal 1.0\n'.format('variable'))
         f.write('{:<20} lambda equal ramp(0.0,1.05)  # ti01\n'.format('variable'))
         f.write('{:<20} dlambda equal 0.0001         # ti01\n'.format('variable'))
         f.write('\n################### scale C ###################\n')
         f.write('{:<20} ADAPT2 all adapt/fep 350000 pair coul/long/soft lambda {}*{} {}*{} v_lambda after yes\n'.
                  format('fix', slab_type[0], slab_type[-1], solvent_type[0], solvent_type[-1]))
         f.write('{:<20} FEP all fep ${{temp}} pair coul/long/soft lambda {}*{} {}*{} v_dlambda\n'.
                  format('compute', slab_type[0], slab_type[-1], solvent_type[0], solvent_type[-1]))
         f.write('{:<20} FEP all ave/time 1000 250 350000 c_FEP[1] c_FEP[2] file fep012.lmp\n'.format('fix'))
         # f.write('\n{:<20} 1 all atom ${{dumpFreq}} dump.${{name}}_prod.lammpstrj\n'.format('dump'))
         f.write('\n{:<20} 1 all custom ${{dumpFreq}} dump.${{name}}_prod.lammpstrj id mol type xs ys zs\n'.format('dump'))

         f.write('{:<20} 1 sort id\n'.format('dump_modify'))
         f.write('{:<20} ${{runStep}}\n'.format('run'))
         f.write('{:<20} data.${{name}}_prod\n'.format('write_data'))
         f.write('{:<20} rst.${{name}}_prod\n'.format('write_restart'))
         f.write('{:<20} dump.${{name}}_prod.lammpstrj first 0 every 1000 last ${{runStep}} dump x y z box yes scaled yes\n'.format('#rerun'))



   def genLAMMPS(self, jobtype='prod'):
      """
      jobtype choices: equil, prod, ti 
      create a folder to run lammps simulation, has file:
      data.ads, input.xx, sublammps.pbs
      """
      path = os.path.join(self.curr_dir, '2-lammps', jobtype)
      self.createFolder(path) # e.g., ./2-lammps/prod/

      if jobtype == 'equil':
         self.writeLmpsInputEquil(path)
      elif jobtype == 'prod':
         self.writeLmpsInputProd(path)
      elif jobtype == 'ti':
         self.writeLmpsInputTI(path)
      self.writeLmpsData(path, jobtype)
      self.writeSubLmps(path, jobtype)


def main():

   curr_dir = sys.argv[1]
   lmps = VaspToLmps(curr_dir)

   lmps.genLAMMPS(jobtype='equil')
   lmps.genLAMMPS(jobtype='prod')
   # ti can be opt out if you do not run free energy calculation
   lmps.genLAMMPS(jobtype='ti')  


if __name__ == '__main__':
   main()


















