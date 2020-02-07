from fenics import *
from mshr import *
import numpy as np
from numpy import linalg as LA
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import xml.etree.ElementTree as ET
import math
from copy import deepcopy
import pickle
from joblib import Parallel, delayed
import time

calculateOnServer = False
#make driver and classes files available in this file
try:
	exec(open('/media/pb/UBUNTU/Masterarbeit/driver_classes.py').read())
except IOError:
	try:
		exec(open("/home/brunner/Masters/driver_classes.py").read())
		calculateOnServer = True
	except IOError:
		exec(open("/home/lea/Schreibtisch/Patrick/driver_classes.py").read())
try:
	exec(open("/media/pb/UBUNTU/Masterarbeit/classes.py").read())
except IOError:
	try:
		exec(open("/home/brunner/Masters/classes.py").read())
	except IOError:
		exec(open("/home/lea/Schreibtisch/Patrick/classes.py").read())

parameters['allow_extrapolation'] = True
#dont show any log messages like PETSc Solver information etc. Errors will still be displayed. Turn off for debugging
set_log_level(50)

#Time scale
T = 800000.0
num_steps = 800000
dt = T / num_steps


#use rafts/active surface areas?
activeSurfaceSource = False



# Define expressions/parameters used in variational forms
k 	= Constant(dt)

Dm 	= Constant(0.00001)
# autocatalytic reaction
nu 	= Constant(0.1) #0.001
k0 	= Constant(0.067)
K 	= Constant(0.001)
Km 	= Constant(0.1)
# decay
eta = Constant(1)
#
Ka 	= Constant(0.01)
# Ds is the diffusion of the pheromones and kb is a kinetic rate constant of the protease (e.g. bar1). The sqrt(Ds/kb * [B]) is also called lambda with [B] being the bar1-concentration..
# All play a role in degradation of the pheromones by proteases.
# If we assume uniform protease concentration: In simple terms, degradation by the protease limits the size of the pheromone cloud
# around each a-cell to a radius lambda, the average distance that a pheromone molecule diffuses before being degraded. 
# It is best if lambda(->sqrt(Ds/kb) to be specific) is about the average distance between cells
Ds 	= Constant(1) #300
kb 	= Constant(1) #0.6

h1_real = 0.3
h1 	= Constant(h1_real)
h2_real = 0.3
h2 	= Constant(h2_real)
h3_real = 0.3
h3 	= Constant(h3_real)
h4_real = 0.3
h4 	= Constant(h4_real)
h5_real = 0.3
h5 	= Constant(h5_real)
h6_real = 0.3
h6 	= Constant(h6_real)
h7_real = 0.3
h7 	= Constant(h7_real)
h8_real = 0.3
h8 	= Constant(h8_real)
h9_real = 0.3
h9 = Constant(h9_real)



#Noise level of the nextSourceCell finding process
Noise = 0.035


# lengthFactor of the straight used for orientating the normals. Check cellOrientation() for details. Should be bigger than r*2
straightLengthFactor = 100

# mesh.hmax() * refinementThreshold. At what distance a cell gets marked for refinement
refinementThreshold = 1.05


#Initialize mesh
r = 1.0
center1 = 0.0
center2 = 4.3

#Startingvalues stimulus
source1 = np.array([0.0, 0.0, -3.30])


sphere1 = Ellipsoid(Point(source1[0], source1[1] , source1[2]), r, r, r+0.30, 40)

source1 = np.array([0.0, 3.0, -3.30])



#Initialize my meshes
Resolution = 30
#load a previous mesh at this specific step.
dataLoadingTime = 5001
loadingCustomEnding = "_3_spheres_pressurized_small_distance"

globalSavingString = "_1_sphere_refinement_no_refinement"
#sometimes needed on the server, can be safely removed most of the time. Probably a fenics bug:
tempMesh = Mesh()

print('Initializing classMsphere1')
classMsphere1 = FEMMesh(Mesh=Mesh('classMsphere1_boundaryMesh_res' + str(Resolution) + '_' + str(dataLoadingTime) + str(loadingCustomEnding) + '.xml'), Name='classMsphere1', Source=source1, h_real=h1_real, Gender='Female', u_max_start=300, SaveFile=globalSavingString[1:] + '/u_1.pvd', Dimension = 2)


#array with all the classes in it, to iterate over. Very slow for alot of objects, try weakrefs instead then
import gc
usedMeshesList = []
for obj in gc.get_objects():
    if isinstance(obj, FEMMesh):
        usedMeshesList.append(obj)
        


#load my data into the specific FEMMesh at the specified dataLoadingTime and Resolution
for i in range(len(usedMeshesList)):
	print(usedMeshesList[i].name)

	usedMeshesList[i].sphereMidpoint = usedMeshesList[i].source

	loadData(usedMeshesList[i], n=dataLoadingTime, res=Resolution, customEnding = loadingCustomEnding, exclude = {'source'})

#Is it necessary to calculate the gradient?
calculateGradientSphere = [True]*len(usedMeshesList)

#for testing, set a custom source
classMsphere1.source = np.array([0.0, 0.0, -5.30])
#classMsphere2.source = np.array([2.0, 0.0, 0.0])



#initialize the PDE for every used Mesh. Needed to check if only one is present. If so, an artificial pheromone source is chosen (namely its own).
for Meshes in usedMeshesList:
	if activeSurfaceSource == False:
		Meshes.initSources(usedMeshesList)
		Meshes.initPDE()
		Meshes.isThereRefinementNecessary = False
	Meshes.u_max_start = 100
	print(Meshes.gender)



########################################################################################################################################
# put my PDEs in a list
listOfPDE = [None]*len(usedMeshesList)
for jj in range(len(usedMeshesList)):
	listOfPDE[jj] = usedMeshesList[jj].PDE


# preprocessing:
for Meshes in usedMeshesList:
	Meshes.getGradient(usedMeshesList)
	Meshes.distance_list = [None]*num_steps
	Meshes.Refinement_has_happened = False

#several data interfaces needed in time stepping
force_on_vertex_array = [None]*num_steps
Refinement_has_happened = False


################  time-stepping: ######################################################################################################

t = 0

for n in range(num_steps):
	# # Update current time
	print ('time =', t)
	t += dt
	print ('n:', n)

	# iterate over all presents meshes
	listOfPDECounter = 0
	for myMesh in usedMeshesList:
		print( 'Calculating mesh', myMesh.name)
		print( '')
		print( '')
		#solve the PDE. RuntimeError catch needed since it sometimes does not converge. Then some small random numbers are added and solved again.
		#basically changing the initial value.
		u_n = myMesh.currentSolutionFunction  #Function(FunctionSpace(myMesh.boundaryMesh, 'CG', 1)) 
		try:
			solve(listOfPDE[listOfPDECounter] == 0, u_n)
		except RuntimeError:
			print('############### RuntimeError ################')
			u_n.vector().set_local(np.random.rand(u_n.vector().size())+1)
			u_n.vector().apply("")
			solve(listOfPDE[listOfPDECounter] == 0, u_n)
		#helper counter, to go through all PDEs during one n 
		listOfPDECounter += 1
		# assign the new u_n to the old u_n
		myMesh.currentSolutionFunction = u_n
		myMesh.trialFunction.assign(myMesh.currentSolutionFunction)
		#save to File
		if n < 100:
			try:
				myMesh.saveFile << myMesh.currentSolutionFunction
			except AttributeError:
				myMesh.saveFile.write(u_n, t)
		else:
			if n % 100 == 0:
				try:
					myMesh.saveFile << myMesh.currentSolutionFunction
				except AttributeError:
					myMesh.saveFile.write(u_n, t) 

		if activeSurfaceSource == True:
			#get the new SourceCell and set it
			myMesh.stimulusCell = nextSourceCell(myMesh.boundaryMesh, myMesh.stimulusCell, myMesh.gradient, noiseFactor=Noise, hmin = myMesh.hmin, lengthFactor=1)
			myMesh.stimulusCellIndex = myMesh.stimulusCell.index()
			myMesh.source = myMesh.stimulusCell.midpoint()
			myMesh.stimulus.source0 = myMesh.source[0]
			myMesh.stimulus.source1 = myMesh.source[1]
			myMesh.stimulus.source2 = myMesh.source[2]


		# sum of all Cdc conenctrations
		u_sum = np.sum(myMesh.currentSolutionFunction.vector().get_local())

		#eval u_max depending on actual vs initial volume
		u_max = myMesh.u_max_start * myMesh.getVolume()/myMesh.startVolume
		#vary h depending on u_sum/u_max
		if u_sum <= u_max:
			print( u_sum/u_max)
			h = Constant(u_sum/u_max)
			myMesh.stimulus.h = h
		else:
			myMesh.stimulus.h = Constant(1.0)
			print( 'new h:', 1.0)
		#calculate the gradient only the beginning since its very expensive and I assume that after 70 steps the DPS has found its partner
		#just in case, every 100 steps a gradient calc is put in
		if n < 70:
			myMesh.getGradient(usedMeshesList)
		elif n % 100 == 0:
			myMesh.getGradient(usedMeshesList)




		if n > 10:
			#if n % (15000) == 0:
				# compare the current growth slope/gradient against the biggest one, if the growth slows down to x percent, refine
			if n == 20:
				# find extremes in Cdc42 concentration and write it to myMesh
				fillConcentrationExtremes(myMesh)

				#if AS == False, define the cell with the greatest Cdc42 concentraton
				if activeSurfaceSource == False:
					myMesh.stimulusCell = myMesh.maxValue_corresponding_cell


			# 	custom marking loop, for testing
				for cells222 in cells(myMesh.boundaryMesh):
					#BASE
					if myMesh.currentSolutionFunction(cells222.midpoint()) >= 0.92 * myMesh.maximumCurrentSolutionFunctionValue:
			# 		#if cells222.volume() > myMesh.cell_volumes_history[-1][cells222.index()] * refinementThreshold and myMesh.cell_markers_boundary[cells222] == True:
						#myMesh.isThereRefinementNecessary = True
			# 		#else:
						myMesh.cell_markers_boundary[cells222] = True
						myMesh.Estar[cells222.index()] = 0.8/(1.0-poisson_ratio1**2)
					# TIP
					if myMesh.currentSolutionFunction(cells222.midpoint()) >= 0.97 * myMesh.maximumCurrentSolutionFunctionValue:
						myMesh.cell_markers_boundary[cells222] = True
						myMesh.Estar[cells222.index()] = 2.5/(1.0-poisson_ratio1**2)
			# refine at this point, for testing
			#if n == 1000:
			#	myMesh.isThereRefinementNecessary = True



################  TRBS calculations: ######################################################################################################

			if n == 11:
				#start the cell volumes history, needed for processing TRBS
				myMesh.cell_volumes_history = [deepcopy(myMesh.cell_volumes)]
				poisson_ratio1 = 0.5


			# doing multiple growth steps here is faster than the n-loop. But it might cause the solving of the PDE to fail/not converge since we change the mesh
			# without taking the concentration changes into account.
			if n > 16 and myMesh.isThereRefinementNecessary == False:
				#update the cell edge map needed for force computation
				myMesh.cell_edges_and_opposite_angle_map = cell_edges_and_opposite_angle_map(myMesh)
				#quick and dirty parallel computation of my TRBS force on vertices.
				#joblib pickles saved data und reloads it to merge every processes results. Since dolphin objects cant be pickled all this mess is necessary.
				if __name__ == '__main__':
					# start = time.time()
					force_on_mesh = np.zeros((myMesh.boundaryMesh.num_vertices(), 3), dtype=np.float)
					sigma_VM_on_mesh = np.zeros((myMesh.boundaryMesh.num_cells(), 1), dtype=np.float)
					number_of_cells_local = myMesh.boundaryMesh.num_cells()
					#print(number_of_cells_local)
					number_of_vertices_local = myMesh.boundaryMesh.num_vertices()
					vertex_to_edges_map_local = myMesh.vertex_to_edges_map
					cell_to_vertices_map_local = myMesh.cell_to_vertices_map
					cell_to_edges_map_local = myMesh.cell_to_edges_map
					#previous_initial_cell_edges_and_opposite_angle_map_local = myMesh.previous_initial_cell_edges_and_opposite_angle_map
					initial_cell_edges_and_opposite_angle_map_local = myMesh.initial_cell_edges_and_opposite_angle_map
					#revious_cell_edges_and_opposite_angle_map_local = myMesh.previous_cell_edges_and_opposite_angle_map
					cell_edges_and_opposite_angle_map_local = myMesh.cell_edges_and_opposite_angle_map
					#print(cell_edges_and_opposite_angle_map_local)
					coordinates_local = myMesh.coordinates
					Estar_local = myMesh.Estar
					cell_volumes_local = myMesh.cell_volumes
					initial_cell_volumes_local = myMesh.initial_cell_volumes
					if calculateOnServer == True:
						number_of_jobs = 4
					else:
						number_of_jobs = 2

					#local_parent_cells = myMesh.parent_cells
					relax_triangles = False
					Refinement_has_happened = myMesh.Refinement_has_happened

						
					force_on_vertex_parallel = Parallel(n_jobs=number_of_jobs)(delayed(TRBS_force_on_mesh_parallel)(number_of_jobs, current_job_number, relax_triangles, Refinement_has_happened) for current_job_number in range(number_of_jobs))
					#properly merge all the results

					for ii in range(len(force_on_vertex_parallel)):
						force_on_mesh += force_on_vertex_parallel[ii][:-1][0]
						sigma_VM_on_mesh += force_on_vertex_parallel[ii][1:2][0]

					myMesh.Refinement_has_happened = False
					#clean up after all this
					del number_of_cells_local, number_of_vertices_local, vertex_to_edges_map_local, cell_to_vertices_map_local, cell_to_edges_map_local, initial_cell_edges_and_opposite_angle_map_local, cell_edges_and_opposite_angle_map_local, coordinates_local, Estar_local, cell_volumes_local, initial_cell_volumes_local


				#calculate the turgor pressures resulting force on every vertex
				myMesh.turgor_on_mesh = turgor_pressure_on_mesh(myMesh, pressure = 0.21)
				myMesh.force_on_mesh = force_on_mesh
			

				#calculate my overall force on every vertex. Since we are discreet the velocity has been omitted
				for vertexIndex2 in range(len(myMesh.coordinates)):
					tempAreaOfAdjacentCells = 0.0
					for cellIndices11 in myMesh.vertex_to_cells_map[vertexIndex2]:
						tempAreaOfAdjacentCells += myMesh.cell_volumes[cellIndices11]

					#since my arbitrary unit here is 1 = 1 um, the conversion of 1e6 is necessary.
					vertex_acceleration = 1/(1/3 * 1/2 * tempAreaOfAdjacentCells * 0.115 * 1e6) * (myMesh.turgor_on_mesh[vertexIndex2] + myMesh.force_on_mesh[vertexIndex2])

					distance_to_grow = vertex_acceleration 
					#do the growing
					myMesh.coordinates[vertexIndex2] += distance_to_grow  * 50#*2.5#* 1/4

				#update my datasets:
				myMesh.normalVectors = cellNormals(myMesh.boundaryMesh)
				myMesh.vertex_normal_map = vertex_normal_map(myMesh, myMesh.vertex_to_cells_map, myMesh.normalVectors)
				#myMesh.cell_volumes = cell_volumes(myMesh)
				myMesh.cell_volumes_history.append(deepcopy(myMesh.cell_volumes))

				#calculate alpha on every cell on this myMesh
				# alpha_on_mesh = TRBS_alpha_on_mesh(myMesh, sigma_VM_on_mesh, sigma_Y = 40)
				
				# for edgeIndex in range(len(myMesh.edges_to_cells_map)):
				# 	cell_T0 = myMesh.edges_to_cells_map[edgeIndex][0]
				# 	cell_T1 = myMesh.edges_to_cells_map[edgeIndex][1]
				# 	alpha_T0 = alpha_on_mesh[cell_T0]
				# 	alpha_T1 = alpha_on_mesh[cell_T1]
				# 	#get the initial edge length:
				# 	for cells_edges in range(len(myMesh.initial_cell_edges_and_opposite_angle_map[cell_T0])):
				# 		#find the correct edge:
				# 		if myMesh.initial_cell_edges_and_opposite_angle_map[cell_T0][cells_edges][0]== edgeIndex:
				# 			#take the initial edge_length
				# 			initial_edge_length = myMesh.initial_cell_edges_and_opposite_angle_map[cell_T0][cells_edges][1]
				# 			#print(1/2*(alpha_T0 + alpha_T1))
				# 			#print(initial_edge_length)
				# 			#change it accordingly
				# 			myMesh.initial_cell_edges_and_opposite_angle_map[cell_T0][cells_edges][1] += 1/2*(alpha_T0 + alpha_T1) * initial_edge_length
				# 			#print(myMesh.initial_cell_edges_and_opposite_angle_map[cell_T0][cells_edges][1] )
				# 			break



################  Refinement: ######################################################################################################



	#Refinement. In every n cycle there can only be one refinement. Why thats so is not known.
		#if isThereRefinementNecessary == True:
	for myMesh in usedMeshesList:
		#myMesh.Refinement_has_happened = False
		if n % len(usedMeshesList) == i:
			if myMesh.isThereRefinementNecessary == True:
				numberOfMarkedCells = 0
				for cells123 in cells(myMesh.boundaryMesh):
					if myMesh.cell_markers_boundary[cells123] == True:
						numberOfMarkedCells += 1
				print(numberOfMarkedCells)



				myMesh.previous_cell_edges_and_opposite_angle_map = myMesh.cell_edges_and_opposite_angle_map

				TEST_energy_on_cells_before_refinement = energy_TRBS(myMesh)

				# all new created cells get the E_value given here. All previously given Estar values remain the same.
				# standard is Estar = 2.5/(1.0-poisson_ratio1**2)
				listOfPDE = TRBSmyRefinement(myMesh, listOfPDE, straightLengthFactor, E_value = 0.8)

				myMesh.new_cells = new_cells_after_refinement(myMesh)



				myMesh.Refinement_has_happened = True
				myMesh.number_of_refinements += 1

				# during TRBSmyRefinement, the initial cell_volumes are written into meshClass.cell_volumes so this can then be added to the _history.
				# For proper growth the current cell_volume is needed, so i have to recompute it. Since refinement does not take place very often, this should not affect comp time
				myMesh.cell_volumes_history.append(deepcopy(myMesh.cell_volumes))
				myMesh.initial_cell_volumes = cell_volumes(myMesh)
				myMesh.cell_volumes = cell_volumes(myMesh)
				myMesh.isThereRefinementNecessary = False



				myMesh.cell_edges_and_opposite_angle_map = cell_edges_and_opposite_angle_map(myMesh)


				(myMesh.k_on_cells, myMesh.c_on_cells_angles_opposing_edges, myMesh.cells_edge_vectors) = k_and_c_on_cells(myMesh)
				
				TRBS_adjust_initial_edges_with_elastic_energy(myMesh, TEST_energy_on_cells_before_refinement)





# safe sim data every 2000 steps
	for i in range(len(usedMeshesList)):
		if n % 2000 == 0:
			saveData(usedMeshesList[i], num_steps+1, res=Resolution, customEnding = globalSavingString + "")
