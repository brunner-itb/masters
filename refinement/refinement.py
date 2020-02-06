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
#growth factors for every cell

#if meshslicing is used, this is the amout of slices each mesh is split into
amountOfSlices = 80



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
source2 = np.array([0.0, 0.0, 0.0])
source3 = np.array([3.8, 0.0, 0.0])

source4 = np.array([4.0, 0.0, 4.5])
source5 = np.array([4.0, 0.0, 4.0])
source6 = np.array([9.0, 0.0, 4.0])
source7 = np.array([0.0, 0.0, 9.0])
source8 = np.array([4.0, 0.0, 9.0])
source9 = np.array([9.0, 0.0, 8.0])


sphere1 = Ellipsoid(Point(source1[0], source1[1] , source1[2]), r, r, r+0.30, 40)
sphere2 = Ellipsoid(Point(source2[0], source2[1] , source2[2]), r+0.30, r, r, 40)
sphere3 = Ellipsoid(Point(source3[0], source3[1] , source3[2]), r+0.30, r, r, 40)

sphere4 = Ellipsoid(Point(source4[0], source4[1] , source4[2]), r, r+0.15, r, 40)
sphere5 = Ellipsoid(Point(source5[0], source5[1] , source5[2]), r, r+0.15, r, 40)
sphere6 = Ellipsoid(Point(source6[0], source6[1] , source6[2]), r, r+0.15, r, 40)
sphere7 = Ellipsoid(Point(source7[0], source7[1] , source7[2]), r, r+0.15, r, 40)
sphere8 = Ellipsoid(Point(source8[0], source8[1] , source8[2]), r, r+0.15, r, 40)
sphere9 = Ellipsoid(Point(source9[0], source9[1] , source9[2]), r, r+0.15, r, 40)

source1 = np.array([0.0, 3.0, -3.30])

#Initialize my meshes
Resolution = 30
#load a previous mesh at this specific step.
dataLoadingTime = 5001
loadingCustomEnding = "_3_spheres_pressurized_small_distance"

globalSavingString = "_1_sphere_refinement"
#sometimes needed on the server, can be safely removed most of the time. Probably a fenics bug:
tempMesh = Mesh()

print('Initializing classMsphere1')
classMsphere1 = FEMMesh(Mesh=Mesh('classMsphere1_boundaryMesh_res' + str(Resolution) + '_' + str(dataLoadingTime) + str(loadingCustomEnding) + '.xml'), Name='classMsphere1', Source=source1, h_real=h1_real, Gender='Female', u_max_start=300, SaveFile='new_refinement_test_bigger/u_1.pvd', Dimension = 2)
#print('Initializing classMsphere2')
#classMsphere2 = FEMMesh(Mesh=Mesh('classMsphere2_boundaryMesh_res' + str(Resolution) + '_' + str(dataLoadingTime) + str(loadingCustomEnding) + '.xml'), Name='classMsphere2', Source=source2, h_real=h2_real, Gender='Male', u_max_start=300, SaveFile='2nd_run_3_ellip_AS/u_2.pvd', Dimension = 2)
# print('Initializing classMsphere3')
# classMsphere3 = FEMMesh(Mesh=Mesh('classMsphere3_boundaryMesh_res' + str(Resolution) + '_' + str(dataLoadingTime) + str(loadingCustomEnding) + '.xml'), Name='classMsphere3', Source=source3, h_real=h3_real, Gender='Female', u_max_start=300, SaveFile='2nd_run_3_ellip_AS/u_3.pvd', Dimension = 2)
# print('Initializing classMsphere4')
# classMsphere4 = FEMMesh(Mesh=Mesh('classMsphere4_boundaryMesh_res' + str(Resolution) + '_' + str(dataLoadingTime) + str(loadingCustomEnding) + '.xml'), Name='classMsphere4', Source=source4, h_real=h4_real, Gender='Male', u_max_start=300, SaveFile='4_spheres/u_4.pvd', Dimension = 2)
# print('Initializing classMsphere5')
# classMsphere5 = FEMMesh(Mesh=Mesh('classMsphere5_boundaryMesh_res' + str(Resolution) + '_' + str(dataLoadingTime) + str(loadingCustomEnding) + '.xml'), Name='classMsphere5', Source=source5, h_real=h5_real, Gender='Female', u_max_start=300, SaveFile='9_spheres/u_5_cont.pvd', Dimension = 2)
# print('Initializing classMsphere6')
# classMsphere6 = FEMMesh(Mesh=Mesh('classMsphere6_boundaryMesh_res' + str(Resolution) + '_' + str(dataLoadingTime) + str(loadingCustomEnding) + '.xml'), Name='classMsphere6', Source=source6, h_real=h6_real, Gender='Male', u_max_start=300, SaveFile='9_spheres/u_6_cont.pvd', Dimension = 2)
# print('Initializing classMsphere7')
# classMsphere7 = FEMMesh(Mesh=Mesh('classMsphere7_boundaryMesh_res' + str(Resolution) + '_' + str(dataLoadingTime) + str(loadingCustomEnding) + '.xml'), Name='classMsphere7', Source=source7, h_real=h7_real, Gender='Female', u_max_start=300, SaveFile='9_spheres/u_7_cont.pvd', Dimension = 2)
# print('Initializing classMsphere8')
# classMsphere8 = FEMMesh(Mesh=Mesh('classMsphere8_boundaryMesh_res' + str(Resolution) + '_' + str(dataLoadingTime) + str(loadingCustomEnding) + '.xml'), Name='classMsphere8', Source=source8, h_real=h8_real, Gender='Male', u_max_start=300, SaveFile='9_spheres/u_8_cont.pvd', Dimension = 2)
# print('Initializing classMsphere9')
# classMsphere9 = FEMMesh(Mesh=Mesh('classMsphere9_boundaryMesh_res' + str(Resolution) + '_' + str(dataLoadingTime) + str(loadingCustomEnding) + '.xml'), Name='classMsphere9', Source=source9, h_real=h9_real, Gender='Female', u_max_start=300, SaveFile='9_spheres/u_9_cont.pvd', Dimension = 2)



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

Dm 	= Constant(0.000001)
#nu 	= Constant(0.0001) #0.001
#k0 	= Constant(0.067)
#K 	= Constant(0.0001)

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

# Time-stepping
t = 0


########################################################################################################################################
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
for n in range(num_steps):
	# # Update current time
	print ('time =', t)
	t += dt
	print ('n:', n)


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
		#save to File, first is pvd, second XDMF
		if n < 100:
			try:
				myMesh.saveFile << myMesh.currentSolutionFunction
			except AttributeError:
				myMesh.saveFile.write(u_n, t)
		else:
			if n % 1 == 0:
				try:
					myMesh.saveFile << myMesh.currentSolutionFunction
				except AttributeError:
					myMesh.saveFile.write(u_n, t) 

		if activeSurfaceSource == True:
			#get the new SourceCell
			myMesh.stimulusCell = nextSourceCell(myMesh.boundaryMesh, myMesh.stimulusCell, myMesh.gradient, noiseFactor=Noise, hmin = myMesh.hmin, lengthFactor=1)
			myMesh.stimulusCellIndex = myMesh.stimulusCell.index()
			myMesh.source = myMesh.stimulusCell.midpoint()
			myMesh.stimulus.source0 = myMesh.source[0]
			myMesh.stimulus.source1 = myMesh.source[1]
			myMesh.stimulus.source2 = myMesh.source[2]
		# if myMesh == usedMeshesList[-1]:
		# 	savingString = 'comparisonEddaFalse7/stimulusprojection%s.pvd' % n
		# 	if activeSurfaceSource == True:
		# 		print("saving pheromone distribution...")
		# 		File(savingString).write(savePheromone([classMsphere1, classMsphere2], 1.2, 15, 15, 15))
		# 	else:
		# 		if n == 1:
		# 			print("saving pheromone distribution...")
					#File(savingString).write(savePheromone([classMsphere2], 7.5, 45, 45, 45, twoDStimulus = True))
					#print("i put an exit here!")
					#exit()


		if activeSurfaceSource == True:
			myMesh.distance_list[n] = myMesh.stimulusCell.distance(Point(myMesh.sphereMidpoint[0], myMesh.sphereMidpoint[1] , myMesh.sphereMidpoint[2]))
			print('distance: ', myMesh.distance_list[n])
		#else:
	#		myMesh.distance_list[n] = myMesh.maxValue_corresponding_cell.distance(Point(myMesh.source[0], myMesh.source[1] , myMesh.source[2]))
		

		# sum of all Cdc conenctrations
		u_sum = np.sum(myMesh.currentSolutionFunction.vector().get_local())

		if n == 10:

			myMesh.cell_volumes_history = [deepcopy(myMesh.cell_volumes)]

			poisson_ratio1 = 0.5
			# myMesh.Estar = [2.5/(1.0-poisson_ratio1**2)]*myMesh.boundaryMesh.num_cells()
			# at refinement, all newly created cells get a certain Estar value, so no need to manually do it here
			#myMesh.initialRange = 0.15





		if n > 10:
			#setting Estar

			poisson_ratio1 = 0.5



			#if n % (15000) == 0:
			if n > 10:
				# compare the current growth slope/gradient against the biggest one, if the growth slows down to x percent, refine
				if n == 20:
					maximumCurrentSolutionFunctionValue = 0
					try:
						minimumCurrentSolutionFunctionValue = myMesh.currentSolutionFunction(Cell(myMesh.boundaryMesh, 0).midpoint())
					except RuntimeError:
						print("Building bounding_box_tree")
						meshClass.boundaryMesh.bounding_box_tree().build(myMesh.boundaryMesh)
						minimumCurrentSolutionFunctionValue = myMesh.currentSolutionFunction(Cell(myMesh.boundaryMesh, 0).midpoint())
					maxValue_corresponding_cell = None
					myMesh.minimumCellVolume = Cell(myMesh.boundaryMesh, 0).volume()
					for cellsObject in cells(myMesh.boundaryMesh):
						try:
							tempMidpoint = myMesh.currentSolutionFunction(cellsObject.midpoint())
						except RuntimeError:
							print("Building bounding_box_tree")
							myMesh.boundaryMesh.bounding_box_tree().build(myMesh.boundaryMesh)
							tempMidpoint = myMesh.currentSolutionFunction(cellsObject.midpoint())

						if tempMidpoint> maximumCurrentSolutionFunctionValue:
							maximumCurrentSolutionFunctionValue = tempMidpoint
							maxValue_corresponding_cell = cellsObject
						if tempMidpoint < minimumCurrentSolutionFunctionValue:
							minimumCurrentSolutionFunctionValue = tempMidpoint
						if cellsObject.volume() < myMesh.minimumCellVolume:
							myMesh.minimumCellVolume = cellsObject.volume()

					#if AS == False, define the cell with the greatest Cdc42 concentraton
					if activeSurfaceSource == False:
						myMesh.stimulusCell = maxValue_corresponding_cell





















				# 	#refinementThreshold = 1 # 2.1 + 0.12*myMesh.number_of_refinements
					for cells222 in cells(myMesh.boundaryMesh):
						if myMesh.currentSolutionFunction(cells222.midpoint()) >= 0.92 * maximumCurrentSolutionFunctionValue:
				# 		#if cells222.volume() > myMesh.cell_volumes_history[-1][cells222.index()] * refinementThreshold and myMesh.cell_markers_boundary[cells222] == True:
							#myMesh.isThereRefinementNecessary = True
				# 		#else:
							myMesh.cell_markers_boundary[cells222] = True
							myMesh.Estar[cells222.index()] = 0.8/(1.0-poisson_ratio1**2)
		
			if n == 10000:
				myMesh.isThereRefinementNecessary = True









































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
					listOfCellsAnglesToCheck = []





					sigma_Y = 40.0


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

				#old serialized force function
				#force_on_mesh = TRBS_force_on_mesh(myMesh)
				#calculate the turgor pressures resulting force on every vertex
				turgor_on_mesh = turgor_pressure_on_mesh(myMesh, pressure = 0.21)

				myMesh.force_on_mesh = force_on_mesh
				myMesh.turgor_on_mesh = turgor_on_mesh				



				#calculate my overall force on every vertex. Since we are discreet the velocity has been omitted
				for vertexIndex2 in range(len(myMesh.coordinates)):
					tempAreaOfAdjacentCells = 0.0
					for cellIndices11 in myMesh.vertex_to_cells_map[vertexIndex2]:
						tempAreaOfAdjacentCells += myMesh.cell_volumes[cellIndices11]
					#	print(tempAreaOfAdjacentCells)
					#since my arbitrary unit here is 1 = 1 um, the conversion of 1e6 is necessary.
					#to speed the whole growing process up, 1e6 has been adjusted. CAREFUL THO, has to be adjusted according to the Resolution.
					vertex_acceleration = 1/(1/3 * 1/2 * tempAreaOfAdjacentCells * 0.115 * 1e6) * (turgor_on_mesh[vertexIndex2] + force_on_mesh[vertexIndex2])
					#the velocity is the change of s over time. Time is in 1s increments, so it is just the previously grown distance
					#myMesh.vertex_velocity[vertexIndex2] = myMesh.distance_to_grow_history[n-1][vertexIndex2]

					distance_to_grow = vertex_acceleration #* 1/(3+7*number_of_refinements) #+ myMesh.vertex_velocity[vertexIndex2]

					#do the growing
					myMesh.coordinates[vertexIndex2] += distance_to_grow  * 50#*2.5#* 1/4

				#myMesh.distance_to_grow_history[n] = myMesh.distance_to_grow
				#force_on_vertex_array[n] = force_on_mesh
				#print("turgor pressure list", myMesh.turgor_pressure_on_vertex_list)
				#print(myMesh.force_on_vertex_list[50])
				#update my datasets:
				myMesh.normalVectors = cellNormals(myMesh.boundaryMesh)
				myMesh.vertex_normal_map = vertex_normal_map(myMesh, myMesh.vertex_to_cells_map, myMesh.normalVectors)
				#myMesh.cell_volumes = cell_volumes(myMesh)
				myMesh.cell_volumes_history.append(deepcopy(myMesh.cell_volumes))





				#calculate alpha on every cell on this myMesh
				# alpha_on_mesh = TRBS_alpha_on_mesh(myMesh, sigma_VM_on_mesh, sigma_Y)
				
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
						

					#exit()



				#exit()





		
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
		if n < 70:
			myMesh.getGradient(usedMeshesList)
		elif n % 100 == 0:
			myMesh.getGradient(usedMeshesList)


		# if n > 13:
		# 	myMesh.distance_list[n] = myMesh.maxValue_corresponding_cell.distance(Point(myMesh.source[0], myMesh.source[1] , myMesh.source[2]))
		# 	#print('greatest width:', myMesh.distance_list[n])
		# 	if n == 14 or myMesh.Refinement_has_happened == True:
		# 		myMesh.greatest_growth_slope = 0
		# 	if n > 15:
		# 		if myMesh.distance_list[n] - myMesh.distance_list[n-1] > myMesh.greatest_growth_slope:
		# 			myMesh.greatest_growth_slope = myMesh.distance_list[n] - myMesh.distance_list[n-1]
		# 			print("greatest_growth_slope:", myMesh.greatest_growth_slope)
		# 		if n > 20:
		# 			if myMesh.greatest_growth_slope != 0:
		# 				print('slope in percent of max:', 100*(myMesh.distance_list[n] - myMesh.distance_list[n-1])/myMesh.greatest_growth_slope)
		# 			else:
		# 				print('slope is at its maximum of ', myMesh.greatest_growth_slope)



	#Refinement. In every n cycle there can only be one refinement. Why that is is not known.
		#if isThereRefinementNecessary == True:
	for i in range(len(usedMeshesList)):
		#usedMeshesList[i].Refinement_has_happened = False
		if n % len(usedMeshesList) == i:
			if usedMeshesList[i].isThereRefinementNecessary == True:
				numberOfMarkedCells = 0
				for cells123 in cells(usedMeshesList[i].boundaryMesh):
					if usedMeshesList[i].cell_markers_boundary[cells123] == True:
						numberOfMarkedCells += 1
				print(numberOfMarkedCells)



				usedMeshesList[i].previous_cell_edges_and_opposite_angle_map = usedMeshesList[i].cell_edges_and_opposite_angle_map

				TEST_energy_on_cells_before_refinement = energy_TRBS(usedMeshesList[i])

				# all new created cells get the E_value given here. All previously given Estar values remain the same.
				# standard is Estar = 2.5/(1.0-poisson_ratio1**2)
				listOfPDE = TRBSmyRefinement(usedMeshesList[i], listOfPDE, straightLengthFactor, E_value = 0.8)

				usedMeshesList[i].new_cells = new_cells_after_refinement(usedMeshesList[i])



				usedMeshesList[i].Refinement_has_happened = True
				usedMeshesList[i].number_of_refinements += 1

				# during TRBSmyRefinement, the initial cell_volumes are written into meshClass.cell_volumes so this can then be added to the _history.
				# For proper growth the current cell_volume is needed, so i have to recompute it. Since refinement does not take place very often, this should not affect comp time
				usedMeshesList[i].cell_volumes_history.append(deepcopy(usedMeshesList[i].cell_volumes))
				usedMeshesList[i].initial_cell_volumes = cell_volumes(usedMeshesList[i])
				usedMeshesList[i].cell_volumes = cell_volumes(usedMeshesList[i])
				usedMeshesList[i].isThereRefinementNecessary = False



				usedMeshesList[i].cell_edges_and_opposite_angle_map = cell_edges_and_opposite_angle_map(usedMeshesList[i])














				# def TRBS_turgor_on_edge(meshClass):

				# 	turgor_on_edges = np.zeros((meshClass.boundaryMesh.num_edges(), 1), dtype=np.float)
				# 	for edgeIndex in range(meshClass.boundaryMesh.num_edges()):

				# 		v1 = Edge(meshClass.boundaryMesh, edgeIndex).entities(0)[0]
				# 		v2 = Edge(meshClass.boundaryMesh, edgeIndex).entities(0)[1]

				# 		# get the component of both vertices turgor force in the direction of the edge vector v1 to v0
				# 		temp_turgor = (np.dot(meshClass.turgor_on_mesh[v1] + meshClass.turgor_on_mesh[v2], meshClass.coordinates[v1] - meshClass.coordinates[v2]))/LA.norm(meshClass.coordinates[v1] - meshClass.coordinates[v2])# * meshClass.coordinates[v1] - meshClass.coordinates[v2]
				# 		turgor_on_edges[edgeIndex] = np.abs(temp_turgor)# LA.norm(temp_turgor)
				# 	print(turgor_on_edges)
				# 	return(turgor_on_edges)


				# usedMeshesList[i].turgor_on_mesh = turgor_pressure_on_mesh(usedMeshesList[i], pressure = 0.21)
	
				# usedMeshesList[i].turgor_on_edges = TRBS_turgor_on_edge(usedMeshesList[i])

				# def TRBS_adjust_initial_edges_after_refinement(meshClass):
				# 	#meshClass.initial_cell_edges_and_opposite_angle_map = cell_edges_and_opposite_angle_map(meshClass)
				# 	meshClass.edges_to_cells_map = edges_to_cells_map(meshClass)

				# 	for edgeIndex in range(meshClass.boundaryMesh.num_edges()):
				# 		# define the vertices according to Delingette and my own TRBS_force_on_mesh_parallel function
				# 		v2 = Edge(meshClass.boundaryMesh, edgeIndex).entities(0)[0]
				# 		v3 = Edge(meshClass.boundaryMesh, edgeIndex).entities(0)[1]

				# 		for cellIndex in meshClass.edges_to_cells_map[edgeIndex]:
				# 			# change it only for the newly created cells
				# 			if cellIndex in meshClass.new_cells:
				# 				# get third vertex:
				# 				for j in range(3):
				# 					if meshClass.cell_to_vertices_map[cellIndex][j] != v2 and meshClass.cell_to_vertices_map[cellIndex][j] != v3:
				# 						v1 = meshClass.cell_to_vertices_map[cellIndex][j]
				# 				# ALTERNATIV: in der k_and_c_on_cells function returnen lassen, dadurch einheitlicher und übersichtlicher!		
				# 				l1_vector = meshClass.cells_edge_vectors[cellIndex][0]
				# 				l2_vector = meshClass.cells_edge_vectors[cellIndex][1]
				# 				l3_vector = meshClass.cells_edge_vectors[cellIndex][2]


				# 				c2_projected = np.abs(meshClass.c_on_cells_angles_opposing_edges[cellIndex][edgeIndex] * (np.dot(l3_vector, l1_vector))/LA.norm(l1_vector))
				# 				c3_projected = np.abs(meshClass.c_on_cells_angles_opposing_edges[cellIndex][edgeIndex] * (np.dot(l2_vector, l1_vector))/LA.norm(l1_vector))

				# 				# zusammenfassen der unterschiedlichen k-werte und l-werte für die edges. Eine Edge hat eigentlich nur einen k-wert, nämlich die summe der k-werte aus
				# 				# dazugehörigen zellen. Stimmt das wirklich?

				# 				delta_l1 = meshClass.turgor_on_edges[edgeIndex]/(meshClass.k_on_cells[cellIndex][edgeIndex] + 2*c2_projected + 2*c3_projected)
				# 				#print(delta_l1)
				# 				#print( np.sqrt(delta_l1))
				# 				# are the edge.index() (0), edge.lengths() (1) and opposing angle (2).
				# 				for i in range(3):
				# 					if meshClass.initial_cell_edges_and_opposite_angle_map[cellIndex][i][0] == edgeIndex:
				# 						#L1 = l1 - delta_l1 = l1 - (l1 - L1)
				# 						#print("previous:", meshClass.initial_cell_edges_and_opposite_angle_map[cellIndex][i][1])
				# 						meshClass.initial_cell_edges_and_opposite_angle_map[cellIndex][i][1] = np.sqrt(Edge(meshClass.boundaryMesh, edgeIndex).length()**2 - np.sqrt(delta_l1))
				# 						#print("new:", meshClass.initial_cell_edges_and_opposite_angle_map[cellIndex][i][1])
				# 		#exit()
				(usedMeshesList[i].k_on_cells, usedMeshesList[i].c_on_cells_angles_opposing_edges, usedMeshesList[i].cells_edge_vectors) = k_and_c_on_cells(usedMeshesList[i])
				
				TRBS_adjust_initial_edges_with_elastic_energy(usedMeshesList[i], TEST_energy_on_cells_before_refinement)
				#TRBS_adjust_initial_edges_after_refinement(usedMeshesList[i])







				## initial cell edge dings überschrieben????