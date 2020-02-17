# only probably half of all this is used!
# many old deprecated function which have not been deleted.
# This file should be parsed at the beginning of each run.


#get Edge Lengths for scaling the movement vector:
def EdgeLengths(mesh):
	EdgeLengths = []
	meshcoord = mesh.coordinates()
	for hhh in range(len(meshcoord)):
		v = Vertex(mesh, hhh)
		EdgeLengths += [Edge(mesh, i).length() for i in v.entities(1)]
	return EdgeLengths

def vertexEdgeLengths(mesh, vertexID):
	vertexEdgeLengths = []
	meshcoord = mesh.coordinates()
	v = Vertex(mesh, vertexID)
	vertexEdgeLengths += [Edge(mesh, i).length() for i in v.entities(1)]
	return vertexEdgeLengths
#########  Calculate the vertex that is closest to my source ##########################################################################

def closestVertex(meshcoord, source):    # source as a point in space, returns vertex index corresponding to mesh.coordinates()
	tempDiff = np.zeros(len(meshcoord))
	for kk in range(len(meshcoord)):
		tempDiff[kk] = np.abs(meshcoord[kk][0] - source[0]) + np.abs(meshcoord[kk][1] - source[1]) + np.abs(meshcoord[kk][2] - source[2])
	closestVertexIndex = np.argmin(tempDiff)
	return closestVertexIndex


### Calculate cell which contains the given point, returns cell object
def correspondingCell(mesh, point):
	x_point = Point(*point) 
	cell_id = mesh.bounding_box_tree().compute_first_entity_collision(x_point)
	cell = Cell(mesh, cell_id)
	#print 'midpoint', cell.midpoint().str(True)
	#cell_id_1 = mesh.bounding_box_tree().compute_first_entity_collision(cell.midpoint())
	return cell


### projects the source onto the mesh, needed for the gradient
def sourceProjection(STIMULUS, FunctionSpace):
	source_projected = interpolate(STIMULUS, FunctionSpace)
	return source_projected

### calculates the gradient for a given function u on the mesh. u should have been projected onto the mesh already
### returns the gradient as an array of vectors with indexes corresponding to the given mesh/functionspace
def gradient(u):
		V = u.function_space()
		mesh = V.mesh()
		degree = V.ufl_element().degree()
		W = VectorFunctionSpace(mesh, 'CG', degree)
		grad_u = project(grad(u), W) #.vector().array().reshape(-1,3)
		return grad_u

### calculates the neighborhood of a given vertex Index on a given mesh, returns vertices
def neighborhood(mesh, Index):
	# Init vertex-edge connectivity
	mesh.init(0,1)
	v = Vertex(mesh, Index)
	idx = v.index()
	neighborhood = [Edge(mesh, i).entities(0) for i in v.entities(1)]
	neighborhood = np.array(neighborhood).flatten()
    # Remove own index from neighborhood
	neighborhood = neighborhood[np.where(neighborhood != idx)[0]]
	return neighborhood

### calculate the Travel "Probabilities", return is a OrderedDict() which remembers the order items were put into it
### Important for comparing it to neighbors, omits the use of maps between the two instances
def getNeighborProbabilities(meshcoord, StartIndex, neighborhood, gradient):   
	probabilities = collections.OrderedDict()
	#zeroCounter = 0
	for neighbor in neighborhood:
		neighborvector = meshcoord[neighbor] - meshcoord[StartIndex]
		probabilities[neighbor] = np.inner(gradient[StartIndex], neighborvector)/LA.norm(neighborvector)
		if probabilities[neighbor] < 0 :
			probabilities[neighbor] = 0
			#zeroCounter += 1
	return probabilities

### Calculate the Probability to move to the given cell based on the gradient length of its vertices. Returns a float
def getCellProbability(cell, gradient):
	#vertex_coordinates = cell.get_vertex_coordinates().reshape(-1,3)
	probability = LA.norm(gradient(cell.midpoint()))
	return probability

### Calculate the Probabilites for the cell itself and its neighbors. Probability value is based on getCellProbability()
def getCellNeighborProbabilities(mesh, cell, cell_neighbors, gradient):
	probabilities = collections.OrderedDict()
	probabilities[cell.index()] = getCellProbability(cell, gradient)
	for neighbor in cell_neighbors[cell.index()]:
		probabilities[neighbor] = getCellProbability(Cell(mesh, neighbor), gradient)

	#print 'probabilities not normed:', probabilities
	
	probabilitiesSum = np.sum(probabilities.values())
	for key in probabilities:
		probabilities[key] *= 1/probabilitiesSum
	return probabilities


# Calculate the gradient Vector and apply Noise in all 3 directions (x,y,z). Calculate which cell is closest to the Vectors tip,
# return that cell as the next source cell.
# Returns a cell object.
def nextSourceCell(mesh, cell, gradient2, noiseFactor, hmin, lengthFactor=1, usedMeshesList = None, meshClass = None):
	global calculateGradientSphere
	dim = 3 #len(gradient2(cell.midpoint()))
	#averageEdgeLengths = np.average(EdgeLengths(mesh))
	maxEdgeLengths = hmin*lengthFactor #np.max(EdgeLengths(mesh))
	#print 'maxEdgeLengths', maxEdgeLengths	#get the gradient2 Vector and scale it with the average edge lengths
	gradientVector = np.zeros(dim)
	try:
		gradientNorm = LA.norm(gradient2(cell.midpoint()))
	except RuntimeError:
		print("Building bounding_box_tree")
		mesh.bounding_box_tree().build(mesh)
		gradientNorm = LA.norm(gradient2(cell.midpoint()))


	for i in range(dim):
		gradientVector[i] = cell.midpoint()[i] + (1/gradientNorm) * gradient2(cell.midpoint())[i] * maxEdgeLengths
	#* (maxEdgeLengths/LA.norm(gradientVectorUnscaled))
	# sometimes the gradient fails. In this case. try to calculate it again
	if gradientNorm == 0:
		print( '################################')
		print( '################################')
		print( '      NaN, gradientNorm = 0     ')
		print( '################################')
		print( '################################')
		if usedMeshesList != None and meshClass != None:
			meshClass.getGradient(usedMeshesList)
			try:
				gradientNorm = LA.norm(meshClass.gradient(cell.midpoint()))
			except RuntimeError:
				print("Building bounding_box_tree")
				mesh.bounding_box_tree().build(mesh)
				gradientNorm = LA.norm(meshClass.gradient(cell.midpoint()))

			for i in range(dim):
				gradientVector[i] = cell.midpoint()[i] + (1/gradientNorm) * meshClass.gradient(cell.midpoint())[i] * maxEdgeLengths


	#get some random noise in there, using the normal distribution with sigma = noiseFactor
	gradientVectorNoise = np.zeros(dim)
	for ii in range(dim):
		gradientVectorNoise[ii] = gradientVector[ii] + noiseFactor*np.random.randn()
	#calculate which neighboring cell.midpoint() is closest to my gradientVectorNoise tip:
	tempDiff = {}
	for meshcells in cells(mesh):
		tempValue = 0
		for iiii in range(dim):
			tempValue += np.abs(gradientVectorNoise[iiii] - meshcells.midpoint()[iiii])
		tempDiff[meshcells] = tempValue
		#tempDiff[meshcells] = {np.sum(np.abs(gradientVectorTip[i] - meshcells.midpoint()[i]) for i in range(dim))}
	#get the closest cell by aquiring the minimum distance and asking for the corresponding cell
	closestCell = min(tempDiff, key=lambda k: tempDiff[k])# tempDiff.keys()[tempDiff.values().index(min(tempDiff.values()))]
	# if cell == closestCell:
	# 	if closestCell == cell1:
	# 		calculateGradientSphere[1] == False
	# 	else:
	# 		calculateGradientSphere[0] == False
	return closestCell

############ Calculate the next Source Vertex based on the neighborProbablilites. Going backwards translates into a probability = 0.
############ Probability to stay is calculated via number of zeroes in the probability divided by the number of neighbors
def nextSourceVertex(meshcoord, StartIndex, neighborhood, gradient):
	global calculateGradientSphere
	getNeighProb = getNeighborProbabilities(meshcoord, StartIndex, neighborhood, gradient)
	#print ''
	#print 'getNeighborProbabilities: ', getNeighProb
	#print ''
	zeroCounter = 0.0
	for ggg in getNeighProb.values():
		if ggg == 0:
			zeroCounter += 1
	#Calculate the probabilities based on this function
	getNeighProb[StartIndex] = np.sum(getNeighProb.values()) * ((zeroCounter*0.5)/len(getNeighProb.values()))
	#add self to neighborhood so the choice of not moving can be made
	neighborhood_plusSelf = np.append(neighborhood, StartIndex)
	#print 'neighborProbablilites with selfProb calculated: ', getNeighProb
	#print ''

	# Norm the probabilities:
	getNeighProbSum = sum(getNeighProb.values())
	if getNeighProbSum == 0: ### if all are Zero we have found the optimum for this time step and dont want to move
		if StartIndex == StartIndex1:
			calculateGradientSphere[1] = False
		else:
			calculateGradientSphere[0] = False
		return StartIndex
	for key in getNeighProb:
		getNeighProb[key] *= 1/getNeighProbSum
	# These can now be used as probabilities. Use numpy to make a weighted random pick from all neighbors:
	newStartIndex = np.random.choice(neighborhood_plusSelf, 1, p=getNeighProb.values())[0]
	print( 'oldStartIndex: ', StartIndex)
	print( 'newStartIndex: ', newStartIndex)
	print( '')
	if newStartIndex == StartIndex:
		if newStartIndex == StartIndex1:
			calculateGradientSphere[1] = False
		else:
			calculateGradientSphere[0] = False
		return newStartIndex
	else:
		return newStartIndex




################ growth, rudimentary implementation based on nodal contribution to the overall concentration #########################

def growth(u, meshcoord, center):
	FunctionSpace = u.function_space()
	u = u.vector().get_local()
	u_sum = np.sum(u)
	#v2d transforms nodal_values (here u.vector().array()) to be in the same order as mesh.coordinates()
	v2d = vertex_to_dof_map(FunctionSpace)
	
	center_n = np.array([center, 0, 0])
	for vertex in range(len(meshcoord)):
		if u[v2d[vertex]] >= 0.02*u_sum:
			movevector = np.zeros(3)
			#normierter Richtungsvektor fuer das Wachstum
			for ooo in range(3):
				movevector[ooo] = (1/LA.norm(meshcoord[vertex] - center_n)) * (meshcoord[vertex] - center_n)[ooo]
			#Laenge des neuen Vectors bestimmt durch Anteil der lokalen Konz. an der Gesamtkonzentration
			new_vector = movevector * u[v2d[vertex]]/u_sum * 0.2
			#Schrumpfen ist nicht moeglich, deshalb heaviside
			meshcoord[vertex] += np.heaviside(new_vector, 0) * new_vector

################ Calculate the neighbors of every cell, save it in a dictionary:
def cellNeighbors(meshClass):
	# Init facet-cell connectivity
	tdim = meshClass.boundaryMesh.topology().dim()
	meshClass.boundaryMesh.init(tdim - 1, tdim)
	# For every cell, build a list of cells that are connected to its facets
	# but are not the iterated cell
	return {cell.index(): sum((list(filter(lambda ci: ci != cell.index(),
                                            vertex.entities(tdim)))
                                    for vertex in vertices(cell)), [])
                for cell in cells(meshClass.boundaryMesh)}

################ Calculate normals on cells in the same order as the cells are ordered #######################
# cell_normals does not always point outwards
def cellNormals(mesh):
	n_1 = np.zeros(3*mesh.num_cells()).reshape(-1,3)
	# for cells1 in cells(mesh):
	# 	tempPointArray = [0]*3
	# 	tempVectorArray = [0]*2 #mesh.topology().dim()
	# 	tempPointCounter = 0
		#tempVectorCounter = 0
		#print tempArray
		# try:
		# 	for facet in facets(cells1):
		# 		tempArray[tempCounter] = facet.normal()[:]
		# 		tempCounter += 1
		# except RuntimeError:
		# for vertices1 in vertices(cells1):
		# 	tempPointArray[tempPointCounter] = vertices1.point()
		# 	#tempArray[tempCounter] = edge.normal()[:]
		# 	tempPointCounter += 1
		# for i in range(2):
		# 	tempVectorArray[i] = -tempPointArray[i] + tempPointArray[i+1]
		# n_1[cells1.index()] = np.cross(tempVectorArray[0], tempVectorArray[1])
		# n_1[cells1.index()] *= 1/LA.norm(n_1[cells1.index()])
		# if cells1.index() % 2 == 0:
		# 	n_1[cells1.index()] *= 1/LA.norm(n_1[cells1.index()])
		# else:
		# 	n_1[cells1.index()] *= -1/LA.norm(n_1[cells1.index()])
		#print n_1[cells1.index()]
	#return n_1
	for ii in range(mesh.num_cells()):
		n_1[ii][0] = Cell(mesh, ii).cell_normal()[0]
		n_1[ii][1] = Cell(mesh, ii).cell_normal()[1]
		n_1[ii][2] = Cell(mesh, ii).cell_normal()[2]
	return n_1


################# get the cell orientation. 1 is outwards, -1 is inwards. Requires a triangle tree, previously loaded from the xml file of the unordered Boundary mesh.
def cellOrientationTriangles(triangles, Cell):
	try:
		triangle = triangles[Cell.index()]
	except AttributeError:
		triangle = triangles[Cell]
	#based on index parity. cyclic shift of the vertex ordering implies correctly ordered crossproduct (counterclockwise)
	if triangle.get('v0') < triangle.get('v1') < triangle.get('v2') or triangle.get('v1') < triangle.get('v2') < triangle.get('v0') or triangle.get('v2') < triangle.get('v0') < triangle.get('v1'):
		return 1
	else:
		return -1

#deprecated function as everything that has to do with slicing!
def parallelMyCellOrientation(meshClass, amountOfSlices, straightLengthFactor, rangeStarter='Standard'):
	n = cellNormals(meshClass.boundaryMesh)
	cellOrientationArray = np.zeros(3*meshClass.boundaryMesh.num_cells()).reshape(-1,3)

	arrayHoldingSlices = meshClass.classArrayHoldingSlices

	xValueMin = np.min(meshClass.coordinates.T[0])
	xValueMax = np.max(meshClass.coordinates.T[0])

	yValueMin = np.min(meshClass.coordinates.T[1])
	yValueMax = np.max(meshClass.coordinates.T[1])

	zValueMin = np.min(meshClass.coordinates.T[2])
	zValueMax = np.max(meshClass.coordinates.T[2])

	sliceSize = (xValueMax - xValueMin)/amountOfSlices
	#radius should be bigger than max distance between two vertices
	radiusOfChecking = meshClass.boundaryMesh.hmax()*1.2
	stepSizeStraight = sliceSize/3

	#this is used so the initial vertex and its sourroundings are not considered a collision. But in the case I dont get any collisions I should recursivly call this function
	#again with a custom rangeStarter, probably 1 or 0.
	if rangeStarter == 'Standard':
		rangeStarter = math.ceil(radiusOfChecking/stepSizeStraight)

	cellOrientationArray = Parallel(n_jobs=2)(delayed(myCellOrientation(meshClass, amountOfSlices, straightLengthFactor, n, cellOrientationArray, arrayHoldingSlices, xValueMin, xValueMax, yValueMin, yValueMax, zValueMin, zValueMax, sliceSize, radiusOfChecking, stepSizeStraight, rangeStarter)))
	return cellOrientationArray

#deprecated function as everything that has to do with slicing!
# the size of the sphere which determines in which slice I am and how far i move with each iteration should be calculated for slice size
def myCellOrientation(meshClass, amountOfSlices, straightLengthFactor, rangeStarter='Standard' ):#, n, cellOrientationArray, arrayHoldingSlices, xValueMin, xValueMax, yValueMin, yValueMax, zValueMin, zValueMax, sliceSize, radiusOfChecking, stepSizeStraight, rangeStarter='Standard'):
	n = cellNormals(meshClass.boundaryMesh)
	cellOrientationArray = np.zeros(3*meshClass.boundaryMesh.num_cells()).reshape(-1,3)

	arrayHoldingSlices = meshClass.classArrayHoldingSlices

	xValueMin = np.min(meshClass.coordinates.T[0])
	xValueMax = np.max(meshClass.coordinates.T[0])

	yValueMin = np.min(meshClass.coordinates.T[1])
	yValueMax = np.max(meshClass.coordinates.T[1])

	zValueMin = np.min(meshClass.coordinates.T[2])
	zValueMax = np.max(meshClass.coordinates.T[2])

	sliceSize = (xValueMax - xValueMin)/amountOfSlices
	#radius should be bigger than max distance between two vertices
	radiusOfChecking = meshClass.boundaryMesh.hmax()*1.3
	stepSizeStraight = sliceSize/7

	#this is used so the initial vertex and its sourroundings are not considered a collision. But in the case I dont get any collisions I should recursivly call this function
	#again with a custom rangeStarter, probably 1 or 0.
	if rangeStarter == 'Standard':
		rangeStarter = math.ceil(radiusOfChecking/stepSizeStraight)

	for cell123 in cells(meshClass.boundaryMesh):
		cellNormal = n[cell123.index()]
		startingPoint = cell123.midpoint().array()
		print(cell123.index())

		FirstEncounterCounter = 0
		SecondEncounterCounter = 0

		for step in range(rangeStarter,int(straightLengthFactor/stepSizeStraight)):
			#move forward one step in the default cellNormal direction
			currentCheckingPoint = startingPoint + cellNormal*stepSizeStraight*step
			#print('startingPoint:', startingPoint, 'cellNormal:', cellNormal, 'stepSizeStraight:', stepSizeStraight)
			#print(currentCheckingPoint)
			xValue = currentCheckingPoint[0] - xValueMin
			#check if any extreme value is achieved. This way if we are definitely not inside we can break. Basically a cube around the mesh
			#is formed and if that cube is exceeded, we break. Should limit computational time.
			if currentCheckingPoint[0] > (xValueMax + radiusOfChecking) or currentCheckingPoint[1] > yValueMax or currentCheckingPoint[2] > zValueMax:
				break
			#the a//b only returns integers, so basically division without remainder.
			#now this division directly tells us in which slice to check, no need for iterations
			sliceToCheck = int(xValue//sliceSize)
			#if we are less than xValueMin, < 0 happens (since xValue is corrected by xValueMin).
			#Then check if we are in the radius of Checking, if so just use the first slice.
			if sliceToCheck < 0: 
				if xValue%sliceSize < radiusOfChecking:
					sliceToCheck = 0
				else:
					break
			#same for greater than xValueMax and inside the radiusOfChecking
			elif sliceToCheck >= amountOfSlices:
				if xValue%sliceSize < radiusOfChecking:
					sliceToCheck = amountOfSlices-1
				else:
					break
			try:
				#calculate distances, and if the distance is low enough, count up
				for vertices in arrayHoldingSlices[sliceToCheck]:
					#print('vertex coordinates:', meshClass.coordinates[vertices], 'currentCheckingPoint:', currentCheckingPoint)
					#print('subtraction:', LA.norm(meshClass.coordinates[vertices] - currentCheckingPoint))
					if LA.norm(meshClass.coordinates[vertices] - currentCheckingPoint) < radiusOfChecking:
						FirstEncounterCounter += 1
			except IndexError:
				print(sliceToCheck)
				print(int(xValue//sliceSize))
				exit()
		
		#if we havent found anything, it is very likely that this orientation is the right one
		#or the rangeStater skipped too much causing no collision to occur!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		if FirstEncounterCounter == 0:
			cellOrientationArray[cell123.index()] = 1
		#if not, invert the vector direction and do it again:
		else:
			cellNormal *= -1

			for step in range(rangeStarter,int(straightLengthFactor/stepSizeStraight)):
				#move forward one step in the default cellNormal direction
				currentCheckingPoint = startingPoint + cellNormal*stepSizeStraight*step
				#print('startingPoint:', startingPoint, 'cellNormal:', cellNormal, 'stepSizeStraight:', stepSizeStraight)
				#print(currentCheckingPoint)
				xvalue = currentCheckingPoint[0] - xValueMin
				#check if any extreme value is achieved. This way if we are definitely not inside we can break. Basically a cube around the mesh
				#is formed and if that cube is exceeded, we break. Should limit computational time.
				if currentCheckingPoint[0] > (xValueMax + radiusOfChecking) or currentCheckingPoint[1] > yValueMax or currentCheckingPoint[2] > zValueMax:
					break
				#the a//b only returns integers, so basically division without remainder.
				#now this division directly tells us in which slice to check, no need for iterations
				sliceToCheck = int(xvalue//sliceSize)
				#if we are less than xValueMin, < 0 happens (since xValue is corrected by xValueMin).
				#Then check if we are in the radius of Checking, if so just use the first slice.
				if sliceToCheck < 0: 
					if xValue%sliceSize < radiusOfChecking:
						sliceToCheck = 0
					else:
						break
				#same for greater than xValueMax and inside the radiusOfChecking
				elif sliceToCheck >= amountOfSlices:
					if xValue%sliceSize < radiusOfChecking:
						sliceToCheck = amountOfSlices-1
					else:
						break
				try:
					#calculate distances, and if the distance is low enough, count up
					for vertices in arrayHoldingSlices[sliceToCheck]:
						if LA.norm(meshClass.coordinates[vertices] - currentCheckingPoint) < radiusOfChecking:
							SecondEncounterCounter += 1
				except IndexError:
					print(sliceToCheck)
					print(int(xValue//sliceSize))
					exit()

			if FirstEncounterCounter != 0 and SecondEncounterCounter != 0:
				if FirstEncounterCounter > SecondEncounterCounter:
					cellOrientationArray[cell123.index()] = 1
				else:
					cellOrientationArray[cell123.index()] = -1
			elif SecondEncounterCounter == 0:
				cellOrientationArray[cell123.index()] = -1
	return cellOrientationArray








# better calculation of the cellOrientation. It basically takes the normal vector given by fenics(cross product of two edges)
# puts a straight through it starting at the cell.midpoint() and ending at the midpoint + cellNormal*straightLengthFactor. 
# Now collision between the given mesh and the straight is checked and saved. Same thing is done again, just with
# normalVector *-1, so the straight goes in the opposite direction. Collision is saved again and compared against
# the previous collision. If there is less collision it is very probable that this orientation points outwards and is saved.
# Its the inital cell orientation since it is very fast but does not work properly in special cases involving shmoos.
def initialCellOrientation(meshClass, straightLengthFactor):
	mesh = meshClass.boundaryMesh
	mesh.init()
	meshTree = BoundingBoxTree()
	meshTree.build(mesh)
	n = cellNormals(mesh)
	cellOrientationArray = np.zeros(3*mesh.num_cells()).reshape(-1,3)
	for cell123 in cells(mesh):
		cellNormal = n[cell123.index()]
		startingPoint = cell123.midpoint()
		#straightLengthFactor = 22.5
		endingPoint = startingPoint + Point(cellNormal[0] * straightLengthFactor, cellNormal[1] * straightLengthFactor, cellNormal[2] * straightLengthFactor)
		testStartingPoint1 = startingPoint + Point(cellNormal[0] * 0.001, cellNormal[1] * 0.001, cellNormal[2] * 0.001)

		tempMesh = Mesh()
		editor = MeshEditor()
		editor.open(tempMesh, 'triangle', 2, 3)
		editor.init_cells(1)
		editor.init_vertices(4)
		editor.add_vertex(1, [testStartingPoint1.x(), testStartingPoint1.y(), testStartingPoint1.z()])
		editor.add_vertex(2, [endingPoint.x(), endingPoint.y(), endingPoint.z()])
		editor.add_vertex(3, [endingPoint.x(), endingPoint.y(), endingPoint.z()])
		#for server version, no idea why
		testVerticesToAdd = np.array([1,2,3], dtype='uintp')
		editor.add_cell(0, testVerticesToAdd)
		editor.close()
	

		tempMeshTree = BoundingBoxTree()
		tempMeshTree.build(tempMesh)

		collisionsFirstTry, onlyZeros = meshTree.compute_collisions(tempMeshTree)

		#testPoint = Cell(mesh, 100).midpoint()
		#print(meshTree.compute_entity_collisions(testPoint))
		
		#print( 'cell:', cell123.index(),'collisions:', collisionsFirstTry)
		# if len(collisionsFirstTry) > 100:
		# 	cellToCheckCoordinates = [0,0,0]
		# 	cellToCheckCoordinates[0] = cell123.midpoint().array()[0]
		# 	cellToCheckCoordinates[1] = cell123.midpoint().array()[1]
		# 	cellToCheckCoordinates[2] = cell123.midpoint().array()[2]

		# 	plottingList_x = []
		# 	plottingList_y = []
		# 	plottingList_z = []
		# 	for cellIndices1234 in collisionsFirstTry:
		# 		plottingList_x.append(Cell(mesh, cellIndices1234).get_vertex_coordinates()[0])
		# 		plottingList_x.append(Cell(mesh, cellIndices1234).get_vertex_coordinates()[3])
		# 		plottingList_x.append(Cell(mesh, cellIndices1234).get_vertex_coordinates()[6])

		# 		plottingList_y.append(Cell(mesh, cellIndices1234).get_vertex_coordinates()[1])
		# 		plottingList_y.append(Cell(mesh, cellIndices1234).get_vertex_coordinates()[4])
		# 		plottingList_y.append(Cell(mesh, cellIndices1234).get_vertex_coordinates()[7])

		# 		plottingList_z.append(Cell(mesh, cellIndices1234).get_vertex_coordinates()[2])
		# 		plottingList_z.append(Cell(mesh, cellIndices1234).get_vertex_coordinates()[5])
		# 		plottingList_z.append(Cell(mesh, cellIndices1234).get_vertex_coordinates()[8])

			#from mpl_toolkits.mplot3d import Axes3D

			# plot(tempMesh)
			# plt.show()
			# plt.close()

			# fig = plt.figure()
			# ax = fig.add_subplot(111, projection='3d')
			# wholeCell_x = mesh.coordinates().T[0]
			# wholeCell_y = mesh.coordinates().T[1]
			# wholeCell_z = mesh.coordinates().T[2]
			# print('wholeCell_x:', wholeCell_x)
			# ax.scatter(wholeCell_x, wholeCell_y, wholeCell_z, zdir='z', s=5, c='Black', depthshade=True )
			# ax.scatter(plottingList_x, plottingList_y, plottingList_z, zdir='z', s=20, c='Green', depthshade=True)
			# ax.scatter(cellToCheckCoordinates[0], cellToCheckCoordinates[1], cellToCheckCoordinates[2], zdir='z', s=200, c='Red', depthshade=True)
			# ax.plot([startingPoint[0], endingPoint[0]], [startingPoint[1], endingPoint[1]], [startingPoint[2], endingPoint[2]], '-', zdir='z')
			# plt.show()

		if len(collisionsFirstTry) > 4:

			cellNormal = -1*n[cell123.index()]
			endingPoint = startingPoint + Point(cellNormal[0] * straightLengthFactor, cellNormal[1] * straightLengthFactor, cellNormal[2] * straightLengthFactor)
			tempMesh = Mesh()
			editor = MeshEditor()
			editor.open(tempMesh, 'triangle', 2, 3)
			editor.init_cells(1)
			editor.init_vertices(4)
			editor.add_vertex(1, [startingPoint.x(), startingPoint.y(), startingPoint.z()])
			editor.add_vertex(2, [endingPoint.x(), endingPoint.y(), endingPoint.z()])
			editor.add_vertex(3, [endingPoint.x(), endingPoint.y(), endingPoint.z()])
			#for server version, no idea why
			testVerticesToAdd = np.array([1,2,3], dtype='uintp')
			editor.add_cell(0, testVerticesToAdd)
			editor.close()

			tempMeshTree = BoundingBoxTree()
			tempMeshTree.build(tempMesh)

			collisionsSecondTry, onlyZeros = meshTree.compute_collisions(tempMeshTree)

			del(tempMesh, tempMeshTree)

			if len(collisionsFirstTry) > 60 and len(collisionsSecondTry) > 60:
				if len(collisionsFirstTry) > len(collisionsSecondTry):
					cellOrientationArray[cell123.index()] = 1
				else:
					cellOrientationArray[cell123.index()] = -1
			elif len(collisionsSecondTry) < len(collisionsFirstTry):
				cellOrientationArray[cell123.index()] = -1
			else:
				cellOrientationArray[cell123.index()] = 1
			
			#print('cell:', cell123.index(),'collisions:', collisionsSecondTry)

		else:
			cellOrientationArray[cell123.index()] = 1
	return cellOrientationArray

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# WARNING: SPLICING ONLY WORKS FOR x >= 0!
# slices the mesh.vertices in different layers based on their x-value.
# the direction of the slices follows the "vector" argument.
# returns a list of lists. list[:] are the slices, list[:][:] are the vertices in said slices
def meshSlicing(meshClass, vector, amountOfSlices):
	meshcoord_copy= deepcopy(meshClass.coordinates)
	# after the meshcoords are deepcopy'ed they are rotated so the normalised "vector" alligns with the x-axis.
	# this allows for quick slicing without extensive collision detection etc.
	a = 1/LA.norm(vector)*np.asarray(vector)
	b = [1.0,0.0,0.0]
	if (a != b).any():
		if (a == np.dot(-1,b)).all():
			R = np.array([[np.cos(pi), -np.sin(pi), 0], [np.sin(pi), np.cos(pi), 0], [0,0,1]])
			print(R)
			for i in range(len(meshcoord_copy)):
				meshcoord_copy[i] = np.dot(R , meshcoord_copy[i])
		else:
			v = np.cross(a,b)
			s = LA.norm(v)
			c = np.dot(a,b)
			I = np.eye(3)
			v_x = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
			R = I + v_x + v_x**2 *(1/(1+c))
			#print(R)
			for i in range(len(meshcoord_copy)):
				meshcoord_copy[i] = np.dot(R , meshcoord_copy[i])
	else:
		print("vector aligns with the x-axis, skipping rotation")
	meshcoordTransposed = meshcoord_copy.T
	xValueMin = np.min(meshcoordTransposed[0])
	xValueMax = np.max(meshcoordTransposed[0])
	sliceSize = (xValueMax - xValueMin)/amountOfSlices
	arrayHoldingSlices = [[] for i in range(amountOfSlices)]

	#print(xValueMin, xValueMax, sliceSize, arrayHoldingSlices)

	for index, vertices in enumerate(meshcoord_copy):
		#get the x-value and correct it with xValueMin. This way all vertices lie inbetween 0 and (xValueMax - xValueMin).
		xvalue = vertices[0] - xValueMin
		#print('xvalue:', xvalue, 'sliceSize:', sliceSize)
		#the a//b only returns integers, so basically division without remainder.
		#now this division directly tells us in which slice this vertex should go, no need for iterations
		sliceTheVertexGoesIn = int(xvalue//sliceSize)
		#print('spliceTheVertexGoesIn:', spliceTheVertexGoesIn)
		#print('index:', index)
		#try is used because it is faster than anything else
		try:
			arrayHoldingSlices[sliceTheVertexGoesIn].append(index)
		#handling of the case where the vertex is on the last splice outline
		except IndexError:
			if sliceTheVertexGoesIn == amountOfSlices:
				arrayHoldingSlices[sliceTheVertexGoesIn-1].append(index)
			else:
				raise IndexError('list index out of range')
	print('meshSlicing completed...')
	#flat_list = [item for sublist in arrayHoldingSlices for item in sublist]
	#print(len(meshClass.coordinates), len(flat_list))
	#meshslices were in the wrong order, so i reversed the order here. Quick and dirty
	return arrayHoldingSlices[::-1]




# not deprecated function
def meshClassOrientation(meshClass, straightLengthFactor):
	#triangles = meshClass.triangles
	tempCellOrientation = initialCellOrientation(meshClass, straightLengthFactor)
	meshOrientationArray = [0]*meshClass.boundaryMesh.num_cells()
	#compare orientation with neighbors. Basically calculate the average direction and
	tempNeighbors = cellNeighbors(meshClass)

	for i in range(meshClass.boundaryMesh.num_cells()):
		meshOrientationArray[i] = tempCellOrientation[i]
		# tempNeighborVectorSum = [0,0,0]
		# for neighbors in tempNeighbors[i]:
		# 	tempNeighborVectorSum += meshClass.normalVectors[neighbors]
		# if np.dot(meshOrientationArray[i], tempNeighborVectorSum) < 0:
		# 	meshOrientationArray[i] *= -1
	return meshOrientationArray


################# Grow the cell by growthFactor. Can be a Function. Shrinking is not yet possible.
def cellGrowth(meshcoord, Cell, Triangles, normalVectors, growthFactor):
	#get vertex coordinates fo the cell
	coordinate_dofs = Cell.get_vertex_coordinates().reshape(-1,3)
	#calculate the non-oriented normal of the cell
	#n = Cell.cell_normal()
	#orient the normal and extend along it by growthFactor
	if cellOrientation(Triangles, Cell) == 1:
		meshcoord[Cell.entities(0)[0]] = coordinate_dofs[0] + growthFactor*normalVectors[Cell.index()]
		meshcoord[Cell.entities(0)[1]] = coordinate_dofs[1] + growthFactor*normalVectors[Cell.index()]
		meshcoord[Cell.entities(0)[2]] = coordinate_dofs[2] + growthFactor*normalVectors[Cell.index()]
	else:	
		meshcoord[Cell.entities(0)[0]] = coordinate_dofs[0] - growthFactor*normalVectors[Cell.index()]
		meshcoord[Cell.entities(0)[1]] = coordinate_dofs[1] - growthFactor*normalVectors[Cell.index()]
		meshcoord[Cell.entities(0)[2]] = coordinate_dofs[2] - growthFactor*normalVectors[Cell.index()]

#def cellGrowthVertices(meshcoord, allKnowingArray, growthFactor):


def dist(x, y):
	D = [x[0] - y[0], x[1] - y[1], x[2] - y[2]]
	return np.sqrt(D[0]**2 + D[1]**2 + D[2]**2)

#gets the x,y,z diameters and averages over those. Only really useful if using a sphere. Returns a float
def getAverageSphereDiameter(meshClass, r_check = 2):
	xmax_comp = [10*r_check,0,0]
	xmin_comp = [-10*r_check,0,0]
	ymax_comp = [0,10*r_check,0]
	ymin_comp = [0,-10*r_check,0]
	zmax_comp = [0,0,10*r_check]
	zmin_comp = [0,0,-10*r_check]


	closestVertex_xmax = closestVertex(meshClass.coordinates, xmax_comp)
	closestVertex_ymax = closestVertex(meshClass.coordinates, ymax_comp)
	closestVertex_zmax = closestVertex(meshClass.coordinates, zmax_comp)

	closestVertex_xmin = closestVertex(meshClass.coordinates, xmin_comp)
	closestVertex_ymin = closestVertex(meshClass.coordinates, ymin_comp)
	closestVertex_zmin = closestVertex(meshClass.coordinates, zmin_comp)

	x_dist = dist(meshClass.coordinates[closestVertex_xmax], meshClass.coordinates[closestVertex_xmin])
	y_dist = dist(meshClass.coordinates[closestVertex_ymax], meshClass.coordinates[closestVertex_ymin])
	z_dist = dist(meshClass.coordinates[closestVertex_zmax], meshClass.coordinates[closestVertex_zmin])
	
	return (x_dist + y_dist + z_dist)/3.0

#create a vertex to cells map, returns an array. The first array index is the vertex index, the corresponding array are the cells.index(). 
#so vertex_to_cells_map[10] returns all cells.index() that contain vertex.index() 10
def vertex_to_cells_map(meshClass):
	mesh = meshClass.boundaryMesh
	vertex_to_cells_map_untrimmed = np.zeros([mesh.num_vertices(),300], dtype=int)
	#fill array with the cells
	for v in vertices(mesh):
		cellCounter = 0
		for c in cells(v):
			vertex_to_cells_map_untrimmed[v.index()][cellCounter] = c.index()
			cellCounter += 1
	vertex_to_cells_map = [0]*mesh.num_vertices()
	#trim all zeros from the back 'b', leave the front ones since there is a cell.index() = 0. Might lose some back zeros
	for trimming in range(len(vertex_to_cells_map_untrimmed)):
		vertex_to_cells_map[trimming] = np.trim_zeros(vertex_to_cells_map_untrimmed[trimming], 'b')
	vertex_to_cells_map = np.asarray(vertex_to_cells_map)
	return vertex_to_cells_map

#create a vertex normal map which takes the corresponding cells of a vertex, sums and norms over all the normal vectors of those cells and appoints that
#normal vector to this vertex. So it Averages over all surrounding normal vectors and takes that vector. This is used to "grow" a vertex.
#returns a map, which is an array where the first index is the vertex.index() and the entries are the normal vector components
def vertex_normal_map(meshClass, vertex_to_cells_map, normalVectors):
	mesh = meshClass.boundaryMesh
	vertex_normal_map = np.zeros([mesh.num_vertices(),3])
	for vertex_index in range(len(vertex_normal_map)): #gives all vertex indices, from 0 to num_vertices()
		for vertex_cells in vertex_to_cells_map[vertex_index]:  #gives all cells for a vertex, eg [0 1 2 3]
			vertex_normal_map[vertex_index] += normalVectors[vertex_cells]*meshClass.orientation[vertex_cells] # cellOrientation(triangles, vertex_cells) #adds all normal vectors of the corresponding cells and check for Orientation
		#if all the normals happen to cancel each other out, apply a small random number to each cells normal vector and calculate from that.
		if LA.norm(vertex_normal_map[vertex_index]) == 0:
			vertex_normal_map[vertex_index] = np.zeros(3)
			for vertex_cells in vertex_to_cells_map[vertex_index]:  #gives all cells for a vertex, eg [0 1 2 3]
				vertex_normal_map[vertex_index] += np.random.randint(-500,500)/3000 * normalVectors[vertex_cells]*meshClass.orientation[vertex_cells]
			
		vertex_normal_map[vertex_index] *= 1.0/LA.norm(vertex_normal_map[vertex_index]) #norms said normal vector
	return vertex_normal_map


# void function, determines Cdc42 concentration extremes and writes them into meshClass
def fillConcentrationExtremes(meshClass):
	maximumCurrentSolutionFunctionValue = 0
	try:
		minimumCurrentSolutionFunctionValue = meshClass.currentSolutionFunction(Cell(meshClass.boundaryMesh, 0).midpoint())
	except RuntimeError:
		print("Building bounding_box_tree")
		meshClass.boundaryMesh.bounding_box_tree().build(meshClass.boundaryMesh)
		minimumCurrentSolutionFunctionValue = meshClass.currentSolutionFunction(Cell(meshClass.boundaryMesh, 0).midpoint())
	maxValue_corresponding_cell = None
	meshClass.minimumCellVolume = Cell(meshClass.boundaryMesh, 0).volume()
	for cellsObject in cells(meshClass.boundaryMesh):
		try:
			tempMidpoint = meshClass.currentSolutionFunction(cellsObject.midpoint())
		except RuntimeError:
			print("Building bounding_box_tree")
			meshClass.boundaryMesh.bounding_box_tree().build(meshClass.boundaryMesh)
			tempMidpoint = meshClass.currentSolutionFunction(cellsObject.midpoint())

		if tempMidpoint> maximumCurrentSolutionFunctionValue:
			maximumCurrentSolutionFunctionValue = tempMidpoint
			maxValue_corresponding_cell = cellsObject
		if tempMidpoint < minimumCurrentSolutionFunctionValue:
			minimumCurrentSolutionFunctionValue = tempMidpoint
			minValue_corresponding_cell = cellsObject
		if cellsObject.volume() < meshClass.minimumCellVolume:
			meshClass.minimumCellVolume = cellsObject.volume()
	meshClass.maximumCurrentSolutionFunctionValue = maximumCurrentSolutionFunctionValue
	meshClass.maxValue_corresponding_cell = maxValue_corresponding_cell
	meshClass.minimumCurrentSolutionFunctionValue = minimumCurrentSolutionFunctionValue
	meshClass.minValue_corresponding_cell = minValue_corresponding_cell




# returns an array containing the lengths of each cells edges. The first index is the cell.index(), the second are the edges(cell).length().
# Since the edges are not globally indexed like the cells, it is assumed that the iterator always calls the edges in the same order.
# there seems to be some sort of indexing, but only 84,78,77.
def cell_edges_lengths_map(meshClass):
	mesh = meshClass.boundaryMesh
	arrayHoldingLengths = np.asarray([[0,0,0]]*mesh.num_cells())
	sum1 = 0
	for cells132 in cells(mesh):
		edgeCounter = 0
		for edges1 in edges(cells132):
			arrayHoldingLengths[cells132.index()][edgeCounter] = edges1.length()
			edgeCounter += 1
	return arrayHoldingLengths


#calculates which cells are supposed to grow based on a growthThreshold. Returns a list/array with cell.index()'s that should be grown in the next step.
def cellGrowthDeterminingArray(meshClass, u_sum, u, activeSurfaceSource, growthThreshold):
	if activeSurfaceSource == True:
		mesh = meshClass.boundaryMesh
		tempCellGrowthDeterminingArray = []
		for tempCells in cells(mesh):
			testTempCellsMidpoint = tempCells.midpoint()
			if u(testTempCellsMidpoint) >= u_sum*growthThreshold:
				tempCellGrowthDeterminingArray.append(tempCells.index())
		return tempCellGrowthDeterminingArray
	else:
		numberOfCells = meshClass.boundaryMesh.num_cells()
		return list(range(numberOfCells)) #np.arange(numberOfCells).tolist()

#get the vertices that should be grown based on the cells that should be grown. Returns a list with unique vertices.
def verticesToGrow(cellGrowthDeterminingArray, cell_to_vertices_map):
	verticesToGrow = []	
	for cellsToBeGrown in range(len(cellGrowthDeterminingArray)): #e.g. [0 3 4 7 22 133]
		for ii in range(3): #every cell has 3 vertices
			verticesToGrow.append(int(cell_to_vertices_map[cellGrowthDeterminingArray[cellsToBeGrown]][ii]))
	#to have only unique entries:
	verticesToGrow = list(set(verticesToGrow))
	return verticesToGrow

#void function, grows a vertex in the by the vertex normal map given direction. It just adds the normal vector scaled by the growthFactor.
def vertexGrowth(meshcoord, verticesToGrow, vertex_normal_map, growthFactor):
	for vertices in verticesToGrow:
		meshcoord[vertices] += growthFactor*vertex_normal_map[vertices]

#grows relative to the averaged u() value of all cells that contain the vertex to be grown
# basically verticesToGrow() and vertexGrowth with a custom growthFactor
def relativeVertexGrowth(meshClass, growthThresh, growthFactor, falseGrowthThreshold=0, activeSurface=True):
	verticesToGrow = []	
	for cellsToBeGrown in range(len(meshClass.cellGrowthDeterminingArray)): #e.g. [0 3 4 7 22 133]
		for ii in range(3): #every cell has 3 vertices
			verticesToGrow.append(int(meshClass.cell_to_vertices_map[meshClass.cellGrowthDeterminingArray[cellsToBeGrown]][ii]))
	#to have only unique entries:
	verticesToGrow = list(set(verticesToGrow))
	for vertices in verticesToGrow:
		tempGrowthFactor = 0
		for cellsIndices in meshClass.vertex_to_cells_map[vertices]:
			if activeSurface == True:
				testCell = Cell(meshClass.boundaryMesh, cellsIndices)
				testCellMidpoint = testCell.midpoint()
				tempGrowthFactor += meshClass.currentSolutionFunction(testCellMidpoint) - growthThresh
			else:
				testCell = Cell(meshClass.boundaryMesh, cellsIndices)
				testCellMidpoint = testCell.midpoint()
				tempGrowthFactor += (meshClass.currentSolutionFunction(testCellMidpoint) - falseGrowthThreshold)
		# possibly obsolete
		if activeSurface == True:
			tempGrowthFactor = tempGrowthFactor * 1/len(meshClass.vertex_to_cells_map[vertices]) * growthFactor
		else:
			tempGrowthFactor = tempGrowthFactor * 1/len(meshClass.vertex_to_cells_map[vertices]) * growthFactor
			if tempGrowthFactor < 0:
				tempGrowthFactor = 0
		meshClass.coordinates[vertices] += tempGrowthFactor*meshClass.vertex_normal_map[vertices]

# refines a mesh based on the refineFunction. Similar to a void function in cpp, just that it returns an updated listOfPDE
# for convenience
def myRefinement(meshClass, refineFunction, listOfPDE, usedMeshesList, straightLengthFactor):
	del listOfPDE
	#marker = MeshFunction("bool", mesh, mesh.topology().dim(), True)

	meshClass.boundaryMesh = refine(meshClass.boundaryMesh, refineFunction)
	# pc = meshClass.boundaryMesh.data().array("parent_vertex_indices", meshClass.boundaryMesh.topology().dim())
	# print(pc)
	# print(len(pc))
	# print(len(meshClass.coordinates))

	meshClass.boundaryMesh.bounding_box_tree().build(meshClass.boundaryMesh)

	meshClass.functionSpace = FunctionSpace(meshClass.boundaryMesh, 'CG', 1)
	#meshClass.trialFunction = interpolate(meshClass.trialFunction, meshClass.functionSpace)
	meshClass.trialFunction = interpolate(u_D, meshClass.functionSpace)
	meshClass.testFunction = TestFunction(meshClass.functionSpace)
	meshClass.currentSolutionFunction = Function(meshClass.functionSpace)

	meshClass.stimulus.element = meshClass.functionSpace.ufl_element()
	meshClass.PDE = inner((meshClass.currentSolutionFunction - meshClass.trialFunction) / k, meshClass.testFunction)*dx - Dm*inner(nabla_grad(meshClass.trialFunction), nabla_grad(meshClass.testFunction))*dx \
		 - (1.0-meshClass.h)*(nu*k0 + (nu*K*meshClass.trialFunction**2)/(Km**2 + meshClass.trialFunction**2))*meshClass.testFunction*dx + eta*meshClass.trialFunction*meshClass.testFunction*dx - meshClass.stimulus*meshClass.testFunction*dx

	listOfPDE = [None]*len(usedMeshesList)
	for jj in range(len(usedMeshesList)):
		listOfPDE[jj] = usedMeshesList[jj].PDE

	#CELL ORIENTATION!
	#write it to File
	meshClass.fileName = 'mesh_%s_unordered.xml' % meshClass.name
	File(meshClass.fileName) << meshClass.boundaryMesh 
	#parse it back in to extract the Orientation
	meshClass.tree = ET.parse(meshClass.fileName)
	meshClass.triangles = meshClass.tree.findall('mesh/cells/triangle')
	#order the mesh so it can be iterated over
	meshClass.boundaryMesh.order()
	#get vertex coordinates for growing purposes
	meshClass.coordinates = meshClass.boundaryMesh.coordinates()
	#initialize vertex edge connectivity
	meshClass.boundaryMesh.init(0,1)
	
	meshClass.normalVectors = cellNormals(meshClass.boundaryMesh)
	#save every cells orientation as an array
	meshClass.orientation = meshClassOrientation(meshClass, straightLengthFactor)

	meshClass.vertex_to_cells_map = vertex_to_cells_map(meshClass)
	meshClass.vertex_normal_map = vertex_normal_map(meshClass, meshClass.vertex_to_cells_map, meshClass.normalVectors)
	meshClass.cell_to_vertices_map = cell_to_vertices_map(meshClass)
	meshClass.cell_markers_boundary = MeshFunction('bool', meshClass.boundaryMesh, meshClass.boundaryMesh.topology().dim(), False)

	return listOfPDE


# adaption of myRefinement. Updates all the maps while maintaining the old Estar, cell_volumes array and initial_cell_edges_and_opposite_angle_map values of cells which have not
# been changed.
def TRBSmyRefinement(meshClass, listOfPDE, straightLengthFactor, E_value = 0.8):

	#do a saving step so the before and after refinement situation can be viewed in paraview
	meshClass.saveFile << meshClass.currentSolutionFunction
	print( meshClass.name, ' is being refined!')
	print( 'number of cells before:', meshClass.boundaryMesh.num_cells())

	# fill a list with the cells vertex_coordinates. Basically meshcoords, but without any pointers
	before_refinement_list = [None]*meshClass.boundaryMesh.num_cells()
	for cellsObjects in cells(meshClass.boundaryMesh):
		before_refinement_list[cellsObjects.index()] = cellsObjects.get_vertex_coordinates()
	before_refinement_list = deepcopy(before_refinement_list)

	vertices_deepcopy = deepcopy(meshClass.coordinates)

	# do the actual refinement
	try:
		listOfPDE = myRefinement(meshClass, meshClass.cell_markers_boundary, listOfPDE, usedMeshesList, straightLengthFactor)
	except RuntimeError:
		print("Building bounding_box_tree")
		meshClass.boundaryMesh.bounding_box_tree().build(meshClass.boundaryMesh)
		listOfPDE = myRefinement(meshClass, meshClass.cell_markers_boundary, listOfPDE, usedMeshesList, straightLengthFactor)


	#Refinement_has_happened = True
	#number_of_refinements += 1

	# check if any previously existing vertex coordinates have been changed
	for olk in range(len(vertices_deepcopy)):
		if (vertices_deepcopy[olk] != meshClass.coordinates[olk]).any():
			print(olk)
			print("CHANGED from", vertices_deepcopy[olk], " TO: ", meshClass.coordinates[olk])

	print( 'number of cells after:', meshClass.boundaryMesh.num_cells())

	refinement_dict = {}

	# corresponding to before_refinement_list
	after_refinement_list = [None]*meshClass.boundaryMesh.num_cells()
	for cellsObjects in cells(meshClass.boundaryMesh):
		after_refinement_list[cellsObjects.index()] = cellsObjects.get_vertex_coordinates()

	# create a dictionary which maps the old vertex coordinates to the new ones, if present
	## refinement_dict, key = new cell, value = old cell
	for iii in range(len(before_refinement_list)):
		try:
			refinement_dict[after_refinement_list.index(before_refinement_list[iii])] = iii
		except ValueError:
			None
			#print("ValueError in the creation of my refinement_dict!")
		#print(next((j for j,x in enumerate(after_refinement_list[:1237]) if x==before_refinement_list[i]), None))
		#if before_refinement_list[i] in after_refinement_list:
		#	print("not there!", i)
	# test = []
	# for jjj in range(len(after_refinement_list)):
	# 	try:

	#print(refinement_dict)


	# update all relevant maps
	meshClass.cell_to_vertices_map = cell_to_vertices_map(meshClass)
	meshClass.vertex_to_edges_map = vertex_to_edges_map(meshClass)
	meshClass.cell_to_edges_map = cell_to_edges_map(meshClass)
	meshClass.cell_volumes = cell_volumes(meshClass)
	meshClass.cell_to_vertices_map = cell_to_vertices_map(meshClass)
	meshClass.vertex_to_cells_map = vertex_to_cells_map(meshClass)
	# create a new initial_cell_edges_..._map. Most conveniently achieved by just taking the current one
	new_initial_cell_edges_and_opposite_angle_map = cell_edges_and_opposite_angle_map(meshClass)
	# create an updated Estar for this mesh
	new_Estar = [E_value/(1.0-poisson_ratio1**2)]*meshClass.boundaryMesh.num_cells()
	# use my refinement_dict to change all cells values which havent been refined to the actual initial values.
	# This way the newly created cells are relaxed and can expand further

	for keys in refinement_dict.keys():
		for iii in range(3):
			#print(new_initial_cell_edges_and_opposite_angle_map[keys][i][1])
			new_initial_cell_edges_and_opposite_angle_map[keys][iii][1] = meshClass.initial_cell_edges_and_opposite_angle_map[refinement_dict[keys]][iii][1]
			new_initial_cell_edges_and_opposite_angle_map[keys][iii][2] = meshClass.initial_cell_edges_and_opposite_angle_map[refinement_dict[keys]][iii][2]
			#print(new_initial_cell_edges_and_opposite_angle_map[keys][iii][1])
			#print(' ')

	# if a cell previously had a different Estar value, keep it
	for values in refinement_dict.values():
		if values < len(meshClass.Estar):
			if meshClass.Estar[values] != E_value/(1.0-poisson_ratio1**2): #< 0.00001:
				new_Estar[list(refinement_dict.keys())[list(refinement_dict.values()).index(values)]] = meshClass.Estar[values] #meshClass.Estar[refinement_dict[values]]
		# only update the new cells, keep the old cells values. This is only needed for the cell_volumes_history to remain correct. 
		meshClass.cell_volumes[list(refinement_dict.keys())[list(refinement_dict.values()).index(values)]] = meshClass.cell_volumes_history[-1][values]
	#for cells11 in cells(meshClass.boundaryMesh):
	#	if cells11.index() not in refinement_dict.values():
	#		new_Estar[cells11.index()] = 0.8/(1.0-poisson_ratio1**2)
	meshClass.previous_initial_cell_edges_and_opposite_angle_map = deepcopy(meshClass.initial_cell_edges_and_opposite_angle_map)
	meshClass.initial_cell_edges_and_opposite_angle_map = deepcopy(new_initial_cell_edges_and_opposite_angle_map)
	# else:
	# 	meshClass.previous_initial_cell_edges_and_opposite_angle_map = deepcopy(meshClass.initial_cell_edges_and_opposite_angle_map)
	# 	meshClass.parent_cells = meshClass.boundaryMesh.data().array("parent_cell", meshClass.boundaryMesh.topology().dim())

	# 	for new_cell_iter in range(len(meshClass.parent_cells)):
	# 		parent_cell = meshClass.parent_cells[new_cell_iter]

	meshClass.Estar = new_Estar
	

	#do a saving step so the before and after refinement situation can be viewed in paraview
	meshClass.saveFile << meshClass.currentSolutionFunction

	return listOfPDE


# calculate which cells are new after refinement. Returns a list with the new cell.index()
def new_cells_after_refinement(meshClass):

	parent_cells = meshClass.boundaryMesh.data().array("parent_cell", meshClass.boundaryMesh.topology().dim())
	# print("len:", len(parent_cells))

	# find out which cells are new
	parents_already_seen = []
	parents_that_multiplied = []
	for parentCells in meshClass.boundaryMesh.data().array("parent_cell", meshClass.boundaryMesh.topology().dim()):
		if parentCells not in parents_already_seen:
			parents_already_seen.append(parentCells)
		else:
			parents_that_multiplied.append(parentCells)
	
	new_cells = []
	for cellIndex in range(meshClass.boundaryMesh.num_cells()):
		if parent_cells[cellIndex] in parents_that_multiplied:
			new_cells.append(cellIndex)
	return new_cells




#Aim is to make a BoxMesh around my sources which i can project them on. It finds the min and max Points and makes them even smaller/bigger by "size".
#Also adds together the desired simuli Expressions which is needed for projection. Returns the projection, a plotable dolfin class.
def savePheromone(listOfMeshClassesToAdd, size, resolutionX, resolutionY, resolutionZ, twoDStimulus = False):
	minPoint = [0,0,0]
	maxPoint = [0,0,0]
	#necessary! With activeSurfaceSource = True, my sources become dolfin Points which cant be deepcopied as a whole, must be piecewise since
	#it is then returned as a skalar
	minPoint[0] = deepcopy(listOfMeshClassesToAdd[0].source[0])
	minPoint[1] = deepcopy(listOfMeshClassesToAdd[0].source[1])
	minPoint[2] = deepcopy(listOfMeshClassesToAdd[0].source[2])
	maxPoint[0] = deepcopy(listOfMeshClassesToAdd[0].source[0])
	maxPoint[1] = deepcopy(listOfMeshClassesToAdd[0].source[1])
	maxPoint[2] = deepcopy(listOfMeshClassesToAdd[0].source[2])
	addedStimuli = None

	for meshClasses in listOfMeshClassesToAdd:
		if meshClasses.source[0] < minPoint[0]:
			minPoint[0] = meshClasses.source[0]
		if meshClasses.source[1] < minPoint[1]:
			minPoint[1] = meshClasses.source[1]
		if meshClasses.source[2] < minPoint[2]:
			minPoint[2] = meshClasses.source[2]

		if meshClasses.source[0] > maxPoint[0]:
			maxPoint[0] = meshClasses.source[0]
		if meshClasses.source[1] > maxPoint[1]:
			maxPoint[1] = meshClasses.source[1]
		if meshClasses.source[2] > maxPoint[2]:
			maxPoint[2] = meshClasses.source[2]
		if twoDStimulus == False:
			if addedStimuli == None:
				addedStimuli = meshClasses.stimulus
			else:
				addedStimuli += meshClasses.stimulus
		else:
			if addedStimuli == None:
				addedStimuli = meshClasses.twoDStimulus
			else:
				addedStimuli += meshClasses.twoDStimulus

	minPoint[0] = minPoint[0] - size
	minPoint[1] = minPoint[1] - size
	minPoint[2] = minPoint[2] - size
	maxPoint[0] = maxPoint[0] + size
	maxPoint[1] = maxPoint[1] + size
	maxPoint[2] = maxPoint[2] + size

	if twoDStimulus == False:
		pheromoneMesh = BoxMesh(Point(minPoint[0],minPoint[1],minPoint[2]), Point(maxPoint[0],maxPoint[1],maxPoint[2]), resolutionX, resolutionY, resolutionZ)
		pheromoneMeshFunctionSpace = FunctionSpace(pheromoneMesh, "CG", 1)

		projectionOfPheromone = project(addedStimuli, pheromoneMeshFunctionSpace)
	else:
		pheromoneMesh = RectangleMesh(Point(minPoint[0],minPoint[1]), Point(maxPoint[0],maxPoint[1]), resolutionX, resolutionY)
		pheromoneMeshFunctionSpace = FunctionSpace(pheromoneMesh, "CG", 1)

		projectionOfPheromone = project(addedStimuli, pheromoneMeshFunctionSpace)
	return projectionOfPheromone







#######################################################################



#calculates all vertex.index() that belong to a cell. Returns a map where the first index is the cell.index() and the corresponding entries are the vertex.index()'s
def cell_to_vertices_map(meshClass):
	mesh = meshClass.boundaryMesh
	tempArray = np.empty([mesh.num_cells(),3], int)
	for cells2 in cells(mesh):
		for j in range(3):
			tempArray[cells2.index()][j] = cells2.entities(0)[j]
	return tempArray

#calculates all edges.index() that belong to a cell. Returns array where first index is cell.index() and entries are edge.index()'s
def cell_to_edges_map(meshClass):
	mesh = meshClass.boundaryMesh
	tempArray = np.empty([mesh.num_cells(),3], int)
	for cells3 in cells(mesh):
		for j in range(3):
			tempArray[cells3.index()][j] = cells3.entities(1)[j]
	return tempArray

#calculates all cell.index() that belong to an edge. Returns array where first index is edge.index() and entries are cell.index()'s
def edges_to_cells_map(meshClass):
	mesh = meshClass.boundaryMesh
	tempArray = np.empty([mesh.num_edges(),2], int)
	D = mesh.topology().dim()
	mesh.init(D-1,D) # Build connectivity between facets and cells
	for facet in facets(mesh):
		tempArray[facet.index()] = facet.entities(D)
	return tempArray

# Returns the unit vector of the vector. 
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

# Returns the angle in radians between vectors 'v1' and 'v2'
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# returns the cell volumes where the index is the cell.index()
def cell_volumes(meshClass):
	mesh = meshClass.boundaryMesh
	volumes = [0]*mesh.num_cells()
	for cellsObject in cells(mesh):
		volumes[cellsObject.index()] = cellsObject.volume()
	return volumes


# returns an array containing the lengths of each cells edges. The first index is the cell.index(), the second are the edges numerated through, the third
# are the edge.index() (0), edge.lengths() (1) and opposing angle (2).
def cell_edges_and_opposite_angle_map(meshClass):
	mesh = meshClass.boundaryMesh
	cell_to_vertices_map = meshClass.cell_to_vertices_map
	#print("cell_edges_and_opposite_angle_map called")
	#mesh = meshClass
	#np.asarray is necessary since the multiplication of nested lists results in association "faults"
	arrayHoldingEdgeLengthsAndAngles = np.asarray([[[0,0.0,0.0],[0,0.0,0.0],[0,0.0,0.0]]]*mesh.num_cells(), float)
	#print(arrayHoldingEdgeLengthsAndAngles.shape)
	for cells132 in cells(mesh):
		#print(cells132.index())
		#additional iterator
		edgeCounter = 0
		for edges1 in edges(cells132):
			#write the edge
			#print("edge.index():", edges1.index())
			arrayHoldingEdgeLengthsAndAngles[cells132.index()][edgeCounter][0] = edges1.index()
			arrayHoldingEdgeLengthsAndAngles[cells132.index()][edgeCounter][1] = edges1.length()
			#print("edges1.length()", edges1.length(), arrayHoldingEdgeLengthsAndAngles[cells132.index()][edgeCounter][0])
			tempVertexSavingList = [0,0]
			#additional additional iterator
			vertexCounter = 0
			#save the vertices spanning edge1
			for vertex in vertices(edges1):
				tempVertexSavingList[vertexCounter] = vertex.index()
				vertexCounter += 1
			#determine the vertex opposite to edge1
			theOpposingVertex = list(set(cell_to_vertices_map[cells132.index()]) - set(tempVertexSavingList))
			#print("tempVertexSavingList:", tempVertexSavingList)
			#print("cell_to_vertices_map:", cell_to_vertices_map(Mesh)[cells132.index()])
			#print("theOpposingVertex:", theOpposingVertex)
			tempVector1 = mesh.coordinates()[int(theOpposingVertex[0])] - mesh.coordinates()[tempVertexSavingList[0]]
			tempVector2 = mesh.coordinates()[int(theOpposingVertex[0])] - mesh.coordinates()[tempVertexSavingList[1]]
			#calculate and write the angle(degrees)
			arrayHoldingEdgeLengthsAndAngles[cells132.index()][edgeCounter][2] = angle_between(tempVector1, tempVector2)
			edgeCounter += 1
	#print(arrayHoldingEdgeLengthsAndAngles)
	return arrayHoldingEdgeLengthsAndAngles

# returns the opposing angle of a given cells given edge
def cell_edge_opposing_angle(cell, edge, cell_edges_and_opposite_angle_map):
	try:
		edgeIndex = edge.index()
	except TypeError:
		edgeIndex = edge
	except AttributeError:
		edgeIndex = edge
	try:
		cellIndex = cell.index()
	except AttributeError:
		cellIndex = cell

	for i in range(3):
		if edgeIndex == cell_edges_and_opposite_angle_map[cellIndex][i][0]:
			#print(cell_edges_and_opposite_angle_map[cellIndex][i][2])
			return cell_edges_and_opposite_angle_map[cellIndex][i][2]

# returns the edge length of a given cells given edge
def cell_edge_length(cell, edge, cell_edges_and_opposite_angle_map):
	try:
		edgeIndex = edge.index()
	except TypeError:
		edgeIndex = edge
	except AttributeError:
		edgeIndex = edge
	try:
		cellIndex = cell.index()
	except AttributeError:
		cellIndex = cell
	for i in range(3):
		if edgeIndex == cell_edges_and_opposite_angle_map[cellIndex][i][0]:
			#print(cell_edges_and_opposite_angle_map[cellIndex][i][1])
			return cell_edges_and_opposite_angle_map[cellIndex][i][1]


# see the Delingette 2008 paper "Biquadratic and quadratic springs..."
def trace_of_C(cell, cell_edges_and_opposite_angle_map):
	return 1/cell.volume() * ( cell_edges_and_opposite_angle_map[cell.index()][0][0]**2 * np.cos(cell_edges_and_opposite_angle_map[cell.index()][0][1])  
							 + cell_edges_and_opposite_angle_map[cell.index()][1][0]**2 * np.cos(cell_edges_and_opposite_angle_map[cell.index()][1][1])
							 + cell_edges_and_opposite_angle_map[cell.index()][2][0]**2 * np.cos(cell_edges_and_opposite_angle_map[cell.index()][2][1]))
def trace_of_Epsilon(cell, cell_edges_and_opposite_angle_map, initial_cell_edges_and_opposite_angle_map):
	return 1/cell.volume() * ( (cell_edges_and_opposite_angle_map[cell.index()][0][0]**2 - initial_cell_edges_and_opposite_angle_map[cell.index()][0][0]**2) * np.cos(cell_edges_and_opposite_angle_map[cell.index()][0][1])
							 + (cell_edges_and_opposite_angle_map[cell.index()][1][0]**2 - initial_cell_edges_and_opposite_angle_map[cell.index()][1][0]**2) * np.cos(cell_edges_and_opposite_angle_map[cell.index()][1][1])
							 + (cell_edges_and_opposite_angle_map[cell.index()][2][0]**2 - initial_cell_edges_and_opposite_angle_map[cell.index()][2][0]**2) * np.cos(cell_edges_and_opposite_angle_map[cell.index()][2][1]))
#k_T_i
def tensile_stiffness_of_edge(meshClass, cell, edge, cell_edges_and_opposite_angle_map, Thickness = 0.1):
	#mesh = meshClass.boundaryMesh
	mesh = meshClass
	try:
		edgeIndex = edge.index()
	except AttributeError:
		edgeIndex = edge
	try:
		cellIndex = cell.index()
	except AttributeError:
		cell = Cell(mesh, cell)
	return Estar*Thickness*(2*(1/tan(cell_edges_and_opposite_angle_map[cell.index()][edge][1]))**2 + 1 - poisson_ratio)/(16*(1 - poisson_ratio**2)*cell.volume())
#c_T_i
def angular_stiffness_of_edge(meshClass, cell, edge_i, edge_j, cell_edges_and_opposite_angle_map, Thickness = 0.1):
	#mesh = meshClass.boundaryMesh
	mesh = meshClass
	try:
		edgeIndex = edge.index()
	except AttributeError:
		edgeIndex = edge
	try:
		cellIndex = cell.index()
	except AttributeError:
		cell = Cell(mesh, cell)
	return Estar*Thickness*(2*(1/tan(cell_edges_and_opposite_angle_map[cell.index()][edge_i][1]))*(1/tan(cell_edges_and_opposite_angle_map[cell.index()][edge_j][1])) - 1 + poisson_ratio)/(16*(1 - poisson_ratio**2)*cell.volume())
#TRBS TRiangluar Biquadratic Springs, energy per cell.
#Currently not used, force is the derivative of this.
def energy_TRBS(meshClass, Thickness = 0.1, poisson_ratio = 0.5):

	mesh = meshClass.boundaryMesh

	vertex_to_edges_map = meshClass.vertex_to_edges_map
	cell_to_vertices_map = meshClass.cell_to_vertices_map

	initial_cell_edges_and_opposite_angle_map = meshClass.initial_cell_edges_and_opposite_angle_map
	cell_edges_and_opposite_angle_map = meshClass.cell_edges_and_opposite_angle_map

	# contruct the array which holds force-vectors
	energy_on_cells = np.zeros(mesh.num_cells())
	# loop over all cells and calculate the resulting forces on each cells vertices. Theses force-vectors are then simply added.
	for cellObject in cells(mesh):
		# which vertices make up the current cell
		vertices_to_consider = cell_to_vertices_map[cellObject.index()]
		# get the current cells edges
		current_cells_edge_Indices = [0,0,0]
		edgeCounter = 0
		for edgeObject in edges(cellObject):
			current_cells_edge_Indices[edgeCounter] = edgeObject.index()
			edgeCounter += 1

		#l1 -> l3 are not the actual adge lengths but rather the edges which are relevant for my force computation. 
		#to get the lengths one has to use the maps with l1->l3 as input.
		# calculate all the necessary factors
		v1 = vertices_to_consider[0]
		l1 = list(set(current_cells_edge_Indices) - set(vertex_to_edges_map[v1]))

		v2 = vertices_to_consider[1]
		l2 = list(set(current_cells_edge_Indices) - set(vertex_to_edges_map[v2]))

		v3 = vertices_to_consider[2]
		l3 = list(set(current_cells_edge_Indices) - set(vertex_to_edges_map[v3]))

		k1 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l1, cell_edges_and_opposite_angle_map)))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])
		k2 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l2, cell_edges_and_opposite_angle_map)))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])
		k3 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l3, cell_edges_and_opposite_angle_map)))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])

		delta_l1 = cell_edge_length(cellObject.index(), l1, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellObject.index(), l1, initial_cell_edges_and_opposite_angle_map)**2
		delta_l2 = cell_edge_length(cellObject.index(), l2, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellObject.index(), l2, initial_cell_edges_and_opposite_angle_map)**2
		delta_l3 = cell_edge_length(cellObject.index(), l3, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellObject.index(), l3, initial_cell_edges_and_opposite_angle_map)**2

		c1 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l2, cell_edges_and_opposite_angle_map)))*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l3, cell_edges_and_opposite_angle_map))) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])
		c2 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l1, cell_edges_and_opposite_angle_map)))*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l3, cell_edges_and_opposite_angle_map))) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])
		c3 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l1, cell_edges_and_opposite_angle_map)))*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l2, cell_edges_and_opposite_angle_map))) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])




		energy_on_cells[cellObject.index()] += 1/4 * k1 * delta_l1**2 \
											+ 1/4 * k2 * delta_l2**2 \
											+ 1/4 * k3 * delta_l3**2 \
											+ 1/2 * c3 * delta_l1*delta_l2 \
											+ 1/2 * c2 * delta_l1*delta_l3 \
											+ 1/2 * c3 * delta_l2*delta_l1 \
											+ 1/2 * c1 * delta_l2*delta_l3 \
											+ 1/2 * c2 * delta_l3*delta_l1 \
											+ 1/2 * c1 * delta_l3*delta_l2


	return energy_on_cells


# calculates all edges attached to a vertex. Returns an array where the index is the vertex.index() and the entries are the edge.index()'s
def vertex_to_edges_map(meshClass):
	mesh = meshClass.boundaryMesh
	#mesh = meshClass
	vertex_to_edges_map_untrimmed = np.zeros([mesh.num_vertices(),30], dtype=int)
	for v in vertices(mesh):
		edgeCounter = 0
		for edge in edges(v):
			vertex_to_edges_map_untrimmed[v.index()][edgeCounter] = edge.index()
			edgeCounter += 1
	vertex_to_edges_map = [0]*mesh.num_vertices()
	#trim all zeros from the back 'b', leave the front ones since there is a cell.index() = 0. Might lose some back zeros
	for trimming in range(len(vertex_to_edges_map_untrimmed)):
		vertex_to_edges_map[trimming] = np.trim_zeros(vertex_to_edges_map_untrimmed[trimming], 'b')
	vertex_to_edges_map = np.asarray(vertex_to_edges_map)
	return vertex_to_edges_map

# #variable distance is 0.006 to 0.013. Those functions are fitted for that. For fit, see test7.py
# def decliningFunction(x, xShift = 0.001, a = 1.36923185e-01, b = 1.85658848e+02, c = 1.00911136e+03, d = 1.44366811e+00):
# 	return a * np.exp(b * (x+xShift)) * np.cos(c*(x+0.001)) + d
# def ascendingFunction(x, xShift = 0, a = 5.34904128e-01, b = 6.24319490e+01, c = 9.63434956e+02, d = 1.50404192e+00):
# 	return (a * np.exp(b * (x + xShift)) * np.cos(c*(x + xShift)) + d)/1.23 +0.29

# def EstarEasy(meshClass, baseThreshold = 0.006, shaftThreshold = 0.0099, tipValue = 2.7, shaftValue = 1.1, baseValue = 2.5, poisson_ratio = 0.5):
# 	list_of_Estar = [shaftValue]*meshClass.boundaryMesh.num_cells()
# 	try:
# 		for cell2010 in cells(meshClass.boundaryMesh):
# 			tempCellMidpoint = cell2010.midpoint()
# 			if meshClass.currentSolutionFunction(tempCellMidpoint) < baseThreshold:
# 				list_of_Estar[cell2010.index()] = baseValue/(1.0-poisson_ratio**2)

# 			elif meshClass.currentSolutionFunction(tempCellMidpoint) < shaftThreshold:
# 				list_of_Estar[cell2010.index()] = shaftValue/(1.0-poisson_ratio**2)
# 				shaftList.append(cell2010.index())

# 			else:
# 				list_of_Estar[cell2010.index()] = tipValue/(1.0-poisson_ratio**2)
# 				tipList.append(cell2010.index())
# 	except RuntimeError:
# 		print("Building bounding_box_tree")
# 		meshClass.boundaryMesh.bounding_box_tree().build(meshClass.boundaryMesh)
# 		for cell2010 in cells(meshClass.boundaryMesh):
# 			tempCellMidpoint = cell2010.midpoint()
# 			if meshClass.currentSolutionFunction(tempCellMidpoint) < baseThreshold:
# 				list_of_Estar[cell2010.index()] = baseValue/(1.0-poisson_ratio**2)

# 			elif meshClass.currentSolutionFunction(tempCellMidpoint) < shaftThreshold:
# 				list_of_Estar[cell2010.index()] = shaftValue/(1.0-poisson_ratio**2)
# 				shaftList.append(cell2010.index())

# 			else:
# 				list_of_Estar[cell2010.index()] = tipValue/(1.0-poisson_ratio**2)
# 				tipList.append(cell2010.index())
# 	return list_of_Estar

# returns the Estar-value for a given x within the range of xmin and xmax.
# The goal was a smooth transistion between the base-value of E = 2.5 and the shaft value of E = 0.8. So a arctan function was used, which 
# lies between 0 and 20, where myarctan(0) = 0.8 and myarctan(20) = 2.5. The x-values represent the Cdc42-concentration where the y-values are E.
# To achieve adaptability, the given x value is compared against the given bounds for percentage of x_max. Said percentage is then multiplied by 20, the
# max value of myarctan. That way i can map my given range to the range of myarctan easily.
def EstarShaft(x ,x_min, x_max, poisson_ratio1 = 0.5):
	if x < x_min or x > x_max:
		raise ValueError("Wrong calling of EstarShaft method. x must be between x_min and x_max")
	fitted_x = 20 * (x - x_min)/(x_max - x_min)
	a=0.59076415
	b=5
	c=1.68864407
	d=1
	return((-a*np.arctan((fitted_x - b)/d) + c)/(1.0-poisson_ratio1**2))




# calculates Estar for different Cdc42 concentrations on the surface. Deprecated
def Estar(meshClass, baseThreshold = 0.006, decliningFunctionThreshold = 0.0082, shaftThreshold = 0.0099, ascendingFunctionThreshold = 0.013, tipValue = 2.7, shaftValue = 1.1, baseValue = 2.5, poisson_ratio = 0.5):
	#Estar = E/(1-poisson_ratio**2)
	list_of_Estar = [shaftValue]*meshClass.boundaryMesh.num_cells()
	try:
		for cell2010 in cells(meshClass.boundaryMesh):
			tempCellMidpoint = cell2010.midpoint()
			if meshClass.currentSolutionFunction(tempCellMidpoint) < baseThreshold:
				list_of_Estar[cell2010.index()] = baseValue/(1.0-poisson_ratio**2)
			elif meshClass.currentSolutionFunction(tempCellMidpoint) < decliningFunctionThreshold:
				list_of_Estar[cell2010.index()] = decliningFunction(meshClass.currentSolutionFunction(tempCellMidpoint))/(1.0-poisson_ratio**2)
				#shaftList.append(cell2010.index())
			elif meshClass.currentSolutionFunction(tempCellMidpoint) < shaftThreshold:
				list_of_Estar[cell2010.index()] = shaftValue/(1.0-poisson_ratio**2)
				shaftList.append(cell2010.index())
			elif meshClass.currentSolutionFunction(tempCellMidpoint) < ascendingFunctionThreshold:
				list_of_Estar[cell2010.index()] = ascendingFunction(meshClass.currentSolutionFunction(tempCellMidpoint))/(1.0-poisson_ratio**2)
				#shaftList.append(cell2010.index())
			else:
				list_of_Estar[cell2010.index()] = tipValue/(1.0-poisson_ratio**2)
				tipList.append(cell2010.index())
	except RuntimeError:
		print("Building bounding_box_tree")
		meshClass.boundaryMesh.bounding_box_tree().build(meshClass.boundaryMesh)
		for cell2010 in cells(meshClass.boundaryMesh):
			tempCellMidpoint = cell2010.midpoint()
			if meshClass.currentSolutionFunction(tempCellMidpoint) < baseThreshold:
				list_of_Estar[cell2010.index()] = baseValue/(1.0-poisson_ratio**2)
			elif meshClass.currentSolutionFunction(tempCellMidpoint) < decliningFunctionThreshold:
				list_of_Estar[cell2010.index()] = decliningFunction(meshClass.currentSolutionFunction(tempCellMidpoint))/(1.0-poisson_ratio**2)
				#shaftList.append(cell2010.index())
			elif meshClass.currentSolutionFunction(tempCellMidpoint) < shaftThreshold:
				list_of_Estar[cell2010.index()] = shaftValue/(1.0-poisson_ratio**2)
				shaftList.append(cell2010.index())
			elif meshClass.currentSolutionFunction(tempCellMidpoint) < ascendingFunctionThreshold:
				list_of_Estar[cell2010.index()] = ascendingFunction(meshClass.currentSolutionFunction(tempCellMidpoint))/(1.0-poisson_ratio**2)
				#shaftList.append(cell2010.index())
			else:
				list_of_Estar[cell2010.index()] = tipValue/(1.0-poisson_ratio**2)
				tipList.append(cell2010.index())
	return list_of_Estar
	



# my force on vertex as seen in the Delingette 2008 paper "Biquadratic and quadratic springs...".
# The showed Energy per cell was derived over 3 dim to get said force. The full formula can be found at Goldenbogen et all.'s
# "Dynamics of cell wall elasticity pattern shapes the cell during yeast matingmorphogenesis" supplementary, page 3.
# It is assumed that each cell consists of 3 regular and 3 angular biquadratic springs. If some deformation of the mesh happens, these springs
# will excert force to return to their relaxed state.
# The relaxed state is given by the "initial_cell_edges_and_opposite_angle_map", therefore it is assumed that at the beginning the whole
# mesh is relaxed.
# this is a old and spaghetti-code version of my TRBS force. TRBS_force_on_mesh and TRBS_force_on_mesh_parallel offer a more comprehensible implementation.
def TRBS_force_on_vertex(meshClass, vertex, Thickness = 0.1, poisson_ratio = 0.5):
	from joblib import Parallel, delayed
	mesh = meshClass.boundaryMesh
	vertex_to_cells_map = meshClass.vertex_to_cells_map
	vertex_to_edges_map = meshClass.vertex_to_edges_map
	initial_cell_edges_and_opposite_angle_map = meshClass.initial_cell_edges_and_opposite_angle_map
	cell_edges_and_opposite_angle_map = meshClass.cell_edges_and_opposite_angle_map

	#print("calculating force for vertex ", vertex.index())
	#mesh = meshClass.boun
	#all the cells attached to my vertex
	cells_to_consider = vertex_to_cells_map[vertex.index()]
	#print("cells_to_consider", cells_to_consider)
	#all the edges attached to my vertex
	edges_to_consider = list(vertex_to_edges_map[vertex.index()])
	#print("edges_to_consider", edges_to_consider)
	# the accumulating force vector:
	if meshClass.mesh.topology().dim() == 3:
		force_on_vertex = [0.0, 0.0, 0.0]
	elif meshClass.mesh.topology().dim() == 2:
		force_on_vertex = [0.0, 0.0, 0.0]
	#iterate over all the cells
	for cellIndices in cells_to_consider:
		#write down the cells edges.index()
		specific_cell_edges_to_consider = [0,0,0]
		edgeCounter = 0
		for edge in edges(Cell(mesh, cellIndices)):
			specific_cell_edges_to_consider[edgeCounter] = edge.index()
			edgeCounter += 1
		#print("specific_cell_edges_to_consider", specific_cell_edges_to_consider)
		the_opposing_edge = list(set(specific_cell_edges_to_consider) - set(edges_to_consider))
		#print("the oppsing edge:", the_opposing_edge)
		# save all the relevant edges, more specifially those attached to the given vertex within the cell. Edges.index()!
		relevant_edges = list(set(edges_to_consider).intersection(specific_cell_edges_to_consider))
		#print("relevant_edges", relevant_edges)
		# gather all the data to those edges
		initial_relevant_edges_and_angles = [[0,0.0],[0,0.0]]
		relevant_edges_and_angles = [[0,0.0],[0,0.0]]
		edgeCounter2 = 0
		#edge indices:
		for edge in relevant_edges:
			#this edge here is j and then k
			initial_relevant_edges_and_angles[edgeCounter2][0] = edge
			initial_relevant_edges_and_angles[edgeCounter2][1] = cell_edge_opposing_angle(cellIndices, edge, initial_cell_edges_and_opposite_angle_map)

			relevant_edges_and_angles[edgeCounter2][0] = edge
			relevant_edges_and_angles[edgeCounter2][1] = cell_edge_opposing_angle(cellIndices, edge, cell_edges_and_opposite_angle_map)#

			# save the vertices of this edge as objects
			vertices_of_this_edge = ["vertexObject","vertexObject"]
			vertexCounter = 0
			for vertex1 in vertices(Edge(mesh, relevant_edges_and_angles[edgeCounter2][0])):
				vertices_of_this_edge[vertexCounter] = vertex1
				vertexCounter += 1
			# we need the vertices in the right order, so the vertex we are calculation the force for should be in [1]
			if vertices_of_this_edge[1].index() != vertex.index():
				vertices_of_this_edge[0], vertices_of_this_edge[1] = vertices_of_this_edge[1], vertices_of_this_edge[0]
			#print(vertices_of_this_edge[0].index(), vertices_of_this_edge[0].point().array(), meshClass.coordinates[vertices_of_this_edge[0].index()])
			#print(vertices_of_this_edge[1].index(), vertices_of_this_edge[1].point().array(), meshClass.coordinates[vertices_of_this_edge[1].index()])
			#exit()
			#if n >=34:
			#	print(meshClass.coordinates[vertices_of_this_edge[0].index()] - meshClass.coordinates[vertices_of_this_edge[1].index()])		





			#print("tangens überprüfen. Im Moment ist der Relaxierungspunkt bei pi/2, heißt alle angular springs streben 90° an. Sollte vllt durch den Anfangswinkel ersetzt werden")
			#exit()

			if meshClass.Estar[cellIndices] == 0:
				edgeCounter2 += 1
			else:
				force_on_vertex += meshClass.Estar[cellIndices]*Thickness*(2.0*(1.0/np.tan(relevant_edges_and_angles[edgeCounter2][1]))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndices]) \
								* (cell_edge_length(cellIndices, edge, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellIndices, edge, initial_cell_edges_and_opposite_angle_map)**2) \
								* (meshClass.coordinates[vertices_of_this_edge[0].index()] - meshClass.coordinates[vertices_of_this_edge[1].index()]) \
								+ (meshClass.Estar[cellIndices] * Thickness \
								* (2*(1.0/ np.tan(cell_edge_opposing_angle(cellIndices, relevant_edges[edgeCounter2-1], cell_edges_and_opposite_angle_map)))\
								* (1.0 / np.tan(cell_edge_opposing_angle(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map))) - 1.0 + poisson_ratio)\
								/ (16.0*(1.0 - poisson_ratio**2)* meshClass.cell_volumes[cellIndices]) \
								* ((cell_edge_length(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map)**2) - cell_edge_length(cellIndices, the_opposing_edge, initial_cell_edges_and_opposite_angle_map)**2) \
								+ meshClass.Estar[cellIndices] * Thickness \
								* (2*(1.0/np.tan(relevant_edges_and_angles[edgeCounter2][1]))*(1.0/np.tan(cell_edge_opposing_angle(cellIndices, relevant_edges[edgeCounter2-1], cell_edges_and_opposite_angle_map))) - 1.0 + poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndices]) \
								* (cell_edge_length(cellIndices, edge, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellIndices, edge, initial_cell_edges_and_opposite_angle_map)**2)) \
								* (meshClass.coordinates[vertices_of_this_edge[0].index()] - meshClass.coordinates[vertices_of_this_edge[1].index()]) 





			#force_on_vertex += Thickness*(2.0*(0.1)**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndices]) \
			#force_on_vertex +=0.1*(cell_edge_length(cellIndices, edge, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellIndices, edge, initial_cell_edges_and_opposite_angle_map)**2) \
			#				* (meshClass.coordinates[vertices_of_this_edge[1].index()] - meshClass.coordinates[vertices_of_this_edge[0].index()]) \
							# + (meshClass.Estar[cellIndices] \
							# * Thickness \
							# * (2*(1.0/ tan(cell_edge_opposing_angle(cellIndices, relevant_edges[edgeCounter2-1], cell_edges_and_opposite_angle_map)))\
							# * (1.0/ tan(cell_edge_opposing_angle(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map))) \
							# - 1.0 + poisson_ratio)\
							# / (16.0*(1.0 - poisson_ratio**2)\
							# * Cell(mesh, cellIndices).volume()) \
							# * ((cell_edge_length(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map)**2) \
							# - cell_edge_length(cellIndices, the_opposing_edge, initial_cell_edges_and_opposite_angle_map)**2) \
							# + meshClass.Estar[cellIndices]*Thickness*(2*(1.0/tan(relevant_edges_and_angles[edgeCounter2][1]))*(1.0/tan(cell_edge_opposing_angle(cellIndices, relevant_edges[edgeCounter2-1], cell_edges_and_opposite_angle_map))) - 1.0 + poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndices]) \
							# * (cell_edge_length(cellIndices, edge, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellIndices, edge, initial_cell_edges_and_opposite_angle_map)**2)) \
							# * (meshClass.coordinates[vertices_of_this_edge[0].index()] - meshClass.coordinates[vertices_of_this_edge[1].index()])
							# #* (vertices_of_this_edge[0].point().array() - vertices_of_this_edge[1].point().array())
			# if vertex.index() == 166 :
			# 	#if force_on_vertex[0] / force_on_vertex_array[n-1][vertex.index()][0] > 100:
			# 	print(vertex.index())
			# 	print("")
			# 	print("current foV", force_on_vertex)
			# 	print("meshClass.Estar[cellIndices]", meshClass.Estar[cellIndices])
			# 	print("arctan1", (1.0/tan(relevant_edges_and_angles[edgeCounter2][1])))
			# 	print("edge length difference 1:", (cell_edge_length(cellIndices, edge, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellIndices, edge, initial_cell_edges_and_opposite_angle_map)**2))
			# 	print("edge length vector", (meshClass.coordinates[vertices_of_this_edge[0].index()] - meshClass.coordinates[vertices_of_this_edge[1].index()]))
			# 	print("arctan2", 1.0/tan(cell_edge_opposing_angle(cellIndices, relevant_edges[edgeCounter2-1], cell_edges_and_opposite_angle_map)))
			# 	print("arctan3", 1.0/tan(cell_edge_opposing_angle(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map)))
			# 	print("cell_volume1", Cell(mesh, cellIndices).volume())
			# 	print("edge length difference 2", ((cell_edge_length(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map)* cell_edge_length(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map)) - cell_edge_length(cellIndices, the_opposing_edge, initial_cell_edges_and_opposite_angle_map)**2))
			# 	print("arctan4", (1.0/tan(relevant_edges_and_angles[edgeCounter2][1])))
			# 	print("arctan5", (1.0/tan(cell_edge_opposing_angle(cellIndices, relevant_edges[edgeCounter2-1], cell_edges_and_opposite_angle_map))))
			# 	print("edge length difference 3", cell_edge_length(cellIndices, edge, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellIndices, edge, initial_cell_edges_and_opposite_angle_map)**2)

			# 	print("")

			# 		print("previous foV:", force_on_vertex_array[n-1][vertex.index()])
					#exit()
			# print("initial edge length:",  cell_edge_length(cellIndices, edge, initial_cell_edges_and_opposite_angle_map))
			# print("new edge length:",  Edge(mesh, relevant_edges_and_angles[edgeCounter2][0]).length())
			# print("")
			# print("vector used to multiply:", (vertices_of_this_edge[0].point().array() - vertices_of_this_edge[1].point().array()))
			# print("")
			# print("force on vertex:", force_on_vertex)
			# print("")
				edgeCounter2 += 1



		# print("initial:", initial_relevant_edges_and_angles)
		# print("new:", relevant_edges_and_angles)
	return force_on_vertex
	#exit()

# Same as the TRBS_force_on_vertex but with the "vertex loop" already integrated. Returns an array with [[0,0,0]]*mesh.num_vertices() where each entry is a force vector.
def TRBS_force_on_mesh(meshClass, Thickness = 0.1, poisson_ratio = 0.5):
	mesh = meshClass.boundaryMesh

	vertex_to_edges_map = meshClass.vertex_to_edges_map
	cell_to_vertices_map = meshClass.cell_to_vertices_map

	initial_cell_edges_and_opposite_angle_map = meshClass.initial_cell_edges_and_opposite_angle_map
	cell_edges_and_opposite_angle_map = meshClass.cell_edges_and_opposite_angle_map

	# contruct the array which holds force-vectors
	force_on_vertex = np.asarray([[0,0,0]]*mesh.num_vertices(), float)
	# loop over all cells and calculate the resulting forces on each cells vertices. Theses force-vectors are then simply added.
	for cellObject in cells(mesh):
		#print("cell.Index()", cellIndex)
		# which vertices make up the current cell
		vertices_to_consider = cell_to_vertices_map[cellObject.index()]
		# get the current cells edges
		current_cells_edge_Indices = [0,0,0]
		edgeCounter = 0
		for edgeObject in edges(cellObject):
			current_cells_edge_Indices[edgeCounter] = edgeObject.index()
			edgeCounter += 1

		#l1 -> l3 are not the actual adge lengths but rather the edges which are relevant for my force computation. 
		#to get the lengths one has to use the maps with l1->l3 as input.
		# calculate all the necessary factors
		v1 = vertices_to_consider[0]
		l1 = list(set(current_cells_edge_Indices) - set(vertex_to_edges_map[v1]))

		v2 = vertices_to_consider[1]
		l2 = list(set(current_cells_edge_Indices) - set(vertex_to_edges_map[v2]))

		v3 = vertices_to_consider[2]
		l3 = list(set(current_cells_edge_Indices) - set(vertex_to_edges_map[v3]))

		l1_vector = meshClass.coordinates[v3] - meshClass.coordinates[v2]
		l2_vector = meshClass.coordinates[v3] - meshClass.coordinates[v1]
		l3_vector = meshClass.coordinates[v2] - meshClass.coordinates[v1]
		#print(l1_vector, l2_vector, l3_vector)

		k1 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l1, cell_edges_and_opposite_angle_map)))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])
		k2 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l2, cell_edges_and_opposite_angle_map)))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])
		k3 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l3, cell_edges_and_opposite_angle_map)))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])

		c1 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l2, cell_edges_and_opposite_angle_map)))*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l3, cell_edges_and_opposite_angle_map))) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])
		c2 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l1, cell_edges_and_opposite_angle_map)))*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l3, cell_edges_and_opposite_angle_map))) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])
		c3 = meshClass.Estar[cellObject.index()]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l1, cell_edges_and_opposite_angle_map)))*(1.0/np.tan(cell_edge_opposing_angle(cellObject.index(), l2, cell_edges_and_opposite_angle_map))) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellObject.index()])

		delta_l1 = cell_edge_length(cellObject.index(), l1, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellObject.index(), l1, initial_cell_edges_and_opposite_angle_map)**2
		delta_l2 = cell_edge_length(cellObject.index(), l2, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellObject.index(), l2, initial_cell_edges_and_opposite_angle_map)**2
		delta_l3 = cell_edge_length(cellObject.index(), l3, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellObject.index(), l3, initial_cell_edges_and_opposite_angle_map)**2

		# print("1", v1, l1, k1, c1)
		# print("2", v2, l2, k2, c2)
		# print("3", v3, l3, k3, c3)
		# print(current_cells_edge_Indices)

		force_on_vertex[v1] += k3 * delta_l3 * l3_vector \
							 + k2 * delta_l2 * l2_vector \
							 + c2 * delta_l1 * l3_vector \
							 + c1 * delta_l2 * l3_vector \
							 + c3 * delta_l1 * l2_vector \
							 + c1 * delta_l3 * l2_vector

		force_on_vertex[v2] += k3 * delta_l3 * -l3_vector \
							 + k1 * delta_l1 * l1_vector \
							 + c1 * delta_l2 * -l3_vector \
							 + c2 * delta_l1 * -l3_vector \
							 + c3 * delta_l2 * l1_vector \
							 + c2 * delta_l3 * l1_vector

		force_on_vertex[v3] += k2 * delta_l2 * -l2_vector \
							 + k1 * delta_l1 * -l1_vector \
							 + c1 * delta_l3 * -l2_vector \
							 + c3 * delta_l1 * -l2_vector \
							 + c2 * delta_l3 * -l1_vector \
							 + c3 * delta_l2 * -l1_vector
		#print("force:", force_on_vertex[v1])
		#for vertexIndex in vertices_to_consider:
	return force_on_vertex

	

#calculates the turgor pressure on a vertex, also seen in the supplementary of Goldenbogen et. alls paper 
#"Dynamics of cell wall elasticity pattern shapes the cell during yeast matingmorphogenesis", page 4
#Takes the vertex.index()
def turgor_pressure_on_vertex(meshClass, vertex, pressure):
	turgor_pressure_on_vertex = [0.0,0.0,0.0]
	cellVolume = 0.0
	for cellsIndices in meshClass.vertex_to_cells_map[vertex]:
		cellVolume += meshClass.cell_volumes[cellsIndices]
	#cellVolume = 6.0
	#print(cellVolume)
	turgor_pressure_on_vertex = 1/3 * cellVolume * pressure * meshClass.vertex_normal_map[vertex]
	return turgor_pressure_on_vertex

#calculates the turgor pressure for each vertex on the mesh by iteration through all cells and adding up the resulting pressure for each vertex.
# Returns a numpy array with the index as the vertexIndex and the entry as the force vector
def turgor_pressure_on_mesh(meshClass, pressure):
	mesh = meshClass.boundaryMesh

	pressure_on_vertex = np.zeros((mesh.num_vertices(), 3), dtype=np.float)

	for cellObject in cells(mesh):
		cellVolume = meshClass.cell_volumes[cellObject.index()]

		cell_turgor_pressure = 1/3 * cellVolume * pressure * meshClass.normalVectors[cellObject.index()]*meshClass.orientation[cellObject.index()]

		for vertexObject in vertices(cellObject):
			pressure_on_vertex[vertexObject.index()] += cell_turgor_pressure
	return pressure_on_vertex


def plotCellIndicesList(meshClass, shaftList):
	shaftListVertices = []
	for cellIndices in shaftList:
		for vertices1 in vertices(Cell(meshClass.boundaryMesh, cellIndices)):
			shaftListVertices.append(vertices1.index())
	shaftListVertices = list(set(shaftListVertices))
	#print(shaftListVertices, len(shaftListVertices), meshClass.boundaryMesh.num_vertices())
	#testList = meshClass.classArrayHoldingSlices[9]


	plottingList_x = []
	plottingList_y = []
	plottingList_z = []
	for vertexIndex in shaftListVertices:
		plottingList_x.append(meshClass.coordinates[vertexIndex][0])

		plottingList_y.append(meshClass.coordinates[vertexIndex][1])

		plottingList_z.append(meshClass.coordinates[vertexIndex][2])

	from mpl_toolkits.mplot3d import Axes3D
	mesh = meshClass.boundaryMesh

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	wholeCell_x = mesh.coordinates().T[0]
	wholeCell_y = mesh.coordinates().T[1]
	wholeCell_z = mesh.coordinates().T[2]
	#print('wholeCell_x:', wholeCell_x)
	ax.scatter(wholeCell_x, wholeCell_y, wholeCell_z, zdir='z', s=5, c='Black', depthshade=True )
	ax.scatter(plottingList_x, plottingList_y, plottingList_z, zdir='z', s=20, c='Green', depthshade=True)
	plt.show()


#save all the relevant data necessary to run the pressuredGrowth etc. Saves n in the filename aswell
def saveData(meshClass, n, res=20, customEnding = ""):
	for attr, value in meshClass.__dict__.items():
		try:
			with open('pickled/{0}_{1}_res{2}_{3}{4}.pkl'.format(meshClass.name, attr, str(res), str(n), customEnding), 'wb') as output:
				pickle.dump(value, output, pickle.HIGHEST_PROTOCOL)
		except TypeError:
			pass
	# meshClass.cell_edges_and_opposite_angle_map.tofile('{0}.cell_edges_and_opposite_angle_map_res{1}_{2}.dat'.format(meshClass.name, str(res), str(n))) 
	# meshClass.initial_cell_edges_and_opposite_angle_map.tofile('{0}.initial_cell_edges_and_opposite_angle_map_res{1}_{2}.dat'.format(meshClass.name, str(res), str(n)))
	# classMsphere1_startVolume_array = np.array([meshClass.startVolume])
	# np.savetxt('{0}.startVolume_res{1}_{2}.txt'.format(meshClass.name, str(res), str(n)), classMsphere1_startVolume_array)
	# classMsphere1_u_max_start_array = np.array([meshClass.u_max_start])
	# np.savetxt('{0}.u_max_start_res{1}_{2}.txt'.format(meshClass.name, str(res), str(n)), classMsphere1_u_max_start_array)
	meshClass.fileName = '{0}_boundaryMesh_res{1}_{2}{3}.xml'.format(meshClass.name, str(res), str(n), customEnding)
	File(meshClass.fileName) << meshClass.boundaryMesh

#load data. Its also possible to exclude a set() of attributes
def loadData(meshClass, n, res=20, customEnding = "", exclude = set(), queingSystemPrefix = ""):
	for attr, value in vars(meshClass).items():
			if attr not in {'currentSolutionFunction','functionSpace','trialFunction','testFunction','twoDStimulus','stimulus','PDE', 'coordinates', 'cell_to_edges_map', 'start_hmax'} | exclude:
				try:
					with open(queingSystemPrefix + 'pickled/{0}_{1}_res{2}_{3}{4}.pkl'.format(meshClass.name, attr, str(res), str(n), customEnding), 'rb') as input:
						value = pickle.load(input)
						setattr(meshClass, attr, value)
						if attr == "stimulusCellIndex":
							print("setting stimulusCell")
							meshClass.stimulusCell = Cell(meshClass.boundaryMesh, meshClass.stimulusCellIndex)
						
				except EOFError:
					print(attr, " is a fenics object and was not loaded")
				except FileNotFoundError:
					print(attr, 'was not found, skipping it')



# same as the TRBS_force_on_mesh but compatible with the joblib delayed architecture. Since joblib uses pickle and dolfin-objects cannot be pickled, a version without objects was 
# necessary. The cells loop is delayed and then split into n jobs.
def TRBS_force_on_mesh_parallel(number_of_jobs, current_job_number, relax_triangles = True, Refinement_has_happened = False, Thickness = 0.1, poisson_ratio = 0.5):

	force_on_vertex = np.zeros((number_of_vertices_local, 3), dtype=np.float)
	sigma_VM_local = np.zeros(((number_of_cells_local), 1), dtype=np.float)
	# joblib: decerning which cells each job should compute
	range_of_each_job = number_of_cells_local//number_of_jobs
	if current_job_number != number_of_jobs-1:
		myRange = range(range_of_each_job * current_job_number, range_of_each_job * (current_job_number + 1), 1)
	else:
		myRange = range(range_of_each_job * current_job_number, number_of_cells_local, 1)
	for cellIndex in myRange:
		#print(myRange)
		#print(cellIndex)
		#force_on_vertex = [0,0,0]*number_of_vertices

		#print("cell.Index()", ,)
		vertices_to_consider = cell_to_vertices_map_local[cellIndex]
		
		current_cells_edge_Indices = np.zeros(3) #[0,0,0]

		current_cells_edge_Indices = cell_to_edges_map_local[cellIndex]


		#l1 -> l3 are not the actual edge lengths but rather the edges which are relevant for my force computation. 
		#to get the lengths one has to use the maps with l1->l3 as input
		v1 = vertices_to_consider[0]
		l1 = list(set(current_cells_edge_Indices) - set(vertex_to_edges_map_local[v1]))

		v2 = vertices_to_consider[1]
		l2 = list(set(current_cells_edge_Indices) - set(vertex_to_edges_map_local[v2]))

		v3 = vertices_to_consider[2]
		l3 = list(set(current_cells_edge_Indices) - set(vertex_to_edges_map_local[v3]))

		l1_vector = coordinates_local[v3] - coordinates_local[v2]
		l2_vector = coordinates_local[v3] - coordinates_local[v1]
		l3_vector = coordinates_local[v2] - coordinates_local[v1]

		a1 = cell_edge_opposing_angle(cellIndex, l1, cell_edges_and_opposite_angle_map_local)
		a2 = cell_edge_opposing_angle(cellIndex, l2, cell_edges_and_opposite_angle_map_local)
		a3 = cell_edge_opposing_angle(cellIndex, l3, cell_edges_and_opposite_angle_map_local)

		a1_factor = 1.0
		a2_factor = 1.0
		a3_factor = 1.0

		# # if a angle gets bigger than 150, i deemed it wrong for it to increase even more in the same manner.
		# # With this here function the resulting force on this vertex is either weakend or even inverted so that it should not go beyond 180 degrees.
		# # This whole process can be avoided by choosing very small force steps, as close to continiously as possible. But this is of course very slow
		# if a1 > math.radians(150):
		# 	a1_factor = -0.5 #5 - math.degrees(a1) * 1/30.0
		# 	#print("a1_factor at cell:", a1_factor)
		# if a2 > math.radians(150):
		# 	a2_factor = -0.5 #5 - math.degrees(a2) * 1/30.0
		# 	#print("a2_factor at cell:", a2_factor)
		# if a3 > math.radians(150):
		# 	a3_factor = -0.5 #5 - math.degrees(a3) * 1/30.0
		# 	#print("a3_factor at cell:", a3_factor)

		# if (a1 > math.radians(150)) or (a2 > math.radians(150)) or (a3 > math.radians(150)):
		# 	listOfCellsAnglesToCheck.append(cellIndex)
		#print(l1_vector, l2_vector, l3_vector)

		k1 = Estar_local[cellIndex]*Thickness*(2.0*(1.0/np.tan(a1))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*cell_volumes_local[cellIndex])
		k2 = Estar_local[cellIndex]*Thickness*(2.0*(1.0/np.tan(a2))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*cell_volumes_local[cellIndex])
		k3 = Estar_local[cellIndex]*Thickness*(2.0*(1.0/np.tan(a3))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*cell_volumes_local[cellIndex])

		c1 = Estar_local[cellIndex]*Thickness*(2.0*(1.0/np.tan(a2))*(1.0/np.tan(a3)) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*cell_volumes_local[cellIndex])
		c2 = Estar_local[cellIndex]*Thickness*(2.0*(1.0/np.tan(a1))*(1.0/np.tan(a3)) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*cell_volumes_local[cellIndex])
		c3 = Estar_local[cellIndex]*Thickness*(2.0*(1.0/np.tan(a1))*(1.0/np.tan(a2)) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*cell_volumes_local[cellIndex])





			

		# if relax_triangles == False and Refinement_has_happened == True and local_parent_cells[cellIndex] != cellIndex:
		# 	#print("this case has happened")
		# 	# l1_old = cell_edge_length(local_parent_cells[cellIndex], l1, previous_cell_edges_and_opposite_angle_map_local)
		# 	# l1_new = cell_edge_length(cellIndex, l1, cell_edges_and_opposite_angle_map_local)
		# 	# L1_old = cell_edge_length(local_parent_cells[cellIndex], l1, previous_initial_cell_edges_and_opposite_angle_map_local)

		# 	# L1_new = np.sqrt(l1_new**2 + L1_old**2 - l1_old**2)


		# 	# l2_old = cell_edge_length(local_parent_cells[cellIndex], l2, previous_cell_edges_and_opposite_angle_map_local)
		# 	# l2_new = cell_edge_length(cellIndex, l2, cell_edges_and_opposite_angle_map_local)
		# 	# L2_old = cell_edge_length(local_parent_cells[cellIndex], l2, previous_initial_cell_edges_and_opposite_angle_map_local)

		# 	# L2_new = np.sqrt(l2_new**2 + L2_old**2- l2_old**2)


		# 	# l3_old = cell_edge_length(local_parent_cells[cellIndex], l3, previous_cell_edges_and_opposite_angle_map_local)
		# 	# l3_new = cell_edge_length(cellIndex, l3, cell_edges_and_opposite_angle_map_local)
		# 	# L3_old = cell_edge_length(local_parent_cells[cellIndex], l3, previous_initial_cell_edges_and_opposite_angle_map_local)

		# 	# L3_new = np.sqrt(l3_new**2 + L3_old**2 - l3_old**2)


		# 	l1_old = previous_cell_edges_and_opposite_angle_map_local[local_parent_cells[cellIndex]][0][1]
		# 	l1_new = cell_edges_and_opposite_angle_map_local[cellIndex][0][1]
		# 	L1_old = previous_initial_cell_edges_and_opposite_angle_map_local[local_parent_cells[cellIndex]][0][1]

		# 	L1_new = np.sqrt(l1_new**2 - (l1_old**2 - L1_old**2))
		# 	delta_l1 = l1_old**2 - L1_old**2

		# 	l2_old = previous_cell_edges_and_opposite_angle_map_local[local_parent_cells[cellIndex]][1][1]
		# 	l2_new = cell_edges_and_opposite_angle_map_local[cellIndex][1][1]
		# 	L2_old = previous_initial_cell_edges_and_opposite_angle_map_local[local_parent_cells[cellIndex]][1][1]

		# 	L2_new = np.sqrt(l2_new**2 + L2_old**2- l2_old**2)
		# 	delta_l2 = l2_old**2 - L2_old**2
		# 	#if l2_new >= L2_new:
		# 	#	print("bad case 2")


		# 	l3_old = previous_cell_edges_and_opposite_angle_map_local[local_parent_cells[cellIndex]][2][1]
		# 	l3_new = cell_edges_and_opposite_angle_map_local[cellIndex][2][1]
		# 	L3_old = previous_initial_cell_edges_and_opposite_angle_map_local[local_parent_cells[cellIndex]][2][1]

		# 	L3_new = np.sqrt(l3_new**2 + L3_old**2 - l3_old**2)
		# 	delta_l3 = l3_old**2 - L3_old**2
		# 	#if l3_new >= L3_new:
		# 	#	print("bad case 3")


		# 	#delta_l1 = l1_new**2 - L1_new**2
		# 	#delta_l2 = l2_new**2 - L2_new**2
		# 	#delta_l3 = l3_new**2 - L3_new**2
		# else:
		delta_l1 = cell_edge_length(cellIndex, l1, cell_edges_and_opposite_angle_map_local)**2 - cell_edge_length(cellIndex, l1, initial_cell_edges_and_opposite_angle_map_local)**2
		delta_l2 = cell_edge_length(cellIndex, l2, cell_edges_and_opposite_angle_map_local)**2 - cell_edge_length(cellIndex, l2, initial_cell_edges_and_opposite_angle_map_local)**2
		delta_l3 = cell_edge_length(cellIndex, l3, cell_edges_and_opposite_angle_map_local)**2 - cell_edge_length(cellIndex, l3, initial_cell_edges_and_opposite_angle_map_local)**2

		# print("1", v1, l1, k1, c1)
		# print("2", v2, l2, k2, c2)
		# print("3", v3, l3, k3, c3)
		# print(current_cells_edge_Indices)

		force_on_vertex[v1] += (k3 * delta_l3 * l3_vector \
							 + k2 * delta_l2 * l2_vector \
							 + c2 * delta_l1 * l3_vector \
							 + c1 * delta_l2 * l3_vector \
							 + c3 * delta_l1 * l2_vector \
							 + c1 * delta_l3 * l2_vector) * a1_factor

		force_on_vertex[v2] += (k3 * delta_l3 * -l3_vector \
							 + k1 * delta_l1 * l1_vector \
							 + c1 * delta_l2 * -l3_vector \
							 + c2 * delta_l1 * -l3_vector \
							 + c3 * delta_l2 * l1_vector \
							 + c2 * delta_l3 * l1_vector) * a2_factor

		force_on_vertex[v3] += (k2 * delta_l2 * -l2_vector \
							 + k1 * delta_l1 * -l1_vector \
							 + c1 * delta_l3 * -l2_vector \
							 + c3 * delta_l1 * -l2_vector \
							 + c2 * delta_l3 * -l1_vector \
							 + c3 * delta_l2 * -l1_vector) * a3_factor
		#print("force:", force_on_vertex[v1])


		myLambda = Estar_local[cellIndex] * poisson_ratio/(1-poisson_ratio**2)
		myMu = Estar_local[cellIndex]/(1+poisson_ratio)
		#local_trace_of_epsilon = 1/cell_volumes_local[cellIndex] * ( (cell_edges_and_opposite_angle_map_local[cellIndex][0][0]**2 - initial_cell_edges_and_opposite_angle_map_local[cellIndex][0][0]**2) * np.cos(cell_edges_and_opposite_angle_map_local[cellIndex][0][1])
		#					 + (cell_edges_and_opposite_angle_map_local[cellIndex][1][0]**2 - initial_cell_edges_and_opposite_angle_map_local[cellIndex][1][0]**2) * np.cos(cell_edges_and_opposite_angle_map_local[cellIndex][1][1])
		#					 + (cell_edges_and_opposite_angle_map_local[cellIndex][2][0]**2 - initial_cell_edges_and_opposite_angle_map_local[cellIndex][2][0]**2) * np.cos(cell_edges_and_opposite_angle_map_local[cellIndex][2][1]))
		local_trace_of_epsilon = 1/cell_volumes_local[cellIndex] * (delta_l1 * np.cos(a1)**2 + delta_l2*np.cos(a2)**2 + delta_l3*np.cos(a3)**2)


		#local_trace_of_C = 1/cell_volumes_local[cellIndex] * ( cell_edges_and_opposite_angle_map_local[cellIndex][0][0]**2 * np.cos(cell_edges_and_opposite_angle_map_local[cellIndex][0][1])  
		#					 + cell_edges_and_opposite_angle_map_local[cellIndex][1][0]**2 * np.cos(cell_edges_and_opposite_angle_map_local[cellIndex][1][1])
		#					 + cell_edges_and_opposite_angle_map_local[cellIndex][2][0]**2 * np.cos(cell_edges_and_opposite_angle_map_local[cellIndex][2][1]))
		local_trace_of_C = 1/cell_volumes_local[cellIndex] * (cell_edge_length(cellIndex, l1, cell_edges_and_opposite_angle_map_local)**2*np.cos(a1) + cell_edge_length(cellIndex, l2, cell_edges_and_opposite_angle_map_local)**2*np.cos(a2) + cell_edge_length(cellIndex, l3, cell_edges_and_opposite_angle_map_local)**2*np.cos(a3))


		det_of_C = cell_volumes_local[cellIndex]/initial_cell_volumes_local[cellIndex]

		trace_of_S = 2*myLambda*local_trace_of_epsilon + 2*myMu*local_trace_of_epsilon
		# print("myLambda", myLambda)
		# print("myMu", myMu)
		# print("local_trace_of_epsilon", local_trace_of_epsilon)
		# print("det_of_C", det_of_C)
		# print("local_trace_of_C", local_trace_of_C)
		det_of_S = (myLambda**2 + 2*myLambda*myMu)*local_trace_of_epsilon**2 + myMu*(det_of_C - local_trace_of_C -1)
		#print("det_of_S", det_of_S)

		sigma_VM_local[cellIndex] = np.sqrt(trace_of_S**2 - 3*det_of_S)

	return (force_on_vertex, sigma_VM_local)



def TRBS_alpha_on_mesh(meshClass, sigma_VM_on_mesh, sigma_Y, activeSurfaceSource = False, R=0.5):

	local_alpha_on_mesh = np.zeros((meshClass.boundaryMesh.num_cells(), 1), dtype=np.float)

	for cells1 in cells(meshClass.boundaryMesh):
		if sigma_VM_on_mesh[cells1.index()] > sigma_Y:
			try:
				local_midpoint = cells1.midpoint()
			except RuntimeError:
				print("Building bounding_box_tree")
				myMesh.boundaryMesh.bounding_box_tree().build(myMesh.boundaryMesh)
				local_midpoint = cells1.midpoint()
			if activeSurfaceSource == False:
				local_phi = meshClass.Estar[cells1.index()] * poisson_ratio1/(1-poisson_ratio1**2) * np.exp(LA.norm(local_midpoint.array()- meshClass.stimulusCell.midpoint().array())**2/2*R)
			else:
				local_phi = meshClass.Estar[cells1.index()] * poisson_ratio1/(1-poisson_ratio1**2) * np.exp(LA.norm(local_midpoint.array()- meshClass.stimulusCell.midpoint().array())**2/2*R)
			local_alpha = local_phi*(sigma_VM_on_mesh[cells1.index()] - sigma_Y)
			local_alpha_on_mesh[cells1.index()] = local_alpha
		else:
			local_alpha_on_mesh[cells1.index()] = 0
	return local_alpha_on_mesh

# void function. Marks cells in meshClass based on their current Cdc42 values. The range corresponds to the percentage of max Cdc42 which is still included. With 0.25,
# the top 25% of cells are marked.
def TRBSmarkForRefinement(meshClass, solutionFunctionRange = 0.25, plotMarked = True, TIP = False):
	# get the max and min current solution function value and the min cell volume. Min cell volume deprecated.
	maximumCurrentSolutionFunctionValue = 0
	try:
		minimumCurrentSolutionFunctionValue = meshClass.currentSolutionFunction(Cell(meshClass.boundaryMesh, 0).midpoint())
	except RuntimeError:
		print("Building bounding_box_tree")
		meshClass.boundaryMesh.bounding_box_tree().build(meshClass.boundaryMesh)
		minimumCurrentSolutionFunctionValue = meshClass.currentSolutionFunction(Cell(meshClass.boundaryMesh, 0).midpoint())
	maxValue_corresponding_cell = None
	meshClass.minimumCellVolume = Cell(meshClass.boundaryMesh, 0).volume()
	for cellsObject in cells(meshClass.boundaryMesh):
		try:
			tempMidpoint = meshClass.currentSolutionFunction(cellsObject.midpoint())
		except RuntimeError:
			print("Building bounding_box_tree")
			meshClass.boundaryMesh.bounding_box_tree().build(meshClass.boundaryMesh)
			tempMidpoint = meshClass.currentSolutionFunction(cellsObject.midpoint())

		if tempMidpoint> maximumCurrentSolutionFunctionValue:
			maximumCurrentSolutionFunctionValue = tempMidpoint
			maxValue_corresponding_cell = cellsObject
		if tempMidpoint < minimumCurrentSolutionFunctionValue:
			minimumCurrentSolutionFunctionValue = tempMidpoint
		if cellsObject.volume() < meshClass.minimumCellVolume:
			meshClass.minimumCellVolume = cellsObject.volume()

	# get a Cdc42 concentration range
	currentSolutionFunctionRange = maximumCurrentSolutionFunctionValue - minimumCurrentSolutionFunctionValue
	# the solutionFunctionRange defines to what u-percentage cells are chosen for refinement. 0.4 = 40%
	meshClass.solutionFunctionRange = solutionFunctionRange
	myRange = (maximumCurrentSolutionFunctionValue - minimumCurrentSolutionFunctionValue)*(meshClass.solutionFunctionRange)
	myRangeMax = maximumCurrentSolutionFunctionValue
	myRangeMin = maximumCurrentSolutionFunctionValue - myRange

	#do the chosing, mark cells for refinement or other tasks
	shaftList = []
	for cells11 in cells(meshClass.boundaryMesh):
		try:
			currentSolutionFunctionValue = meshClass.currentSolutionFunction(cells11.midpoint())
		except RuntimeError:
			print("Building bounding_box_tree")
			meshClass.boundaryMesh.bounding_box_tree().build(meshClass.boundaryMesh)
			currentSolutionFunctionValue = meshClass.currentSolutionFunction(cells11.midpoint())
		if currentSolutionFunctionValue >= myRangeMin:
			if TIP == False:
				meshClass.cell_markers_boundary[cells11.index()] = True
				shaftList.append(cells11.index())
			else:
				meshClass.tip_markers[cells11.index()] = True
				shaftList.append(cells11.index())
	# if not on server, plot the chosen cells
	if calculateOnServer == False and plotMarked == True:
		print("range:",meshClass.solutionFunctionRange)
		print("number of cells marked:", len(shaftList))
		plotCellIndicesList(meshClass, shaftList)












# def TRBS_k_edge(meshClass, Thickness = 0.1, poisson_ratio = 0.5):
# 		# create a num_edges() x num_edges() matrix for storing the equation system
# 		k_on_edges = np.zeros((meshClass.boundaryMesh.num_edges(), 1), dtype=np.float)

# 		for vertexIndex in range(len(meshClass.coordinates)):
# 		#	for edgeIndex in meshClass.vertex_to_edges_map[vertexIndex]:
# 			for cellIndex in meshClass.vertex_to_cells_map[vertexIndex]:
# 				relevant_edges = []
# 				for i in range(3):
# 					if meshClass.cell_to_edges_map[cellIndex][i] in meshClass.vertex_to_edges_map[vertexIndex]:
# 						relevant_edges.append(meshClass.cell_to_edges_map[cellIndex][i])
# 				#print(relevant_edges)
# 				#print(meshClass.cell_to_edges_map[cellIndex])
# 				#print(meshClass.vertex_to_edges_map[vertexIndex])
# 				l1 = relevant_edges[0]
# 				l2 = relevant_edges[1]
# 				#print(l1, l2)


# 				v10 = Edge(meshClass.boundaryMesh, l1).entities(0)[0]
# 				v11 = Edge(meshClass.boundaryMesh, l1).entities(0)[1]

# 				v20 = Edge(meshClass.boundaryMesh, l2).entities(0)[0]
# 				v21 = Edge(meshClass.boundaryMesh, l2).entities(0)[1]
# 				#print(v10,v11,l1)
# 				#print(v20, v21, l2)

# 				l1_vector = (meshClass.coordinates[v10] - meshClass.coordinates[v11])
# 				l2_vector = (meshClass.coordinates[v20] - meshClass.coordinates[v21])
# 				#cell_edges_and_opposite_angle_map = meshClass.cell_edges_and_opposite_angle_map
# 				#print(cell_edges_and_opposite_angle_map)
				

# 				#k1 = meshClass.Estar[cellIndex]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellIndex, l1, cell_edges_and_opposite_angle_map)))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndex])
# 				#k2 = meshClass.Estar[cellIndex]*Thickness*(2.0*(1.0/np.tan(cell_edge_opposing_angle(cellIndex, l2, cell_edges_and_opposite_angle_map)))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndex])
				
# 				a1 = cell_edge_opposing_angle(cellIndex, l1, meshClass.cell_edges_and_opposite_angle_map)
# 				a2 = cell_edge_opposing_angle(cellIndex, l2, meshClass.cell_edges_and_opposite_angle_map)

# 				k1 = meshClass.Estar[cellIndex]*Thickness*(2.0*(1.0/np.tan(a1))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndex])
# 				k2 = meshClass.Estar[cellIndex]*Thickness*(2.0*(1.0/np.tan(a2))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndex])


# 				#l1_dash = k1 #* l1_vector
# 				#l2_dash = k2 #* l2_vector

# 				#delta_l1 = cell_edge_length(cellIndex, l1, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellIndex, l1, initial_cell_edges_and_opposite_angle_map)**2
# 				#delta_l2 = cell_edge_length(cellIndex, l2, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellIndex, l2, initial_cell_edges_and_opposite_angle_map)**2
# 				k_on_edges[l1] = k1
# 				k_on_edges[l2] = k2


# 		#print(k_on_edges)
# 		return k_on_edges



# calculates the k and c values according to "Delingette at al." for each cell.
# Returns a list with the first index as the cellIndex and the entry as a dictionary containing the edgeIndex as "key" and the k or c value as "value"
def k_and_c_on_cells(meshClass, Thickness = 0.1, poisson_ratio = 0.5):

	# both values are dependent on which cell is being observed right now since both k and c depends on the cells angles
	# so there is a set of k's and c's for every cell. k's and c's are then stored in dictionaries
	k_on_cells_edges = [0] * meshClass.boundaryMesh.num_cells()
	c_on_cells_angles_opposing_edges = [0] * meshClass.boundaryMesh.num_cells()
	cells_edge_vectors = [[0,0,0]] * meshClass.boundaryMesh.num_cells()

	for cellIndex in range(meshClass.boundaryMesh.num_cells()):
		#print(myRange)
		#print(cellIndex)
		#force_on_vertex = [0,0,0]*number_of_vertices

		#get the cells verticies and edges
		vertices_to_consider = meshClass.cell_to_vertices_map[cellIndex]
		
		current_cells_edge_Indices = np.zeros(3) #[0,0,0]

		current_cells_edge_Indices = meshClass.cell_to_edges_map[cellIndex]


		#l1 -> l3 are not the actual edge lengths but rather the edges which are relevant for my force computation. 
		#to get the lengths one has to use the maps with l1->l3 as input
		v1 = vertices_to_consider[0]
		l1 = list(set(current_cells_edge_Indices) - set(meshClass.vertex_to_edges_map[v1]))

		v2 = vertices_to_consider[1]
		l2 = list(set(current_cells_edge_Indices) - set(meshClass.vertex_to_edges_map[v2]))

		v3 = vertices_to_consider[2]
		l3 = list(set(current_cells_edge_Indices) - set(meshClass.vertex_to_edges_map[v3]))

		l1_vector = meshClass.coordinates[v3] - meshClass.coordinates[v2]
		l2_vector = meshClass.coordinates[v3] - meshClass.coordinates[v1]
		l3_vector = meshClass.coordinates[v2] - meshClass.coordinates[v1]

		cells_edge_vectors[cellIndex][0] = l1_vector
		cells_edge_vectors[cellIndex][1] = l2_vector
		cells_edge_vectors[cellIndex][2] = l3_vector


		a1 = cell_edge_opposing_angle(cellIndex, l1, meshClass.cell_edges_and_opposite_angle_map)
		a2 = cell_edge_opposing_angle(cellIndex, l2, meshClass.cell_edges_and_opposite_angle_map)
		a3 = cell_edge_opposing_angle(cellIndex, l3, meshClass.cell_edges_and_opposite_angle_map)


		k1 = meshClass.Estar[cellIndex]*Thickness*(2.0*(1.0/np.tan(a1))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndex])
		k2 = meshClass.Estar[cellIndex]*Thickness*(2.0*(1.0/np.tan(a2))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndex])
		k3 = meshClass.Estar[cellIndex]*Thickness*(2.0*(1.0/np.tan(a3))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndex])
		
		local_k_dict = {}
		local_k_dict[l1[0]] = k1
		local_k_dict[l2[0]] = k2
		local_k_dict[l3[0]] = k3

		k_on_cells_edges[cellIndex] = local_k_dict
		# print(k_on_cells_edges[cellIndex])


		c1 = meshClass.Estar[cellIndex]*Thickness*(2.0*(1.0/np.tan(a2))*(1.0/np.tan(a3)) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndex])
		c2 = meshClass.Estar[cellIndex]*Thickness*(2.0*(1.0/np.tan(a1))*(1.0/np.tan(a3)) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndex])
		c3 = meshClass.Estar[cellIndex]*Thickness*(2.0*(1.0/np.tan(a1))*(1.0/np.tan(a2)) + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*meshClass.cell_volumes[cellIndex])

		local_c_dict = {}
		local_c_dict[l1[0]] = c1
		local_c_dict[l2[0]] = c2
		local_c_dict[l3[0]] = c3

		c_on_cells_angles_opposing_edges[cellIndex] = local_c_dict
		# print(c_on_cells_angles_opposing_edges[cellIndex])


	return k_on_cells_edges, c_on_cells_angles_opposing_edges, cells_edge_vectors



# Void function. After refinement, the fitting initial edges have to be calculated for each new triangle. Since the elastic energy
# of each triangle is conserved, a parents elastic energy has to be shared between every child. sum(E_child) = E_parent
def TRBS_adjust_initial_edges_with_elastic_energy(meshClass, elastic_energy_on_cells_before_refinement):
	
	meshClass.edges_to_cells_map = edges_to_cells_map(meshClass)
	# get the parent cells data array
	# index is the current(refined) cell index, value is the parent cell of that index
	parent_cells = meshClass.boundaryMesh.data().array("parent_cell", meshClass.boundaryMesh.topology().dim())
	#print(parent_cells)

	dict_with_parent_cells_daughters = {}

	for parent in parent_cells:
		dict_with_parent_cells_daughters[parent] = []

	for i,parent in enumerate(parent_cells):
		dict_with_parent_cells_daughters[parent_cells[i]].append(parent)
	#print(dict_with_parent_cells_daughters)

	for edgeIndex in range(meshClass.boundaryMesh.num_edges()):
		# define the vertices according to Delingette and my own TRBS_force_on_mesh_parallel function
		v2 = Edge(meshClass.boundaryMesh, edgeIndex).entities(0)[0]
		v3 = Edge(meshClass.boundaryMesh, edgeIndex).entities(0)[1]

		for cellIndex in meshClass.edges_to_cells_map[edgeIndex]:
			# change it only for the newly created cells
			if cellIndex in meshClass.new_cells:
				
				l1 = Edge(meshClass.boundaryMesh, edgeIndex).length()

				# 0.25 is a simplification. Most of the refined triangles were turned into 4 new triangles, therefore the
				# 1/4 factor. This works well, but is of course a simplification.
				if elastic_energy_on_cells_before_refinement[parent_cells[cellIndex]] > 0:
					factor_1 = len(dict_with_parent_cells_daughters[parent_cells[cellIndex]])
					if factor_1 == 0:
						factor_1 = 1
					new_L1 = np.sqrt( \
							l1**2
							- factor_1*np.sqrt(1/3 * elastic_energy_on_cells_before_refinement[parent_cells[cellIndex]] \
							* 4/meshClass.k_on_cells[cellIndex][edgeIndex]) \
							)
				else:
					new_L1 = np.sqrt( \
							l1**2
							+ factor_1*np.sqrt(-1/3 * elastic_energy_on_cells_before_refinement[parent_cells[cellIndex]] \
							* 4/meshClass.k_on_cells[cellIndex][edgeIndex]) \
							)
				# set the new initial length new_L1
				for i in range(3):
					if meshClass.initial_cell_edges_and_opposite_angle_map[cellIndex][i][0] == edgeIndex:
						#print("previous:", meshClass.initial_cell_edges_and_opposite_angle_map[cellIndex][i][1])
						meshClass.initial_cell_edges_and_opposite_angle_map[cellIndex][i][1] = new_L1
						#print("new:", meshClass.initial_cell_edges_and_opposite_angle_map[cellIndex][i][1])

