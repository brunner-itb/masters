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
def nextSourceCell(mesh, cell, gradient2, noiseFactor, hmin, lengthFactor=1):
	global calculateGradientSphere
	dim = 3 #len(gradient2(cell.midpoint()))
	#averageEdgeLengths = np.average(EdgeLengths(mesh))
	maxEdgeLengths = hmin*lengthFactor #np.max(EdgeLengths(mesh))
	#print 'maxEdgeLengths', maxEdgeLengths	#get the gradient2 Vector and scale it with the average edge lengths
	gradientVector = np.zeros(dim)
	
	gradientNorm = LA.norm(gradient2(cell.midpoint()))
	for i in range(dim):
		gradientVector[i] = cell.midpoint()[i] + (1/gradientNorm) * gradient2(cell.midpoint())[i] * maxEdgeLengths
	#* (maxEdgeLengths/LA.norm(gradientVectorUnscaled))
	#get some random noise in there, using the normal distribution with sigma = 0.1
	if gradientNorm == 0:
		print( '################################')
		print( '################################')
		print( '      NaN, gradientNorm = 0     ')
		print( '################################')
		print( '################################')
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
def cellNeighbors(mesh):
	# Init facet-cell connectivity
	tdim = mesh.topology().dim()
	mesh.init(tdim - 1, tdim)
	# For every cell, build a list of cells that are connected to its facets
	# but are not the iterated cell
	return {cell.index(): sum((filter(lambda ci: ci != cell.index(),
                                            facet.entities(tdim))
                                    for facet in facets(cell)), [])
                for cell in cells(mesh)}

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
def initialCellOrientation(meshClass, amountOfSlices, straightLengthFactor):
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
# slices the mesh.vertices in different layers based on their x-value. Used for collision detection to reduce computing power
# returns a list of lists. list[:] are the slices, list[:][:] are the vertices in said slices
def meshSlicing(meshClass, amountOfSlices):
	meshcoordTransposed = meshClass.coordinates.T
	xValueMin = np.min(meshClass.coordinates.T[0])
	xValueMax = np.max(meshClass.coordinates.T[0])
	sliceSize = (xValueMax - xValueMin)/amountOfSlices
	arrayHoldingSlices = [[] for i in range(amountOfSlices)]

	#print(xValueMin, xValueMax, sliceSize, arrayHoldingSlices)

	for index, vertices in enumerate(meshClass.coordinates):
		#get the x-value and correct it with xValueMin. This way all vertices ly inbetween 0 and (xValueMax - xValueMin).
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
	return arrayHoldingSlices




# not deprecated function
def meshClassOrientation(meshClass, amountOfSlices, straightLengthFactor):
	#triangles = meshClass.triangles
	tempCellOrientation = initialCellOrientation(meshClass, amountOfSlices, straightLengthFactor)
	meshOrientationArray = [0]*meshClass.boundaryMesh.num_cells()
	for i in range(meshClass.boundaryMesh.num_cells()):
		meshOrientationArray[i] = tempCellOrientation[i]
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

#create a vertex to cells map, returns an array. The first array index is the vertex index, the corresponding array are the cells.index(). 
#so vertex_to_cells_map[10] returns all cells.index() that contain vertex.index() 10
def vertex_to_cells_map(meshClass):
	mesh = meshClass.boundaryMesh
	vertex_to_cells_map_untrimmed = np.zeros([mesh.num_vertices(),20], dtype=int)
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
		vertex_normal_map[vertex_index] *= 1.0/LA.norm(vertex_normal_map[vertex_index]) #norms said normal vector
	return vertex_normal_map

#calculates all vertex.index() that belong to a cell. Returns a map where the first index is the cell.index() and the corresponding entries are the vertex.index()'s'
def cell_to_vertices_map(meshClass):
	mesh = meshClass.boundaryMesh
	tempArray = np.empty([mesh.num_cells(),3])
	for cells2 in cells(mesh):
		for j in range(3):
			tempArray[cells2.index()][j] = cells2.entities(0)[j]
	return tempArray


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


	meshClass.functionSpace = FunctionSpace(meshClass.boundaryMesh, 'CG', 1)
	meshClass.trialFunction = interpolate(meshClass.trialFunction, meshClass.functionSpace)
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
	#save every cells orientation as an array
	meshClass.orientation = meshClassOrientation(meshClass, amountOfSlices, straightLengthFactor)

	meshClass.normalVectors = cellNormals(meshClass.boundaryMesh)
	meshClass.vertex_to_cells_map = vertex_to_cells_map(meshClass)
	meshClass.vertex_normal_map = vertex_normal_map(meshClass, meshClass.vertex_to_cells_map, meshClass.normalVectors)
	meshClass.cell_to_vertices_map = cell_to_vertices_map(meshClass)
	meshClass.cell_markers_boundary = MeshFunction('bool', meshClass.boundaryMesh, meshClass.boundaryMesh.topology().dim(), False)

	return listOfPDE

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



#calculates all vertex.index() that belong to a cell. Returns a map where the first index is the cell.index() and the corresponding entries are the vertex.index()'s'
def cell_to_vertices_map(meshClass):
	mesh = meshClass.boundaryMesh
	tempArray = np.empty([mesh.num_cells(),3])
	for cells2 in cells(mesh):
		for j in range(3):
			tempArray[cells2.index()][j] = cells2.entities(0)[j]
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
# Since the edges are not globally indexed like the cells, it is assumed that the iterator always calls the edges in the same order.
# there seems to be some sort of indexing, but only 84,78,77.
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
	return Estar*Thickness*(2*(1/np.tan(cell_edges_and_opposite_angle_map[cell.index()][edge][1]))**2 + 1 - poisson_ratio)/(16*(1 - poisson_ratio**2)*cell.volume())
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
	return Estar*Thickness*(2*(1/np.tan(cell_edges_and_opposite_angle_map[cell.index()][edge_i][1]))*(1/np.tan(cell_edges_and_opposite_angle_map[cell.index()][edge_j][1])) - 1 + poisson_ratio)/(16*(1 - poisson_ratio**2)*cell.volume())
#TRBS TRiangluar Biquadratic Springs, energy per cell.
#Currently not used, force is the derivative of this.
def energy_TRBS(meshClass, cell, initial_cell_edges_and_opposite_angle_map, cell_edges_and_opposite_angle_map, Thickness = 0.1):
	# mesh = meshClass.boundaryMesh
	mesh = meshClass

	try:
		edgeIndex = edge.index()
	except AttributeError:
		edgeIndex = edge
	try:
		cellIndex = cell.index()
	except AttributeError:
		cell = Cell(mesh, cell)

	first_sum = 0
	mixed_term_1 = 0
	mixed_term_2 = 0

	#edge=0,1,2
	for edge in range(3):
		first_sum += 1/4 * tensile_stiffness_of_edge(meshClass, cell, edge, cell_edges_and_opposite_angle_map) \
						* (initial_cell_edges_and_opposite_angle_map[cell.index()][edge][0]**2 - cell_edges_and_opposite_angle_map[cell.index()][edge][0]**2)**2
		mixed_term_1 += 1/2 * angular_stiffness_of_edge(meshClass, cell, edge, edge-1, cell_edges_and_opposite_angle_map) \
						* (initial_cell_edges_and_opposite_angle_map[cell.index()][edge][0]**2 - cell_edges_and_opposite_angle_map[cell.index()][edge][0]**2) \
						* (initial_cell_edges_and_opposite_angle_map[cell.index()][edge-1][0]**2 - cell_edges_and_opposite_angle_map[cell.index()][edge-1][0]**2)
		mixed_term_2 += 1/2 * angular_stiffness_of_edge(meshClass, cell, edge, edge-1, cell_edges_and_opposite_angle_map) \
						* (initial_cell_edges_and_opposite_angle_map[cell.index()][edge][0]**2 - cell_edges_and_opposite_angle_map[cell.index()][edge][0]**2) \
						* (initial_cell_edges_and_opposite_angle_map[cell.index()][edge-2][0]**2 - cell_edges_and_opposite_angle_map[cell.index()][edge-2][0]**2)
	return first_sum + mixed_term_1 + mixed_term_2



# calculates all edges attached to a vertex. Returns an array where the index is the vertex.index() and the entries are the edge.index()'s
def vertex_to_edges_map(meshClass):
	mesh = meshClass.boundaryMesh
	#mesh = meshClass
	vertex_to_edges_map_untrimmed = np.zeros([mesh.num_vertices(),20], dtype=int)
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



# calculates Estar for different Cdc42 concentrations on the surface.
def Estar(meshClass, tipThreshold, shaftThreshold, baseThreshold, tipValue, shaftValue, baseValue, poisson_ratio = 0.5):
	#Estar = E/(1-poisson_ratio**2)
	list_of_Estar = [0.0]*meshClass.boundaryMesh.num_cells()

	for cell2010 in cells(meshClass.boundaryMesh):
			tempCellMidpoint = cell2010.midpoint()
			if meshClass.currentSolutionFunction(tempCellMidpoint) >= meshClass.u_sum_initial*tipThreshold:
				list_of_Estar[cell2010.index()] = tipValue/(1.0-poisson_ratio**2)
			elif meshClass.currentSolutionFunction(tempCellMidpoint) <= meshClass.u_sum_initial*baseThreshold:
				list_of_Estar[cell2010.index()] = baseValue/(1.0-poisson_ratio**2)
			else:
				list_of_Estar[cell2010.index()] = shaftValue/(1.0-poisson_ratio**2)
	return list_of_Estar
	



# my force on vertex as seen in the Delingette 2008 paper "Biquadratic and quadratic springs...".
# The showed Energy per cell was derived over 3 dim to get said force. The full formula can be found at Goldenbogen et all.'s
# "Dynamics of cell wall elasticity pattern shapes the cell during yeast matingmorphogenesis" supplementary, page 3.
# It is assumed that each cell consists of 3 regular and 3 angular biquadratic springs. If some deformation of the mesh happens, these springs
# will excert force to return to their relaxed state.
# The relaxed state is given by the "initial_cell_edges_and_opposite_angle_map", therefore it is assumed that at the beginning the whole
# mesh is relaxed.
def TRBS_force_on_vertex(meshClass, vertex, Thickness = 0.1, poisson_ratio = 0.5):
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
	force_on_vertex = [0.0,0.0,0.0]
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
		# save all the relevant edges, more specifially those attached to the given vertex
		relevant_edges = list(set(edges_to_consider).intersection(specific_cell_edges_to_consider))
		#print("relevant_edges", relevant_edges)
		# gather all the data to those edges
		initial_relevant_edges_and_angles = [[0,0.0],[0,0.0]]
		relevant_edges_and_angles = [[0,0.0],[0,0.0]]
		edgeCounter2 = 0
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
			#print(vertices_of_this_edge[0].index())
			#print(vertices_of_this_edge[1].index())
			#if n >=34:
			#	print(meshClass.coordinates[vertices_of_this_edge[0].index()] - meshClass.coordinates[vertices_of_this_edge[1].index()])		

			force_on_vertex += meshClass.Estar[cellIndices]*Thickness*(2.0*(1.0/np.tan(relevant_edges_and_angles[edgeCounter2][1]))**2 + 1.0 - poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*Cell(mesh, cellIndices).volume()) \
							* (Edge(mesh, relevant_edges_and_angles[edgeCounter2][0]).length()**2 - cell_edge_length(cellIndices, edge, initial_cell_edges_and_opposite_angle_map)**2) \
							* (1.0 + 0) \
							* (meshClass.coordinates[vertices_of_this_edge[0].index()] - meshClass.coordinates[vertices_of_this_edge[1].index()]) \
							+ (meshClass.Estar[cellIndices] \
							* Thickness \
							* (2*(1.0\
							/ np.tan(cell_edge_opposing_angle(cellIndices, relevant_edges[edgeCounter2-1], cell_edges_and_opposite_angle_map)))\
							* (1.0\
							/ np.tan(cell_edge_opposing_angle(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map))) \
							- 1.0 + poisson_ratio)\
							/ (16.0*(1.0 - poisson_ratio**2)\
							* Cell(mesh, cellIndices).volume()) \
							* ((cell_edge_length(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map)\
							* cell_edge_length(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map)) \
							- cell_edge_length(cellIndices, the_opposing_edge, initial_cell_edges_and_opposite_angle_map)**2) \
							+ meshClass.Estar[cellIndices]*Thickness*(2*(1.0/np.tan(relevant_edges_and_angles[edgeCounter2][1]))*(1.0/np.tan(cell_edge_opposing_angle(cellIndices, relevant_edges[edgeCounter2-1], cell_edges_and_opposite_angle_map))) - 1.0 + poisson_ratio)/(16.0*(1.0 - poisson_ratio**2)*Cell(mesh, cellIndices).volume()) \
							* (cell_edge_length(cellIndices, edge, cell_edges_and_opposite_angle_map)**2 - cell_edge_length(cellIndices, edge, initial_cell_edges_and_opposite_angle_map)**2)) \
							* (vertices_of_this_edge[0].point().array() - vertices_of_this_edge[1].point().array())
			if n > 25:
				if force_on_vertex[0] / force_on_vertex_array[n-1][vertex.index()][0] > 100:
					print("current foV", force_on_vertex)
					print("meshClass.Estar[cellIndices]", meshClass.Estar[cellIndices])
					print("arctan1", (1.0/np.tan(relevant_edges_and_angles[edgeCounter2][1])))
					print("edge length difference:", (Edge(mesh, relevant_edges_and_angles[edgeCounter2][0]).length()**2 - cell_edge_length(cellIndices, edge, initial_cell_edges_and_opposite_angle_map)**2))
					print("edge length vector", (meshClass.coordinates[vertices_of_this_edge[0].index()] - meshClass.coordinates[vertices_of_this_edge[1].index()]))
					print("arctan2", 1.0/np.tan(cell_edge_opposing_angle(cellIndices, relevant_edges[edgeCounter2-1], cell_edges_and_opposite_angle_map)))
					print("arctan3", 1.0/np.tan(cell_edge_opposing_angle(cellIndices, the_opposing_edge, cell_edges_and_opposite_angle_map)))
					print("cell_volume1", Cell(mesh, cellIndices).volume())
					print("previous foV:", force_on_vertex_array[n-1][vertex.index()])
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

#calculates the turgor pressure on a vertex, also seen in the supplementary of Goldenbogen et. alls paper 
#"Dynamics of cell wall elasticity pattern shapes the cell during yeast matingmorphogenesis", page 4
#Takes the vertex.index()
def turgor_pressure_on_vertex(meshClass, vertex, pressure):
	turgor_pressure_on_vertex = [0.0,0.0,0.0]
	cellVolume = 0.0
	for cellsIndices in meshClass.vertex_to_cells_map[vertex]:
		cellVolume += meshClass.cell_volumes[cellsIndices]
	cellVolume = 16.0
	#print(cellVolume)
	turgor_pressure_on_vertex = 1/3 * cellVolume * pressure * meshClass.vertex_normal_map[vertex]
	return turgor_pressure_on_vertex
