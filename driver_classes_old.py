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

# better calculation of the cellOrientation. It basically takes the normal vector given by fenics(cross product of two edges)
# puts a straight through it starting at the cell.midpoint() and ending at the midpoint + cellNormal*straightLengthFactor. 
# Now collision between the given mesh and the straight is checked and saved. Same thing is done again, just with
# normalVector *-1, so the straight goes in the opposite direction. Collision is saved again and compared against
# the previous collision. If there is less collision it is very probable that this orientation points outwards and is saved.
def cellOrientation(mesh, straightLengthFactor):
	mesh.init()
	meshTree = BoundingBoxTree()
	meshTree.build(mesh)
	n = cellNormals(mesh)
	cellOrientationArray = np.zeros(3*mesh.num_cells()).reshape(-1,3)
	for cell123 in cells(mesh):
	    cellNormal = n[cell123.index()]
	    startingPoint = cell123.midpoint()
	    endingPoint = startingPoint + Point(cellNormal[0] * straightLengthFactor, cellNormal[1] * straightLengthFactor, cellNormal[2] * straightLengthFactor)
	    
	    tempMesh = Mesh()
	    editor = MeshEditor()
	    editor.open(tempMesh, 'triangle', 2, 3)
	    editor.init_cells(1)
	    editor.init_vertices(4)
	    editor.add_vertex_global(1, 1, startingPoint)
	    editor.add_vertex_global(2, 2, endingPoint)
	    editor.add_vertex_global(3, 3, Point(endingPoint.x(), endingPoint.y()+0.00001, endingPoint.z()))
	    #for server version, no idea why
	    testVerticesToAdd = np.array([1,2,3], dtype='uintp')
	    editor.add_cell(0, testVerticesToAdd)
	    editor.close()

	    tempMeshTree = BoundingBoxTree()
	    tempMeshTree.build(tempMesh)

	    collisionsFirstTry, onlyZeros = meshTree.compute_collisions(tempMeshTree)
	   # print 'cell:', cell123.index(),'collisions:', collisionsFirstTry

	    if len(collisionsFirstTry) > 4:

	        cellNormal = -1*n[cell123.index()]
	        endingPoint = startingPoint + Point(cellNormal[0] * straightLengthFactor, cellNormal[1] * straightLengthFactor, cellNormal[2] * straightLengthFactor)
	        tempMesh = Mesh()
	        editor = MeshEditor()
	        editor.open(tempMesh, 'triangle', 2, 3)
	        editor.init_cells(1)
	        editor.init_vertices(4)
	        editor.add_vertex_global(1, 1, startingPoint)
	        editor.add_vertex_global(2, 2, endingPoint)
	        editor.add_vertex_global(3, 3, Point(endingPoint.x(), endingPoint.y()+0.00001, endingPoint.z()))
	        #for server version, no idea why
	        testVerticesToAdd = np.array([1,2,3], dtype='uintp')
	        editor.add_cell(0, testVerticesToAdd)
	        editor.close()

	        tempMeshTree = BoundingBoxTree()
	        tempMeshTree.build(tempMesh)

	        collisionsSecondTry, onlyZeros = meshTree.compute_collisions(tempMeshTree)
	        if len(collisionsSecondTry) < len(collisionsFirstTry):
	            cellOrientationArray[cell123.index()] = -1
	        else:
	            cellOrientationArray[cell123.index()] = 1
	        #print 'cell:', cell123.index(),'collisions:', collisionsSecondTry
	    else:
	        cellOrientationArray[cell123.index()] = 1
	return cellOrientationArray




# hopefully deprecated function
def meshClassOrientation(meshClass, straightLengthFactor):
	#triangles = meshClass.triangles
	tempCellOrientation = cellOrientation(meshClass.boundaryMesh, straightLengthFactor)
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
		vertex_normal_map[vertex_index] *= 1/LA.norm(vertex_normal_map[vertex_index]) #norms said normal vector
	return vertex_normal_map

#calculates all vertex.index() that belong to a cell. Returns a map where the first index is the cell.index() and the corresponding entries are the vertex.index()'s'
def cell_to_vertices_map(meshClass):
	mesh = meshClass.boundaryMesh
	tempArray = np.empty([mesh.num_cells(),3])
	for cells2 in cells(mesh):
		for j in range(3):
			tempArray[cells2.index()][j] = cells2.entities(0)[j]
	return tempArray

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
	meshClass.boundaryMesh = refine(meshClass.boundaryMesh, refineFunction)

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
	meshClass.orientation = meshClassOrientation(meshClass, straightLengthFactor)

	meshClass.normalVectors = cellNormals(meshClass.boundaryMesh)
	meshClass.vertex_to_cells_map = vertex_to_cells_map(meshClass)
	meshClass.vertex_normal_map = vertex_normal_map(meshClass, meshClass.vertex_to_cells_map, meshClass.normalVectors)
	meshClass.cell_to_vertices_map = cell_to_vertices_map(meshClass)
	meshClass.cell_markers_boundary = MeshFunction('bool', meshClass.boundaryMesh, meshClass.boundaryMesh.topology().dim(), False)

	return listOfPDE
