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


# Calculate the next source cell by calculating the neighboring probabilities + self, then making a weighted random pick
# Returns a cell object
def nextSourceCell(mesh, cell, gradient2, noiseFactor, lengthFactor=1):
	global calculateGradientSphere1
	global calculateGradientSphere2
	dim = 3 #len(gradient2(cell.midpoint()))
	#averageEdgeLengths = np.average(EdgeLengths(mesh))
	maxEdgeLengths = mesh.hmin()*lengthFactor #np.max(EdgeLengths(mesh))
	#print 'maxEdgeLengths', maxEdgeLengths	#get the gradient2 Vector and scale it with the average edge lengths
	gradientVector = np.zeros(dim)
	
	gradientNorm = LA.norm(gradient2(cell.midpoint()))
	for i in range(dim):
		gradientVector[i] = cell.midpoint()[i] + (1/gradientNorm) * gradient2(cell.midpoint())[i] * maxEdgeLengths
	#* (maxEdgeLengths/LA.norm(gradientVectorUnscaled))
	#get some random noise in there, using the normal distribution with sigma = 0.1
	#print 'LA.norm(gradient2(cell.midpoint()))', gradientNorm
	#print 'my norm function: ', norm(gradient2(cell.midpoint()))
	if gradientNorm == 0:
		print '################################'
		print '################################'
		print '              NaN               '
		print '################################'
		print '################################'
	gradientVectorNoise = np.zeros(dim)
	for ii in range(dim):
		gradientVectorNoise[ii] = gradientVector[ii] + noiseFactor*np.random.randn()
	#calculate which neighboring cell.midpoint() is closest to my gradientVectorNoise tip:
	# gradientVectorTip = np.zeros(dim)
	# for iii in range(dim):
	# 	gradientVectorTip[iii] = cell.midpoint()[iii] + gradientVectorNoise[iii]
	#dictionary with cell object as key, distance as value
	# if cell==cell2:
	# 	plot(gradient2*3000) ##MAYBE NAME ERROR, i give "gradient" to this function and also have a function called gradient!!
	# 	plt.show()
	#print 'LA.norm(gradientVector)', LA.norm(gradientVector)
	tempDiff = {}
	for meshcells in cells(mesh):
		tempValue = 0
		for iiii in range(dim):
			tempValue += np.abs(gradientVectorNoise[iiii] - meshcells.midpoint()[iiii])
		tempDiff[meshcells] = tempValue
		#tempDiff[meshcells] = {np.sum(np.abs(gradientVectorTip[i] - meshcells.midpoint()[i]) for i in range(dim))}
	#get the closest cell by aquiring the minimum distance and asking for the corresponding cell
	closestCell = min(tempDiff, key=lambda k: tempDiff[k])# tempDiff.keys()[tempDiff.values().index(min(tempDiff.values()))]
	if cell == closestCell:
		if closestCell == cell1:
			calculateGradientSphere2 == False
		else:
			calculateGradientSphere1 == False
	return closestCell


	#get probs as a ordered dictionary
	# probabilities = getCellNeighborProbabilities(mesh, cell, cell_neighbors, gradient)

	# newSourceCell = np.random.choice(probabilities.keys(), 1, p=probabilities.values())[0]
	# print 'old Cell: ', cell.index()
	# print 'new Cell: ', newSourceCell
	# if newSourceCell == cell.index():
	# 	if Cell(mesh, newSourceCell) == cell1:
	# 		calculateGradientSphere2 = False
	# 	else:
	# 		calculateGradientSphere1 = False
	# return Cell(mesh, newSourceCell)


############ Calculate the next Source Vertex based on the neighborProbablilites. Going backwards translates into a probability = 0.
############ Probability to stay is calculated via number of zeroes in the probability divided by the number of neighbors
def nextSourceVertex(meshcoord, StartIndex, neighborhood, gradient):
	global calculateGradientSphere1
	global calculateGradientSphere2
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
			calculateGradientSphere2 = False
		else:
			calculateGradientSphere1 = False
		return StartIndex
	for key in getNeighProb:
		getNeighProb[key] *= 1/getNeighProbSum
	# These can now be used as probabilities. Use numpy to make a weighted random pick from all neighbors:
	newStartIndex = np.random.choice(neighborhood_plusSelf, 1, p=getNeighProb.values())[0]
	print 'oldStartIndex: ', StartIndex
	print 'newStartIndex: ', newStartIndex
	print ''
	if newStartIndex == StartIndex:
		if newStartIndex == StartIndex1:
			calculateGradientSphere2 = False
		else:
			calculateGradientSphere1 = False
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

def cellNormals(mesh):
	n_1 = np.zeros(3*mesh.num_cells()).reshape(-1,3)
	#i = 0
	#for c in cells(mesh):
	#	n_1[i][0] = c.cell_normal()[0]
	#	n_1[i][1] = c.cell_normal()[1]
	#	n_1[i][2] = c.cell_normal()[2]	
	#	i += 1
	for ii in range(mesh.num_cells()):
		n_1[ii][0] = Cell(mesh, ii).cell_normal()[0]
		n_1[ii][1] = Cell(mesh, ii).cell_normal()[1]
		n_1[ii][2] = Cell(mesh, ii).cell_normal()[2]
	return n_1




################# get the cell orientation. 1 is outwards, -1 is inwards. Requires a triangle tree, previously loaded from the xml file of the unordered Boundary mesh.
def cellOrientation(triangles, Cell):
	try:
		triangle = triangles[Cell.index()]
	except AttributeError:
		triangle = triangles[Cell]
	#based on index parity. cyclic shift of the vertex ordering implies correctly ordered crossproduct (counterclockwise)
	if triangle.get('v0') < triangle.get('v1') < triangle.get('v2') or triangle.get('v1') < triangle.get('v2') < triangle.get('v0') or triangle.get('v2') < triangle.get('v0') < triangle.get('v1'):
		return 1
	else:
		return -1
def meshOrientation(meshClass):
	triangles = meshClass.triangles
	meshOrientationArray = [0]*meshClass.boundaryMesh.num_cells()
	for i in range(meshClass.boundaryMesh.num_cells()):
		meshOrientationArray[i] = cellOrientation(triangles, i)
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

#create a vertex to cells map, returns an array. The first array index is the vertex index, the corresponding array are the cells. 
#so vertex_to_cells_map[10] returns all cells that contain vertex.index() 10
def vertex_to_cells_map(mesh):
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
def vertex_normal_map(mesh, vertex_to_cells_map, normalVectors, triangles):
	vertex_normal_map = np.zeros([mesh.num_vertices(),3])
	for vertex_index in range(len(vertex_normal_map)): #gives all vertex indices, from 0 to num_vertices()
		for vertex_cells in vertex_to_cells_map[vertex_index]:  #gives all cells for a vertex, eg [0 1 2 3]
			vertex_normal_map[vertex_index] += normalVectors[vertex_cells]*cellOrientation(triangles, vertex_cells) #adds all normal vectors of the corresponding cells and check for Orientation
		vertex_normal_map[vertex_index] *= 1/LA.norm(vertex_normal_map[vertex_index]) #norms said normal vector
	return vertex_normal_map

#calculates all vertex.index() that belong to a cell. Returns a map where the first index is the cell.index() and the corresponding entries are the vertex.index()'s'
def cell_to_vertices_map(mesh):
	tempArray = np.empty([mesh.num_cells(),3])
	for cells2 in cells(mesh):
		for j in range(3):
			tempArray[cells2.index()][j] = cells2.entities(0)[j]
	return tempArray

#calculates which cells are supposed to grow based on a growthThreshold. Returns a list/array with cell.index()'s that should be grown in the next step.
def cellGrowthDeterminingArray(mesh, normalVectors, u, u_sum_initial, growthThreshold):
	cellGrowthDeterminingArray = []
	for tempCells in cells(mesh):
		if u(tempCells.midpoint()) >= u_sum_initial*growthThreshold:
			cellGrowthDeterminingArray.append(tempCells.index())
	return cellGrowthDeterminingArray

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
		meshcoord[verticesToGrow] += growthFactor*vertex_normal_map[verticesToGrow]
