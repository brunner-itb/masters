class InitialCondition(Expression):
	def eval_cell(self, value, x, ufc_cell):
		value[0] = np.random.rand(1)

u_D = Expression("rand()/100000", degree=1)


class FEMMesh:
	'A class which should be able to incorporate all meshes, created or given, and provides all necessary parameters and values'
	def __init__(self, Mesh, Name, Gender, Source, h_real, u_max_start, SaveFile): #createMesh = True):
		#if createMesh == False:
		#read in the given mesh
		self.mesh = Mesh
		#define name for FileName purposes
		self.name = Name
		#define the gender
		self.gender = Gender
		#get the h value
		self.h_real = h_real
		#make it a dolfin Constant:
		self.h = Constant(h_real)
		#define the u_max_start value, the max concentration of Cdc42, used for h calculation:
		self.u_max_start = u_max_start

		#create the File to save the calculated data in
		try:
			self.saveFile = File(SaveFile)
		except RuntimeError:
			self.saveFile = XDMFFile(SaveFile)

		self.mesh = Mesh

		#Make it an only surface mesh, unordered, meaning every normal vector points outwards
		self.boundaryMesh = BoundaryMesh(Mesh, 'exterior', False)
		#write it to File
		#self.fileName = 'mesh_%s_unordered.xml' % self.name
		#File(self.fileName) << self.boundaryMesh 
		#parse it back in to extract the Orientation
		#self.tree = ET.parse(self.fileName)
		#self.triangles = self.tree.findall('mesh/cells/triangle')
		#order the mesh so it can be iterated over
		self.boundaryMesh.order()
		#get vertex coordinates for growing purposes
		self.coordinates = self.boundaryMesh.coordinates()
		#initialize vertex edge connectivity
		self.boundaryMesh.init(0,1)
		#splice the mesh into x splices and determine which vertex to put into which splice_
		#self.classArrayHoldingSlices = meshSlicing(self, amountOfSlices)
		#save every cells orientation as an array
		self.orientation = meshClassOrientation(self, straightLengthFactor)
		#self.orientation = myCellOrientation(self, amountOfSlices, straightLengthFactor)
		#get normalvector for every cell:
		self.normalVectors = cellNormals(self.boundaryMesh)
		#get the starting rmax() (inner radius of all cells, max) value to compare against to trigger Refinement
		self.start_hmax = self.boundaryMesh.hmax()

		#create a functionSpace, for future use
		self.functionSpace = FunctionSpace(self.boundaryMesh, 'CG', 1)
		#create trial and test-functions:
		if activeSurfaceSource == True:
			self.trialFunction = interpolate(Constant(0.0), self.functionSpace) #(u)
		else:
			self.trialFunction = interpolate(u_D, self.functionSpace) #interpolate(Constant(0.0), self.functionSpace)
		self.testFunction = TestFunction(self.functionSpace) #(v)
		#create function for solutions at current time-step: (u_1_n)
		self.currentSolutionFunction = Function(self.functionSpace)

		#define the meshes Source
		self.source = Source
		#define the meshes Stimulus, refers to global parameters as of now, should be changed in the future
		#Element is given so dolfin evaluates the optimal quadrature degree according to the given Expression.
		#testStimulus is for plotting in 2D

		#self.stimulus = Expression('(1.0-h)*Ka/std::sqrt(pow(x[0] - source0, 2) + pow(x[1] - source1, 2) + pow(x[2] - source2, 2)) * exp(-std::sqrt(pow(x[0] - source0, 2) + pow(x[1] - source1, 2) + pow(x[2] - source2, 2))/std::sqrt(Ds/kb))',\
 		#	element = self.functionSpace.ufl_element(), Ka = Ka, Ds = Ds, kb = kb, h=Constant(self.h_real), source0=self.source[0], source1=self.source[1], source2=self.source[2])
		self.twoDStimulus = Expression('(1.0-h)*Ka/std::sqrt(pow(x[0] - source0, 2) + pow(x[1] - source1, 2)) * exp(-std::sqrt(pow(x[0] - source0, 2) + pow(x[1] - source1, 2))/std::sqrt(Ds/(kb * Ka/std::sqrt(pow(x[0] - source0, 2) + pow(x[1] - source1, 2)))))',\
 			element = self.functionSpace.ufl_element(), Ka = Ka, Ds = Ds, kb = kb, h=Constant(self.h_real), source0=self.source[0], source1=self.source[1])
		self.stimulus = Expression('(1.0-h)*Ka/std::sqrt(pow(x[0] - source0, 2) + pow(x[1] - source1, 2) + pow(x[2] - source2, 2)) * exp(-std::sqrt(pow(x[0] - source0, 2) + pow(x[1] - source1, 2) + pow(x[2] - source2, 2))/std::sqrt(Ds/(kb * Ka/std::sqrt(pow(x[0] - source0, 2) + pow(x[1] - source1, 2) + pow(x[2] - source2, 2)))))',\
 			element = self.functionSpace.ufl_element(), Ka = Ka, Ds = Ds, kb = kb, h=Constant(self.h_real), source0=self.source[0], source1=self.source[1], source2=self.source[2])
		

		#get the stimulusString, required for the user defined add function
		self.stimulusString = self.getStimulusString()		
		#get the starting vertex and the corresponding cell:
		if activeSurfaceSource == True:
			self.stimulusCell = correspondingCell(self.boundaryMesh, self.coordinates[closestVertex(self.coordinates, self.source)])


		#init the PDE. Rmember that the myRefinement Function creates a new PDE so PDE and after-refinement-mesh are matching.
		#Because of lack of better knowing it is just these lines copied. Always change both if you want to make changes!
		self.PDE = inner((self.currentSolutionFunction - self.trialFunction) / k, self.testFunction)*dx - Dm*inner(nabla_grad(self.trialFunction), nabla_grad(self.testFunction))*dx \
		 - (1.0-self.h)*(nu*k0 + (nu*K*self.trialFunction**2)/(Km**2 + self.trialFunction**2))*self.testFunction*dx + eta*self.trialFunction*self.testFunction*dx - self.stimulus*self.testFunction*dx



		#create a variable that saves the inital u_sum, should be assigned only at n=0 of course
		self.u_sum_initial = None
		#get the needed maps:
		#get a vertex to cells map, so which vertex is part of which cells
		self.vertex_to_cells_map = vertex_to_cells_map(self)
		#create array with vertices and corresponding normals:
		self.vertex_normal_map = vertex_normal_map(self, self.vertex_to_cells_map, self.normalVectors)
		#calculate the map to get the vertices to be grown. Which cell has which vertices
		self.cell_to_vertices_map = cell_to_vertices_map(self)

		#initialize the gradient, for later use
		self.gradient = None
		#calculate the startingvolumes of the cells to scale u_max later on. Since the volume is expanding more Cdc42 should be available
		self.startVolume = self.getVolume()

		#if activeSurfaceSource == False, a growthThreshold is needed
		self.growthThreshold = None
		#initialize the cellGrowthDeterminingArray, is used in the growth
		self.cellGrowthDeterminingArray = None
		#init list of vertices to Grow
		self.verticesToGrow = None
		#create CellFunction
		self.cell_markers_boundary = MeshFunction('bool', self.boundaryMesh, self.boundaryMesh.topology().dim(), False) #CellFunction("bool", self.boundaryMesh)
		#self.cell_markers_boundary.set_all(True)
		self.isThereRefinementNecessary = False
		self.hmin = self.boundaryMesh.hmin()

		#force on vertex:

		self.vertex_to_edges_map = vertex_to_edges_map(self)
		self.initial_cell_edges_and_opposite_angle_map = cell_edges_and_opposite_angle_map(self)
		#at creation these should be equal, so no need to calculate twice
		self.cell_edges_and_opposite_angle_map = None #self.initial_cell_edges_and_opposite_angle_map #cell_edges_and_opposite_angle_map(self)

		self.force_on_vertex_list = [None]*self.boundaryMesh.num_vertices()

		self.Estar = None

		self.turgor_pressure_on_vertex_list = [None]*self.boundaryMesh.num_vertices()
		self.cell_volumes = cell_volumes(self)

	#calculate the volume of the mesh
	def getVolume(self):
		return assemble(Constant(1)*Measure("dx", domain=self.boundaryMesh))

	# helper function, the stimulus string is needed to add Stimuli and create fitting Expressions
	def getStimulusString(self):
		tempStimulusString = '(1.0-%s)*Ka/std::sqrt(pow(x[0] - %f, 2) + pow(x[1] - %f, 2) + pow(x[2] - %f, 2)) * exp(-std::sqrt(pow(x[0] - %f, 2) + pow(x[1] - %f, 2) + pow(x[2] - %f, 2))/std::sqrt(Ds/kb))' %(self.name ,self.source[0],self.source[1],self.source[2],self.source[0],self.source[1],self.source[2])
		return tempStimulusString
	def getStimulusStringWithH(self):
		tempStimulusString = '(1.0-h)*Ka/std::sqrt(pow(x[0] - %f, 2) + pow(x[1] - %f, 2) + pow(x[2] - %f, 2)) * exp(-std::sqrt(pow(x[0] - %f, 2) + pow(x[1] - %f, 2) + pow(x[2] - %f, 2))/std::sqrt(Ds/kb))' %(self.source[0],self.source[1],self.source[2],self.source[0],self.source[1],self.source[2])
		return tempStimulusString

	# ONLY USED TO ADD THE STIMULI, NOT THE WHOLE CLASS!
	def __add__(self, other):
		if isinstance(self, FEMMesh):
			if isinstance(other, FEMMesh):
				#print 'self and other are FEMMesh'
				newExpressionString = self.getStimulusString() + ' + ' + other.getStimulusString()
				#print(newExpressionString)
				#print("first source: ", self.source[:], 'second source: ', other.source[:])
				kwargs = {'element' : self.functionSpace.ufl_element() ,str(self.name) : Constant(self.h_real), str(other.name) : Constant(other.h_real), "Ka" : Ka, 'Ds' : Ds, 'kb' : kb}
				return Expression(newExpressionString, **kwargs)
			elif isinstance(other, Expression):
				#print 'self is FEMMesh, other is Expression'
				newExpressionString = self.getStimulusString()
				kwargs = {'element' : self.functionSpace.ufl_element() ,str(self.name) : Constant(self.h_real), "Ka" : Ka, 'Ds' : Ds, 'kb' : kb}
				return Expression(newExpressionString, **kwargs) + other
			# if the other is already a Sum of two Expressions, needed an extra case:
			elif str(type(other)) == "<class 'ufl.algebra.Sum'>":
				newExpressionString = self.getStimulusString()
				kwargs = {'element' : self.functionSpace.ufl_element() ,str(self.name) : Constant(self.h_real), "Ka" : Ka, 'Ds' : Ds, 'kb' : kb}
				return Expression(newExpressionString, **kwargs) + other
		# elif isinstance(self, Expression):
		# 	if isinstance(other, FEMMesh):
		# 		#print 'self is Expression, other is FEMMesh'
		# 		newExpressionString = other.getStimulusString()
		# 		kwargs = {'element' : other.functionSpace.ufl_element() ,str(other.name) : Constant(other.h_real), "Ka" : Ka, 'Ds' : Ds, 'kb' : kb}
		# 		return Expression(newExpressionString, **kwargs) + self
		# 	elif isinstance(other, Expression):
		# 		#print 'self and other are Expressions'
		# 		return self + other

	# get the gradient of all relevant Stimuli on my mesh
	def getGradient(self, usedMeshesList):
		tempCumulatedStimuli = None
		for Mesh in usedMeshesList:
			if Mesh != self: #everyone but myself
				if Mesh.gender != self.gender: #everyone of the opposite gender
					#print self.name, self.gender,'s opposite gender:', Mesh.name, Mesh.gender
					if tempCumulatedStimuli == None:
						tempCumulatedStimuli = Mesh
					else:
						tempCumulatedStimuli = Mesh + tempCumulatedStimuli
		#if no other gender is detected, take your "own" stimulus. Should mean an artifical stimulus has been applied
		if tempCumulatedStimuli == None:
			tempCumulatedStimuli = self.stimulus
		#print tempCumulatedStimuli
		#check if the overloaded add function has been used, if not make an Expression:
		try:
			self.gradient = gradient(project(tempCumulatedStimuli, self.functionSpace))
			return self.gradient
		except TypeError:
			if isinstance(tempCumulatedStimuli, Expression):
				print( 'gradient creation: tempCumulatedStimuli is an Expression')
				self.gradient = gradient(interpolate(tempCumulatedStimuli, self.functionSpace))
				return self.gradient
			elif isinstance(tempCumulatedStimuli, self.__class__):
				print( 'gradient creation: tempCumulatedStimuli is a Sum')
				self.gradient = gradient(interpolate(tempCumulatedStimuli, self.functionSpace))
				return self.gradient
			# elif isinstance(tempCumulatedStimuli, FEMMesh): #create an Expression which can be used in the gradient function. Similar to the __add__ function, just with one class
			# 	print 'gradient creation: tempCumulatedStimuli is a FEMMesh'
			# 	kwargs = {'element' : self.functionSpace.ufl_element() ,str(tempCumulatedStimuli.name) : Constant(tempCumulatedStimuli.h_real), "Ka" : Ka, 'Ds' : Ds, 'kb' : kb}
			# 	tempOnlyOneStimulusExpression = Expression(tempCumulatedStimuli.getStimulusString(), **kwargs)
			# 	self.gradient = gradient(project(tempOnlyOneStimulusExpression, self.functionSpace))
			# 	return self.gradient
			return self.gradient
		# if there is only one Mesh for the gradient to consider:
		except:
			#kwargs = {'element' : self.functionSpace.ufl_element() ,str(tempCumulatedStimuli.name) : Constant(tempCumulatedStimuli.h_real), "Ka" : Ka, 'Ds' : Ds, 'kb' : kb}
			#tempOnlyOneStimulusExpression = Expression(tempCumulatedStimuli.getStimulusString(), **kwargs)
			self.gradient = gradient(interpolate(tempCumulatedStimuli.stimulus, self.functionSpace))
			return self.gradient

	#if there are no activeSurfaceAreas the stimulus for each mesh has to be precalculated after initializing all meshes.
	#this is achieved by adding the stimuli strings of all other FEMMeshes and compiling it into an Expression
	def initSources(self, usedMeshesList):
		#i have to create a new self.stimulus which is basically all stimuli but my own added
		tempNewStimulusString = None
		for FEMMeshes in usedMeshesList:
			if FEMMeshes != self and FEMMeshes.gender != self.gender:
				if tempNewStimulusString == None:
					tempNewStimulusString = FEMMeshes.getStimulusStringWithH()
				else:
					tempNewStimulusString = tempNewStimulusString + ' + ' + FEMMeshes.getStimulusStringWithH()
		#if no other gender is detected, take your "own" stimulus. Should mean an artifical stimulus has been applied
		if tempNewStimulusString == None:
			tempNewStimulusString = self.getStimulusStringWithH()
			print('i am here:', tempNewStimulusString)
		kwargs = {'element' : self.functionSpace.ufl_element(), 'h' : Constant(self.h_real), "Ka" : Ka, 'Ds' : Ds, 'kb' : kb}
		self.stimulus = Expression(tempNewStimulusString, **kwargs)

	# used to reinitialize the PDE after initSources() was run. Updates the PDE function with the latest stimuli
	def initPDE(self):
		self.PDE = inner((self.currentSolutionFunction - self.trialFunction) / k, self.testFunction)*dx - Dm*inner(nabla_grad(self.trialFunction), nabla_grad(self.testFunction))*dx \
		 - (1.0-self.h)*(nu*k0 + (nu*K*self.trialFunction**2)/(Km**2 + self.trialFunction**2))*self.testFunction*dx + eta*self.trialFunction*self.testFunction*dx - self.stimulus*self.testFunction*dx


