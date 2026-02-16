# Neuroevolution with NEAT
## Vocabulary
## Node
mutable struct NENode
	id::Int
	kind::Symbol
	built_connections
	inputs_count::Int
	z_value::Number
	received_values_count::Int
	result::Number
	NENode(id::Int, kind::Symbol) = new(id, kind, [], 0, 0, 0, 0)
end


## Different kind of nodes
is_input(node::NENode) = node.kind == :input
is_output(node::NENode) = node.kind == :output
is_hidden(node::NENode) = node.kind == :hidden
is_bias(node::NENode) = node.kind == :bias


Base.copy(node::NENode) = NENode(node.id, node.kind)


# Provide an input value to the node, and contribute to the intermediate zValue
function evaluate!(node::NENode, value::Number)
	node.z_value += value
	node.received_values_count += 1

	# If we have not less values than expected, this means other nodes must
	# be evaluated before, so we return the function
	is_only_partially_evaluated(node) && return

	if is_input(node) || is_bias(node)
		node.result = node.z_value
	else
		node.result = sigmoid(node.z_value)
	end
	for (weight, outgoing_node) in node.built_connections
		evaluate!(outgoing_node, node.result * weight)
	end
end


is_only_partially_evaluated(node::NENode) = node.received_values_count < node.inputs_count


function Base.show(io::IO, n::NENode)
	print(io, "NENode(")
	print(io, n.id)
	print(io, ", :")
	print(io, n.kind)
	print(io, ")")
end


function reset_node(node::NENode)
	node.built_connections = []
	node.z_value = 0
	node.inputs_count = 0
	node.received_values_count = 0
	node.result = 0
	(is_input(node) || is_bias(node)) && (node.inputs_count = 1)
end


sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))


## Connections
mutable struct NEConnection
	in::Int
	out::Int
	weight::Number
	enabled::Bool
	innovation_number::Int64
end


function Base.copy(c::NEConnection)
	return NEConnection(c.in, c.out, c.weight, c.enabled, c.innovation_number)
end


is_enabled(c::NEConnection) = c.enabled


function outgoing_nodes(node::NENode)
	return map(p -> p[2], node.built_connections)
end


## Individual
mutable struct NEIndividual
	nodes::Vector{NENode}
	connections::Vector{NEConnection}
	fitness::Union{Number, Nothing}
	species

	NEIndividual() = NEIndividual([], [])
	NEIndividual(nodes, connections) = new(nodes, connections, nothing, nothing)
end


function Base.show(io::IO, individual::NEIndividual)
    print(io, "NEIndividual($(individual.nodes), $(individual.connections))")
end


function Base.copy(individual::NEIndividual)
	return NEIndividual(
		convert(Vector{NENode}, map(copy, individual.nodes)),
		convert(Vector{NEConnection}, map(copy, individual.connections))
	)
end


function add_node!(individual::NEIndividual, kind::Symbol)
	reset_network(individual)
	node = NENode(length(individual.nodes) + 1, kind)
	push!(individual.nodes, node)
	return node
end


add_hidden_node!(individual::NEIndividual) = add_node!(individual, :hidden)
add_input_node!(individual::NEIndividual)  = add_node!(individual, :input)
add_bias_node!(individual::NEIndividual)   = add_node!(individual, :bias)
add_output_node!(individual::NEIndividual) = add_node!(individual, :output)


input_nodes(individual::NEIndividual) = filter(is_input, individual.nodes)
input_and_bias_nodes(individual::NEIndividual) = filter(n -> is_input(n) || is_bias(n), individual.nodes)


output_nodes(individual::NEIndividual) = filter(is_output, individual.nodes)


function inputs_count(individual::NEIndividual)
	return length(input_nodes(individual))
end


function get_node(individual::NEIndividual, id::Int)
	return first(filter(n->n.id == id, individual.nodes))
end


function add_connection(individual::NEIndividual, connection)
	push!(individual.connections, connection)
	reset_network(individual)
end


## Networking building
function connect(individual::NEIndividual, from_node::Int, to_node::Int, weight::Number)
	connect(get_node(individual, from_node), get_node(individual, to_node), weight)
end

function connect(from_node::NENode, to_node::NENode, weight::Number)
	push!(from_node.built_connections, weight => to_node)
end


function build_network(individual::NEIndividual)
	@assert is_bias(get_node(individual, 1)) "Missing bias node"
	reset_network(individual)

	for c in filter(is_enabled, individual.connections)
		connect(individual.nodes[c.in], individual.nodes[c.out], c.weight)
		individual.nodes[c.out].inputs_count += 1
	end
end


function compute_fitness(individual::NEIndividual, fitness::Function)
	# If the fitness is already computed, then return it
	isnothing(individual.fitness) || return individual.fitness

	# Else it is computed
	individual.fitness = fitness(individual)
	return individual.fitness
end


# Evaluate the network using some input values. Returns the outputs.
function evaluate(individual::NEIndividual, values)
	if length(values) != inputs_count(individual)
		@error "Missing input values when evaluating the network"
		return
	end
	build_network(individual)

	# The value 1 is provided to the bias node
	for (n, v) in zip(input_and_bias_nodes(individual), vcat([1], values))
		evaluate!(n, v)
	end

	# Return the result of the network evaluation
	return map(n->n.result, output_nodes(individual))
end


function innovation_number_sequence(individual::NEIndividual)
	isempty(individual.connections) && return [0]
	return map(c->c.innovation_number, individual.connections)
end


connections_count(individual::NEIndividual) = length(individual.connections)


# Make a prediction. This method assumes that the number of outputs is the same as the
# number of different values the network can output.
function predict(individual::NEIndividual, inputs)
	outputs = evaluate(individual, inputs)

	# The prediction has to begin with 0
	return findmax(outputs)[2] - 1
end


function reset_network(individual::NEIndividual)
	individual.fitness = nothing
	for n in individual.nodes
		reset_node(n)
	end
end


## Species
mutable struct NESpecies
	id
	individuals
	size

	function NESpecies(id, individuals)
		new_species = new(id, individuals, length(individuals))

		# We make sure all the individuals have this species
		for ind in individuals
			ind.species = new_species
		end
		return new_species
	end
end


# Release all the individuals of a species. This considerably reduce the memory footprint
function release(species::NESpecies)
	species.individuals = nothing
end


## Speciation
mutable struct NESpeciation
	frame_size
	species
	NESpeciation(frame_size::Int = 3) = new(frame_size, [])
end


# Run the speciation algorithm for a given collection of individuals.
function process(speciation::NESpeciation, individuals)
	species = Dict{Int, Vector{NEIndividual}}()
	for ind in individuals
		seq = innovation_number_sequence(ind)
		innov_number = length(seq) <= speciation.frame_size ?
							seq[1] :
							seq[end-speciation.frame_size]
		grps = get!(species, innov_number, Vector{NEIndividual}())
		push!(grps, ind)
	end
	speciation.species = []
	for (id, inds) in species
		push!(speciation.species, NESpecies(id, inds))
	end
end


## Crossover operation
# Return a child individual that is the result of a crossover between individuals i1 and i2
# Assumes that the fitness of i1 is higher than the one of i2
function crossover(i1::NEIndividual, i2::NEIndividual)
	# i1 has a better fitness than i2
	if i1.fitness < i2.fitness
		return crossover(i2, i1)
	end

	# nodes of the child individual, a copy of i1 nodes
	new_nodes = map(copy, i1.nodes)

	# If i1 or i2 has no connection, then we create a new individual with no connection
	if isempty(i1.connections) || isempty(i2.connections)
		return NEIndividual(new_nodes, map(copy, i1.connections))
	end

	# connections of the child being produced
	new_connections = []

	index_i1 = 1
	index_i2 = 1
	should_iterate = true

	while(should_iterate)
		c1 = i1.connections[index_i1]
		c2 = i2.connections[index_i2]
		if c1.innovation_number == c2.innovation_number
			push!(new_connections, copy(rand()>0.5 ? c1 : c2))
			index_i1 += 1
			index_i2 += 1
		else
			should_iterate = false
		end

		if index_i1 > connections_count(i1) || index_i2 > connections_count(i2)
			new_connections = vcat(
				new_connections,
				map(copy, i1.connections[index_i1:end]),
			)
			return NEIndividual(new_nodes, new_connections)
		end
	end

	new_connections = vcat(new_connections, [copy(c) for c in i1.connections[index_i1:end]])
	new_individual = NEIndividual(new_nodes, new_connections)

	return new_individual
end


### Mutating by adding a connection
# Add a random connection between two nodes to an individual
function mutation_add_connection(neat, individual::NEIndividual)
	# Find two nodes for which we can connect.
	array = find_missing_connection_in(individual)
	# We did not find any, so we exit
	isnothing(array) && return

	# Add the connection
	c = NEConnection(array[1], array[2], random_weight(), true, neat.innovation_number)
	neat.innovation_number += 1
	add_connection(individual, c)
end


random_weight() = rand() * 40 - 20


function find_missing_connection_in(individual, left_tries = 5)
	# No connection can be made
	left_tries == 0 && return nothing

	# The connection goes from node1 to node2. Node1 cannot be output therefore
	is_relevant_as_first_node = n -> is_hidden(n) || is_input(n) || is_bias(n)
	node1 = rand(filter(is_relevant_as_first_node, individual.nodes))

	# Similarly, node2 cannot be input or bias
	is_relevant_as_second_node = n -> (is_hidden(n) || is_output(n)) && n !== node1
	node2 = rand(filter(is_relevant_as_second_node, individual.nodes))

	# Is there already a connection from node1 to node2?
	function connection_exist(c)
		return c.in == node1.id && c.out == node2.id
	end
	if !isnothing(findfirst(connection_exist, individual.connections))
		# We found a connection, we try once more
		return find_missing_connection_in(individual, left_tries - 1)
	end

	# We check there is no path from node1 to node2.
	# Adding a connection should not introduce a cycle
	if is_accessible(individual, node2, node1)
		return find_missing_connection_in(individual, left_tries - 1)
	end

	return (node1.id, node2.id)
end


function is_accessible(individual::NEIndividual, node2::NENode, node1::NENode)
	build_network(individual)
	return node1 in all_accessible_nodes_from(node2)
end


function all_accessible_nodes_from(node::NENode)
	reachable_nodes = Set{NENode}()
	_all_accessible_nodes_from(node, reachable_nodes)
	return reachable_nodes
end

function _all_accessible_nodes_from(node::NENode, reachable_nodes::Set{NENode})
	for n in outgoing_nodes(node)
		n in reachable_nodes && continue
		push!(reachable_nodes, n)
		_all_accessible_nodes_from(n, reachable_nodes)
	end
end


### Mutating by adding a node
# Add a hidden node and two connections in the individual
function mutation_add_node(neat, individual::NEIndividual)
	relevant_connections = filter(is_enabled, individual.connections)
	isempty(relevant_connections) && return

	# Disable a random connection
	c = rand(relevant_connections)
	c.enabled = false

	# Add a hidden node ...
	added_node = add_hidden_node!(individual)

	# ... and two connections
	c1 = NEConnection(c.in, added_node.id, 1, true, neat.innovation_number)
	add_connection(individual, c1)
	neat.innovation_number += 1

	c2 = NEConnection(added_node.id, c.out, c.weight, true, neat.innovation_number)
	push!(individual.connections, c2)
	neat.innovation_number += 1

	reset_network(individual)
end


### Mutating by changing a connection weight
# Modify the weight of a connection
function mutation_connection_weight(neat, individual::NEIndividual)
	isempty(individual.connections) && return
	rand(individual.connections).weight = random_weight()
	reset_network(individual)
end


## Logging the evolution
mutable struct NELog
	generation::Int
	speciation::NESpeciation
	min_fitness::Number
	max_fitness::Number
	average_fitness::Number
	best_individual::NEIndividual
end


## NEAT
mutable struct NEAT
	configuration::Dict
	population_size::Int
	population::Vector{NEIndividual}
	inputs_count::Int
	outputs_count::Int
	logs::Vector{NELog}
	fitness::Function
	speciation::NESpeciation
	generations_max::Int
	should_use_elitism::Bool
	verbose::Bool
	innovation_number::Int
	termination_function::Function

	function NEAT(
		inputs_count::Int,
		output_count::Int,
		fitness::Function ;
		configuration::Dict{Function,Number} = default_neat_configuration(),
		population_size = 150,
		generations_max = 10,
		should_use_elitism = true,
		verbose = true,
		termination_function::Function = (best_fit -> false)
	)
		return new(
			configuration,
			population_size,
			[],
			inputs_count,
			output_count,
			[],
			fitness,
			NESpeciation(),
			generations_max,
			should_use_elitism,
			verbose,
			0,
			termination_function
		)
	end
end


function default_neat_configuration()
	return Dict{Function,Number}(
		mutation_connection_weight => 0.2,
		mutation_add_connection => 0.2,
		mutation_add_node => 0.01,
		crossover => 0.2,
	)
end


function do_mutate(neat::NEAT, individual::NEIndividual)
	operations = shuffle([
		mutation_connection_weight,
		mutation_add_connection,
		mutation_add_node
	])
	for f in operations
		haskey(neat.configuration, f) || continue
		rand() > neat.configuration[f] && continue
		f(neat, individual)

		individual.fitness = nothing
		return
	end
	individual.fitness = nothing
end


function run(neat::NEAT)
	before_time = now()

	build_initial_population(neat)
	do_speciation(neat)
	do_log(neat)
	run_for(neat, neat.generations_max)

	after_time = now()
	if neat.verbose
		duration = Dates.canonicalize(after_time - before_time)
		println("Evolution duration: $(duration)")
	end
end


function build_initial_population(neat::NEAT)
	neat.population = []
	for _ in 1:neat.population_size
		i = NEIndividual()
		add_bias_node!(i)
		for _ in 1:neat.inputs_count
			add_input_node!(i)
		end
		for _ in 1:neat.outputs_count
			add_output_node!(i)
		end
		push!(neat.population, i)
	end
end


function do_speciation(neat::NEAT)
	neat.speciation = NESpeciation()
	process(neat.speciation, neat.population)
end


function do_log(neat::NEAT)
	all_fitnesses = map(i->compute_fitness(i, neat.fitness), neat.population)
	log = NELog(
		current_iteration(neat),
		neat.speciation,
		min(all_fitnesses...),
		max(all_fitnesses...),
		sum(all_fitnesses) / length(neat.population),
		find_best_individual(neat)
	)
	push!(neat.logs, log)
end


current_iteration(neat::NEAT) = length(neat.logs)


# Return the best individual, i.e., the fittest neural network
function find_best_individual(neat::NEAT)
	winner = neat.population[1]
	for i in neat.population[2:end]
		if compute_fitness(winner, neat.fitness) < compute_fitness(i, neat.fitness)
			winner = i
		end
	end
	return winner
end


function run_for(neat::NEAT, generations::Int)
	for g in 1:generations
		run_one_generation(neat)
		fit = neat.logs[end].best_individual.fitness
		avg = neat.logs[end].average_fitness
		worse = neat.logs[end].min_fitness
		if neat.verbose
			b = round(fit, digits=4)
			a = round(avg, digits=4)
			w = round(worse, digits=4)
			println("$g/$generations -- Fitness best: $(b), average: $(a), min: $(w)")
		end
		neat.termination_function(fit) && return
	end
end


# Run the evolution algorithm for one generation
function run_one_generation(neat::NEAT)
	do_speciation(neat)
	new_population = Vector{NEIndividual}(undef, neat.population_size)

	# Number of individual to create is either neat.population_size or minus 1
	index_to_start = 1
	if neat.should_use_elitism && current_iteration(neat) > 1
		new_population[1] = copy(find_best_individual(neat))
		index_to_start = 2
	end

	# Building a new population
	for index in index_to_start:neat.population_size
		new_population[index] = produce_new_individual(neat)
	end
	neat.population = new_population

	# Release all the individuals from the species.
	for s in neat.speciation.species
		release(s)
	end
	do_log(neat)
end


function produce_new_individual(neat::NEAT)
	local new_ind
	if rand() <= crossover_rate(neat)
		i1 = select_individual(neat)
		i2 = select_individual(neat, i1.species.individuals)
		if i1 !== i2
			new_ind = crossover(i1, i2)
		else
			new_ind = copy(i1)
		end
	else
		ind = select_individual(neat)
		new_ind = copy(select_individual(neat))
	end
	do_mutate(neat, new_ind)
	return new_ind
end


crossover_rate(neat::NEAT) = get(neat.configuration, crossover, 0)


select_individual(neat::NEAT) = select_individual(neat, neat.population)


# Use the tournament selection algorithm to pick the best individual
function select_individual(neat::NEAT, individuals; k=5)
	winner = rand(individuals)
	for _ in 1:k-1
		i = rand(winner.species.individuals)
		if compute_fitness(winner, neat.fitness) < compute_fitness(i, neat.fitness)
			winner = i
		end
	end
	return winner
end


# Return the best individual, i.e., the fittest neural network
best_individual(neat::NEAT) = neat.logs[end].best_individual


## Visualizing the evolution
using Plots: plot, plot!

function ne_plot(neat::NEAT)
	logs = neat.logs
	best_fitnesses = [l.max_fitness for l in logs]
	worse_fitnesses = [l.min_fitness for l in logs]
	average_fitnesses = [l.average_fitness for l in logs]

	p = plot()
	plot!(p, best_fitnesses, color=:green, label="best")
	plot!(p, average_fitnesses, color=:gray70, label="average")
	plot!(p, worse_fitnesses, color=:lightblue2, label="worse")
end


function ne_plot_species_evolution(neat::NEAT)
	species_count = map(l->length(l.speciation.species), neat.logs)
	plot(species_count, color=:blue, label="number of species")
end


## The XOR example
dataset = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

function xor_fitness(ind)
	score = 0
	for tup in dataset
		diff = evaluate(ind, tup[1:end-1])[1] - tup[end]
		score += diff^2
	end
	return -score / length(dataset)
end

Random.seed!(42)
neat = NEAT(2, 1, xor_fitness; generations_max = 180)
run(neat)


evaluate(best_individual(neat), [1, 1])


## The Iris example
# Data preparation
using CSV, DataFrames
df = CSV.read(Base.download("https://agileartificialintelligence.github.io/Datasets/iris.csv"), DataFrame)
iris_data = []
for index in 1:nrow(df)
	tp = values(df[index,:])
	local hot
	if tp[end] == "setosa" hot = 0 end
	if tp[end] == "versicolor" hot = 1 end
	if tp[end] == "virginica" hot = 2 end
	push!(iris_data, collect((tp[1:end-1]..., (hot))))
end

Random.seed!(42)
shuffled_iris_data = shuffle(iris_data)
cut = 0.8
data_training_length = round(Int, length(shuffled_iris_data) * cut)
data_test_length = round(Int, length(shuffled_iris_data) - data_training_length)
data_training = shuffled_iris_data[1:data_training_length]
data_test = shuffled_iris_data[end - data_test_length + 1:end]

# Running NEAT
function iris_fitness(ind)
	score = 0
	for row in data_training
		diff = predict(ind, row[1:end-1]) - row[end]
		score += diff^2
	end
	return -score
end

neat = NEAT(4, 3, iris_fitness ; generations_max=100)
run(neat)


best_ind = neat.logs[end].best_individual
correct_ratio = sum([predict(best_ind, row[1:end-1]) == row[end] for row in data_test]) / data_test_length
round(correct_ratio, digits = 2)


## Loading and saving
dataset = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
function xor_fitness(ind)
	score = 0
	for tup in dataset
		diff = evaluate(ind, tup[1:end-1])[1] - tup[end]
		score += diff^2
	end
	return -score / length(dataset)
end

population_size = 150
Random.seed!(42)

neat = NEAT(2, 1, xor_fitness; generations_max = 180, verbose = false,
	configuration = Dict{Function,Number}(
		mutation_connection_weight => 0.2,
		mutation_add_connection => 0.2,
		mutation_add_node => 0.01,
		crossover => 0.2,
	),
	population_size = population_size)
run(neat)

ind1 = neat.logs[end].best_individual

using Serialization
Serialization.serialize("result_of_xor.bin", ind1)


ind2 = Serialization.deserialize("result_of_xor.bin")
xor_fitness(ind2)


## What have we seen in this chapter?
