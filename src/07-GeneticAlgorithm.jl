# Genetic Algorithm
## Algorithms inspired from natural evolution
## Example of a genetic algorithm
## Relevant vocabulary
## Modeling individual
mutable struct Individual
	genes::Vector
	fitness::Union{Number,Nothing}

	Individual(genes::Vector) = new(genes, nothing)
end


# `genes_count` is the number of genes of the individuals
# `gene_factor` is a function (index, individual) -> value
function build_individual(genes_count::Integer, gene_factory::Function)
	return Individual([gene_factory(i, nothing) for i in 1:genes_count])
end


function compute_fitness(individual::Individual, f::Function)
	# Was the fitness previously computed? If yes, simply return it
	!isnothing(individual.fitness) && return individual.fitness

	# Else fitness is computed
	individual.fitness = f(individual.genes)
	return individual.fitness
end


# the number of genes the individual has
genes_count(individual::Individual) = length(individual.genes)


# Build a population made of `individual_count` individual, each having `gene_count` genes
# `gene_factory` is a two-argument function
function build_population(
	individual_count::Integer,
	gene_count::Integer,
	gene_factory::Function
)
	return [build_individual(gene_count, gene_factory) for _ in 1:individual_count]
end


@testset "Individual with characters" begin
	Random.seed!(42)
	pop = build_population(100, 10, (i, ind)->rand('a':'z'))
	@test length(pop) == 100
	@test all(i->genes_count(i) == 10, pop)
end


Random.seed!(42)
POPULATION_SIZE = 1000
individuals = build_population(POPULATION_SIZE, 3, (i, ind)->rand('a':'z'))
cat_fitness(genes) = sum(map(t->t[1] == t[2], zip(['c', 'a', 't'], genes)))
foreach(i->compute_fitness(i, cat_fitness), individuals)

# Representing the population
tick_positions = [0.5, 1.5, 2.5, 3.5]
histogram(
	map(i->i.fitness, individuals),
	bins=[0, 1, 2, 3, 4],
	xticks = (tick_positions, ["0", "1", "2", "3"]),
	xlabel="Fitness",
	label="Number of individuals")


Random.seed!(42)
POPULATION_SIZE = 100_000
individuals = build_population(POPULATION_SIZE, 3, (i, ind)->rand('a':'z'))
cat_fitness(genes) = sum(map(t->t[1] == t[2], zip(['c', 'a', 't'], genes)))
foreach(i->compute_fitness(i, cat_fitness), individuals)

# Representing the population
tick_positions = [0.5, 1.5, 2.5, 3.5]
histogram(
	map(i->i.fitness, individuals),
	bins=[0, 1, 2, 3, 4],
	xticks = (tick_positions, ["0", "1", "2", "3"]),
	xlabel="Fitness",
	label="Number of individuals",
	yaxis=:log)


## Crossover genetic operation
abstract type GAAbstractCrossoverOperation end


struct GACrossoverOperation <: GAAbstractCrossoverOperation end


# Return a new individual, result of crossover between `i1` and `i2`
function crossover(op::GACrossoverOperation, i1::Individual, i2::Individual)
	return crossover(op, i1, i2, pick_cutpoint(i1))
end


pick_cutpoint(i::Individual) = rand(1:genes_count(i))


# Return a new individual
function crossover(
	::GACrossoverOperation,
	i1::Individual,
	i2::Individual,
	cutpoint::Integer
)
	# Sanity checks
	@assert genes_count(i1) == genes_count(i2)
	@assert 1 <= cutpoint <= genes_count(i1)

	return Individual(vcat(i1.genes[1:cutpoint], i2.genes[cutpoint+1:end]))
end


@testset "Crossover" begin
	i1 = Individual(['a', 'b', 'c', 'd'])
	i2 = Individual(['e', 'f', 'g', 'h'])
	op = GACrossoverOperation()
	child = crossover(op, i1, i2, 2)
	@test child.genes == ['a', 'b', 'g', 'h']

	child = crossover(op, i1, i2, 3)
	@test child.genes == ['a', 'b', 'c', 'h']

	child = crossover(op, i1, i2, 4)
	@test child.genes == ['a', 'b', 'c', 'd']

	child = crossover(op, i1, i2, 1)
	@test child.genes == ['a', 'f', 'g', 'h']
end


## Mutation genetic operation
abstract type GAAbstractMutationOperation end


function mutate(op::T, individual::Individual, gene_factory) where T<:GAAbstractMutationOperation
	new_individual = Individual(copy(individual.genes))
	mutate!(op, new_individual, gene_factory)
	return new_individual
end


struct GAMutationOperation <: GAAbstractMutationOperation
	mutation_rate::Float64

	function GAMutationOperation(mutation_rate::Float64)
		@assert 0.0 <= mutation_rate <= 1.0 "Incorrect mutation_rate"
		return new(mutation_rate)
	end
end


# Mutate the individual provided as second argument
function mutate!(op::GAMutationOperation, individual::Individual, gene_factory::Function)
	for i in 1:genes_count(individual)
		if rand() <= op.mutation_rate
			individual.genes[i] = gene_factory(i, individual)
		end
	end
end


@testset "Mutation" begin
	Random.seed!(42)
	i = Individual(['a', 'b', 'c', 'd'])
	gene_factory = (idx, ind)-> rand('a':'z')
	op = GAMutationOperation(0.5)
	mutated_i = mutate(op, i, gene_factory)
	@test mutated_i.genes == ['a', 'm', 'c', 'd']

	mutated_i = mutate(op, i, gene_factory)
	@test mutated_i.genes == ['p', 'b', 'h', 'd']

	mutated_i = mutate(op, i, gene_factory)
	@test mutated_i.genes == ['a', 'g', 'c', 'h']
end


## Parent selection
struct GATournamentSelection
	tournament_size::Int
	GATournamentSelection() = GATournamentSelection(5)
	GATournamentSelection(tournament_size::Int) = new(tournament_size)
end


# Select an individual from a `population` using the `fitness function`.
# Individuals are compared using the `compare_fitness` and the parameters of the selection
# are carried out in `selection`.
function select_individual(
	selection::GATournamentSelection,
	population::Vector{Individual},
	fitness::Function,
	compare_fitness::Function
)
	best = nothing
	population_size = length(population)

	for i in 1:selection.tournament_size
		ind = population[rand(1:population_size)]
		if isnothing(best) || is_better(fitness, ind, best, compare_fitness)
			best = ind
		end
	end
	return best
end


# Return true if `ind1` is better than `ind2`
function is_better(fitness::Function, ind1::Individual, ind2::Individual, compare_fitness::Function)
	# Make sure we have the fitness of both individuals
	compute_fitness!(ind1::Individual, fitness::Function)
	compute_fitness!(ind2::Individual, fitness::Function)
	return compare_fitness(ind1.fitness, ind2.fitness)
end

# Avoid to compute the fitness more than once per individual
function compute_fitness!(ind::Individual, fitness::Function)
	isnothing(ind.fitness) || return
	ind.fitness = fitness(ind.genes)
end


# Build a new population from an existing population
function build_new_population(
	selection::GATournamentSelection,
	current_population::Vector{Individual},
	fitness::Function,
	gene_factory::Function,
	compare_fitness::Function,
	mutation_operator::GAAbstractMutationOperation,
	crossover_operator::GAAbstractCrossoverOperation,
	elitism::Bool
)
	new_population = Individual[]
	best = nothing

	offset = 0
	if elitism
		best_in_current = current_population[1]
		for i in 1:length(current_population)
			if is_better(fitness, current_population[i], best_in_current, compare_fitness)
				best_in_current = current_population[i]
			end
		end
		push!(new_population, best_in_current)
		offset = 1
	end

	for _ in 1:(length(current_population) - offset)
		ind1 = select_individual(selection, current_population, fitness, compare_fitness)
		ind2 = select_individual(selection, current_population, fitness, compare_fitness)
		child = mutate(mutation_operator, crossover(crossover_operator, ind1, ind2), gene_factory)
		push!(new_population, child)
		if isnothing(best) || is_better(fitness, child, best, compare_fitness)
			best = child
		end
	end
	return new_population
end


## Monitoring the evolution and termination
struct GALog
	generation::Int 			# Index of the population
	best_fitness::Float64		# Best fitness of the population
	worse_fitness::Float64		# Worse fitness
	average_fitness::Float64	# Average fitness
	median_fitness::Float64		# Median fitness
	best::Individual			# Best individual
end


## Terminating the algorithm
terminate_after_10_generations(logs) = logs[end].generation == 10


function terminate(
	frame::Int=5,
	delta_fitness::Number=0.01,
	max_generations::Int=100
)
	function f(logs)
		current_generation = logs[end].generation

		# The number of generations is limited
		current_generation < frame && return false

		# The number of generations is limited
		current_generation > max_generations && return true

		old_fitness = logs[end-frame+1].best_fitness
		current_fitness = logs[end].best_fitness
		return ((current_fitness - old_fitness) / old_fitness) <= delta_fitness
	end
	return f
end


function best_individual(population::Vector{Individual}, fitness::Function, compare_fitness::Function)
	best = nothing
	for ind in population
		if isnothing(best) || is_better(fitness, ind, best, compare_fitness)
			best = ind
		end
	end
	return best
end


function worse_individual(population::Vector{Individual}, fitness::Function, compare_fitness::Function)
	worse = nothing
	for ind in population
		if isnothing(worse) || is_better(fitness, worse, ind, compare_fitness)
			worse = ind
		end
	end
	return worse
end


## Running the genetic algorithm
# Entry point of the algorithm.
function ga_run(
	fitness::Function,
	build_gene::Function,
	genes_count::Int
	;
	population_size::Int = 10,
	termination::Function = terminate_after_10_generations,
	seed::Int = 42,
	compare_fitness::Function = >,
	mutation_operator::GAAbstractMutationOperation=GAMutationOperation(0.1),
	crossover_operator::GAAbstractCrossoverOperation=GACrossoverOperation(),
	selection = GATournamentSelection(),
	elitism::Bool = true,
	verbose::Bool = true,
)
	Random.seed!(seed)

	logs = []
	population = build_population(population_size, genes_count, build_gene)

	generation = 0
	while(isempty(logs) || !termination(logs))
		generation += 1
		new_population = build_new_population(
							selection,
							population,
							fitness,
							build_gene,
							compare_fitness,
							mutation_operator,
							crossover_operator,
							elitism,
						)

		best = best_individual(new_population, is_better, compare_fitness)
		worse = worse_individual(new_population, is_better, compare_fitness)
		fitness_values = map(i->i.fitness, new_population)
		average_fitness = mean(fitness_values)
		median_fitness = median(fitness_values)
		log = GALog(
			generation,
			best.fitness,
			worse.fitness,
			average_fitness,
			median_fitness,
			best
		)
		push!(logs, log)
		verbose && @info "Generation $(log.generation) : $(log.best.fitness)"

		population = new_population
	end

	return logs
end


## Our algorithm in action
distance_from_cat(genes) = sum(c1 == c2 for (c1, c2) in zip(genes, "cat"))


distance_from_cat(['a', 'b', 'c']) 		# return 0
distance_from_cat(['a', 'b', 't']) 		# return 1
distance_from_cat(['c', 'a', 't']) 		# return 3


logs = ga_run(
	distance_from_cat,			# Our fitness function
	(_, _) -> rand('a':'z'),	# A character between 'a' and 'z' is the value of a gene
	3							# Our secret word has 3 letters
	;
	population_size = 100,
	termination=(logs->logs[end].best_fitness == 3)
)


Individual(['c', 'a', 't'])


@testset "Secret word" begin
	logs = ga_run(
		distance_from_cat,			# Our fitness function
		(_, _) -> rand('a':'z'),	# A character between 'a' and 'z' is the value of a gene
		3							# Our secret word has 3 letters
		;
		population_size = 100,
		termination=(logs->logs[end].best_fitness == 3)
	)
	@test join(logs[end].best.genes) == "cat"
end


## What have we seen in this chapter?
