# Traveling Salesman Problem
## Illustration of the problem
## Relevance of the Traveling Salesman Problem
## Naive approach
d = Dict("AB"=>10, "AD"=>10, "BC"=>10, "CD"=>10, "AC"=>20, "BD"=>8)

function tsp_abcd_f(genes)
	current_city = first(genes)
	path_length = 0
	for next_city in genes[2:end]
		segment = string(current_city, next_city)
		opposite_segment = string(next_city, current_city)
		if haskey(d, segment)
			path_length += d[segment]
		elseif haskey(d, opposite_segment)
			path_length += d[opposite_segment]
		end
		current_city = next_city
	end
	return path_length
end

logs = ga_run(
	tsp_abcd_f,
	(_, _) -> rand("ABCD"),		# A value is a city
	4							# We need to cover 4 cities
	;
	termination=terminate(10, 0, 100),
	compare_fitness=<
)
ga_plot(logs)


d = Dict("AB"=>10, "AD"=>10, "BC"=>10, "CD"=>10, "AC"=>20, "BD"=>8)

function tsp_abcd_f(genes)
	current_city = first(genes)
	path_length = 0
	for next_city in genes[2:end]
		segment = string(current_city, next_city)
		opposite_segment = string(next_city, current_city)
		if haskey(d, segment)
			path_length += d[segment]
		elseif haskey(d, opposite_segment)
			path_length += d[opposite_segment]
		end
		current_city = next_city
	end
	penalty = (4 - length(Set(genes))) * 100
	return path_length + penalty
end

logs = ga_run(
	tsp_abcd_f,
	(_, _) -> rand("ABCD"),		# A value is a city
	4							# We need to cover 4 cities
	;
	population_size = 1000,
	termination=terminate(10, 0, 100),
	compare_fitness=<
)
ga_plot(logs)


## The Roassal graphic system
import Pkg
Pkg.add(url="https://github.com/bergel/Roassal.jl")
using Roassal


c = RCanvas()
add!(c, translate_to!(RCircle(),  0,  0))
add!(c, translate_to!(RCircle(), 50,  0))
add!(c, translate_to!(RCircle(), 50, 50))
add!(c, translate_to!(RCircle(),  0, 50))
circles = get_shapes(c)

foreach(c -> oscillate!(c; duration=5.0, vertical=true, horizontal=true), circles)

add!(c, RLine(circles[1], circles[2]))
add!(c, RLine(circles[2], circles[3]))
add!(c, RLine(circles[3], circles[4]))
add!(c, RLine(circles[4], circles[1]))

rshow(c)


## Naive approach on a larger graph
# we now consider 20 cities
cities = [
	(100, 160), (20, 40), (60, 20), (180, 100), (200, 40),
	(60, 200), (80, 180), (40, 120), (140, 180), (140, 140),
	(20, 160), (200, 160), (180, 60), (100, 120), (120, 80),
	(100, 40), (20, 20), (60, 80), (180, 200), (160, 20)
]

function path_length_f(genes)
	path_length = 0
	for index in 2:length(genes)
		p1 = genes[index-1]
		p2 = genes[index]
		path_length += sqrt((p2[1]-p1[1])^2 + (p2[2]-p1[2])^2)
	end
	penalty = (length(genes) - length(Set(genes))) * 1000
	return path_length + penalty
end

logs = ga_run(
	path_length_f,
	(_, _) -> rand(cities),
	length(cities)
	;
	population_size = 1000,
	termination=terminate(30, 0, 100),
	compare_fitness=<
)
ga_plot(logs)


function view_path(logs)
	c = RCanvas()
	path = logs[end].best.genes

	for pos in cities
		box = RCircle(; color=RColor(0, 1.0, 0), radius=5, model=pos)
		translate_to!(box, pos)
		add!(c, box)
	end

	origin_pos = path[1]
	for pos in path[2:end]
		add!(c, RLine(get_shape(c, origin_pos), get_shape(c, pos)))
		origin_pos = pos
	end

	push_lines_back(c)

	rshow(c)
end
view_path(logs)


## Adequate genetic operations
## Swap mutation operation
struct GASwapMutationOperation <: GAAbstractMutationOperation
	mutation_rate::Float64

	function GASwapMutationOperation(mutation_rate::Float64)
		@assert 0.0 <= mutation_rate <= 1.0 "Incorrect mutation_rate"
		return new(mutation_rate)
	end
end


function mutate!(op::GASwapMutationOperation, individual::Individual, ::Function)
	for i1 in 1:genes_count(individual)
		if rand() <= op.mutation_rate
			i2 = rand(1:genes_count(individual))
			tmp = individual.genes[i1]
			individual.genes[i1] = individual.genes[i2]
			individual.genes[i2] = tmp
		end
	end
end


## Ordered crossover operation
struct GAOrderedCrossoverOperation <: GAAbstractCrossoverOperation end


function crossover(op::GAOrderedCrossoverOperation, i1::Individual, i2::Individual)
	cp1 = pick_cutpoint(i1)
	cp2 = pick_cutpoint(i2)
	if cp1 > cp2
		t = cp1
		cp1 = cp2
		cp2 = t
	end

	return crossover(op, i1, i2, cp1, cp2)
end


function crossover(
	op::GAOrderedCrossoverOperation,
	i1::Individual,
	i2::Individual,
	cp1::Int,
	cp2::Int
)
	swath = i1.genes[cp1:cp2]

	# We have a sequence made of `nothing`
	sequence = convert(Vector{Any}, fill(nothing, genes_count(i1)))

	# Swath is inserted in the sequence
	sequence[cp1:cp2] = swath

	# Values remaining to be inserted
	remaining = setdiff(i2.genes, swath)
	running_index = 1
	for v in remaining
		if !(v in sequence)
			# Look for a free slot to insert v
			while !isnothing(sequence[running_index])
				running_index += 1
			end
			sequence[running_index] = v
		end
	end
	return Individual(sequence)
end


@testset "Ordered crossover" begin
	i1 = Individual([8, 4, 7, 3, 6, 2, 5, 1, 9, 0])
	i2 = Individual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	op = GAOrderedCrossoverOperation()
	@test crossover(op, i1, i2, 4, 8).genes == [0, 4, 7, 3, 6, 2, 5, 1, 8, 9]
	@test crossover(op, i1, i2, 1, 4).genes == [8, 4, 7, 3, 0, 1, 2, 5, 6, 9]
	@test crossover(op, i1, i2, 9, 10).genes == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
	@test crossover(op, i1, i2, 5, 9).genes == [0, 3, 4, 7, 6, 2, 5, 1, 9, 8]
end


## Revisiting our large example
cities = [
	(100, 160), (20, 40), (60, 20), (180, 100), (200, 40),
	(60, 200), (80, 180), (40, 120), (140, 180), (140, 140),
	(20, 160), (200, 160), (180, 60), (100, 120), (120, 80),
	(100, 40), (20, 20), (60, 80), (180, 200), (160, 20)
]

function path_length_f(genes)
	path_length = 0
	for index in 2:length(genes)
		p1 = genes[index-1]
		p2 = genes[index]
		path_length += sqrt((p2[1]-p1[1])^2 + (p2[2]-p1[2])^2)
	end
	return path_length
end

function pick_value(individual_being_build::Individual)
	c = rand(cities)
	while c in individual_being_build.genes
		c = rand(cities)
	end
	return c
end

logs = ga_run(
	path_length_f,
	(_, ind) -> pick_value(ind),
	length(cities)
	;
	population_size = 500,
	termination=terminate(10, 0, 100),
	compare_fitness=<,
	mutation_operator=GASwapMutationOperation(0.01),
	crossover_operator=GAOrderedCrossoverOperation()
)
view_path(logs)


## What have we seen in this chapter?
