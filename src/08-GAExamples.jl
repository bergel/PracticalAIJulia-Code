# Genetic Algorithm in Action
## Visualizing the evolution
function ga_plot(logs::Vector)
	best_fitnesses = [l.best_fitness for l in logs]
	worse_fitnesses = [l.worse_fitness for l in logs]
	average_fitnesses = [l.average_fitness for l in logs]

	p = plot()
	plot!(p, best_fitnesses, color=:green, label="best")
	plot!(p, average_fitnesses, color=:gray70, label="average")
	plot!(p, worse_fitnesses, color=:lightblue2, label="worse")
end


SECRET_WORD="anticonstitutionnellement"
distance_from_word(genes) = sum(c1 == c2 for (c1, c2) in zip(genes, SECRET_WORD))
logs = ga_run(
	distance_from_word,
	(_, _) -> rand('a':'z'),
	length(SECRET_WORD)
	;
	population_size = 100,
	termination=(logs->logs[end].best_fitness == length(SECRET_WORD))
)
ga_plot(logs)


SECRET_WORD="anticonstitutionnellement"
distance_from_word(genes) = sum(c1 == c2 for (c1, c2) in zip(genes, SECRET_WORD))
logs = ga_run(
	distance_from_word,
	(_, _) -> rand('a':'z'),
	length(SECRET_WORD)
	;
	population_size = 30,
	termination=(logs->logs[end].best_fitness == length(SECRET_WORD)),
	mutation_operator=GAMutationOperation(0.01)
)
ga_plot(logs)


## Mutation rate vs population size
## Fundamental theorem of arithmetic
function break_down(number_to_break_down)
	prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

	function prime_numbers_f(genes)
		numbers = filter(x->x > 0, prime_numbers .* genes)
		return abs(prod(numbers) - number_to_break_down)
	end

	logs = ga_run(
		prime_numbers_f,
		(_, _) -> rand([0, 1]),
		length(prime_numbers)
		;
		population_size = 1_000,
		termination=(logs)->logs[end].best_fitness == 0 || (logs[end].generation > 100),
		compare_fitness=<
	)

	pn = filter(!iszero, prime_numbers .* (logs[end].best.genes))
	if prod(pn) == number_to_break_down
		println("$(join(pn, " * ")) = $(number_to_break_down)")
	else
		println("Not able to break down $(number_to_break_down)")
	end
end

break_down(345)


# Return true
(big(2)^63 - 1) > prod(map(big, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]))


# Return false
(big(2)^63 - 1) > prod(map(big, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]))


## Knapsack problem
### The _unbounded knapsack problem_ variant
knapsack_max_weight = 15

# a box = (value, weight)
boxes = [(4, 12), (2, 1), (2, 2), (1, 1), (10, 4), (0, 0)]

function unbounded_knapsack_f(genes)
	sum_values = sum(x -> x[1], genes)
	sum_weights = sum(x -> x[2], genes)

	penalty = knapsack_max_weight < sum_weights ?
				abs(knapsack_max_weight - sum_weights) * 50 :
				0
	return sum_values - penalty
end

logs = ga_run(
	unbounded_knapsack_f,
	(_, _) -> rand(boxes),
	15
	;
	population_size = 100,
	termination=terminate(10, 0, 100)
)

solution = filter(x -> x != (0, 0), logs[end].best.genes)
total_values = sum(x -> x[1], solution)
total_weights = sum(x -> x[2], solution)
println("Total value = $(total_values), total weight = $(total_weights), solutions = $(solution)")


### The _0-1 knapsack problem_ variant
knapsack_max_weight = 15

# a box = (value, weight)
boxes = [(4, 12), (2, 1), (2, 2), (1, 1), (10, 4)]

binary_to_boxes(values) = [(iszero(x) ? (0, 0) : boxes[i]) for (i, x) in enumerate(values)]

function zero_one_bounded_knapsack_f(genes)
	decoded_boxes = binary_to_boxes(genes)

	sum_values = sum(x -> x[1], decoded_boxes)
	sum_weights = sum(x -> x[2], decoded_boxes)

	penalty = knapsack_max_weight < sum_weights ?
				abs(knapsack_max_weight - sum_weights) * 50 :
				0
	return sum_values - penalty
end

logs = ga_run(
	zero_one_bounded_knapsack_f,
	(_, _) -> rand([0, 1]),
	length(boxes)
	;
	population_size = 10000,
	termination=terminate(10, 0, 100)
)

solution = filter(x -> x != (0, 0), binary_to_boxes(logs[end].best.genes))
total_values = sum(x -> x[1], solution)
total_weights = sum(x -> x[2], solution)
println("Total value = $(total_values), total weight = $(total_weights), solutions = $(solution)")


### Coding and encoding
## Meeting room scheduling problem
# We assume each meeting is correctly defined
# A meeting = (start time, end time)
# both start time and end time are within 1 and 10
meetings = [(1, 3), (2, 3), (5, 6), (7, 9), (4, 7)]
meetings_count = length(meetings)

function room_scheduling_f(genes)
	distribution = []
	for _ in 1:meetings_count
		push!(distribution, [])
	end
	for (index, room_number) in enumerate(genes)
		push!(distribution[room_number], meetings[index])
	end
	overlap_count = 0
	for meetings_per_room in distribution
		table = zeros(Int, 10)
		for a_meeting in meetings_per_room
			for i in a_meeting[1]:a_meeting[2]
				table[i] += 1
			end
			overlap_count += length(filter(v -> v >= 2, table))
		end
	end
	return length(filter(!isempty, distribution)) + overlap_count
end

logs = ga_run(
	room_scheduling_f,
	(_, _) -> rand(1:meetings_count),
	meetings_count
	;
	population_size = 10000,
	termination=terminate(10, 0, 100),
	compare_fitness = <
)
necessary_rooms = length(Set(logs[end].best.genes))
println("Number of necessary rooms = $(necessary_rooms)")
for (i, m) in enumerate(meetings)
	println("Meeting $(m) in room $(logs[end].best.genes[i])")
end


## Mini sodoku
# Useful for padding the numbers
using Printf

numbers = [2, 4, 6, 8, 10, 12, 14, 16, 18]

# The different combinations to sum.
# E.g., the three first cells are summed up [1, 2, 3]
#		the diagonal top-left to bottom-right [1, 5, 9]
sums = [
	# Horizontal sums
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9],

	# Diagonal sums
	[1, 5, 9],
	[7, 5, 3],

	# Vertical sums
	[1, 4, 7],
	[2, 5, 8],
	[3, 6, 9]
]

function mini_soduku_f(genes)
	score = 0
	for s in sums
		sum_in_one_direction = sum(map(index -> genes[index], s))
		score += map(x -> abs(x - 30), sum_in_one_direction)
	end
	penalty = length(genes) - length(Set(genes))
	return score + penalty * 3
end

logs = ga_run(
	mini_soduku_f,
	(_, _) -> rand(numbers),
	length(numbers)
	;
	population_size = 10000,
	termination=terminate(20, 0, 100),
	compare_fitness = <
)

for (i, n) in enumerate(logs[end].best.genes)
	@printf "%3i" n
	mod(i, 3) == 0 && println()
end


## Exploration and Exploitation
## What have we seen in this chapter?
## Acknowledgement
