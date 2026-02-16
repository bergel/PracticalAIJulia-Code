# Evolving Walking Creature
## Straight application of genetic algorithm
# This function will be redefined along this chapter
# Return the distance walked by a creature with the provided genes.
function open_race(logs)
	red_creature = CCreature(nodes_count, :red)
	yellow_creature = CCreature(nodes_count, :yellow)
	green_creature = CCreature(nodes_count, :green)
	generation_count = length(logs)

	# 1/2 of the evolution
	materialize!(yellow_creature, logs[generation_count >> 1].best.genes)

	# 3/4 of the evolution
	three_quarter = convert(Int, floor(length(logs) * 3 / 4))
	materialize!(green_creature, logs[three_quarter].best.genes)

	# Result of the evolution
	materialize!(green_creature, logs[end].best.genes)

	w = build_world()
	add_creature!(w, green_creature)
	add_creature!(w, yellow_creature)
	add_creature!(w, red_creature)

	open_world(w)
end


## Take inspiration from Nature to address the competing conventions problem
## Constrained crossover operation
struct GAConstrainedCrossoverOperation <: GAAbstractCrossoverOperation
	cutpoints::Vector{Int}
end


# Return a new individual, result of crossover between `i1` and `i2`
function crossover(op::GAConstrainedCrossoverOperation, i1::Individual, i2::Individual)
	return crossover(op, i1, i2, rand(op.cutpoints))
end


## Moving forward
# Return the distance walked by a creature with the provided genes.
## Serializing the muscle attributes
green_creature = CCreature(3, :green)
materialize!(green_creature,
	[85, 86, 0.02361862608786352, 110, 91, 46, 41, 0.30997012932762247, 11, 99, 90, 63,
	0.40216495744506836, 20, 11, 39, 61, 0.011287745017745423, 45, 34, 46, 46,
	0.4031165394837463, 46, 41, 75, 39, 0.4609650353077659, 31, 40]
)

w = CWorld()
add_creature!(w, green_creature)
open_world(w)


## Climbing stairs
## What have we seen in this chapter?
