# Neuroevolution On The Platform Game
## Building an artificial player
SEED = 420
MONSTER_COUNT = 15
PLATFORM_COUNT = 60
TUBES_COUNT = 5


struct MNAIAgent_NEAT <: MNAIAgent
	ind::NEIndividual
end


function ask_ai(ai_agent::MNAIAgent_NEAT, abstract_view::Matrix{Int8})
	linear_values = reshape(copy(abstract_view), (400, 1))
	output = evaluate(ai_agent.ind, linear_values)
	actions = []
	output[1] > 0.2 && push!(actions, :move_right)
	output[2] > 0.2 && push!(actions, :move_left)
	output[3] > 0.2 && push!(actions, :jump)

	return actions
end


function game_fitness(ind::NEIndividual)
	world = MNWorld(;
		ai_agent = MNAIAgent_NEAT(ind),
		monsters_count=MONSTER_COUNT,
		platforms_count=PLATFORM_COUNT,
		tubes_count=TUBES_COUNT,
		seed=SEED
	)
	for _ in 1:2000
		beat!(world)
	end

	world.has_won && return 3000

	# Return the X value of the position as the fitness
	return get_position(world.hero)[1]
end


Random.seed!(42)
neat = NEAT(400, 3, game_fitness ;
	population_size=100,
	generations_max=100,
	termination_function=(best_fit -> best_fit == 3000)
)
run(neat)


using Serialization
Serialization.serialize("mario.bin", best_individual(neat))


ind = Serialization.deserialize("mario.bin")
world = MNWorld(;
	ai_agent = MNAIAgent_NEAT(copy(ind)),
	monsters_count=MONSTER_COUNT,
	platforms_count=PLATFORM_COUNT,
	tubes_count=TUBES_COUNT,
	seed=SEED
)
open_game(world)


## Memory less game
## What have we seen in this chapter?
