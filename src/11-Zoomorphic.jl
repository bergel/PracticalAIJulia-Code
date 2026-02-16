# Building Walking Creatures
using Roassal


## Modeling join points
mutable struct CNode
	force
	speed_vector::Tuple{Float64, Float64}
	is_on_platform::Bool
	shape

	CNode() = CNode(:gray)
	CNode(color::Symbol) = new((0.0, 0.0), (0.0, 0.0), false, RCircle(; color=color))
end


function Base.show(io::IO, n::CNode)
    print(io, "CNode{ speed_vector: $(n.speed_vector), is_on_platform: $(n.is_on_platform), force: $(n.force) }")
end


get_pos(node::CNode) = pos(node.shape)


function add_force!(node::CNode, force)
	node.force = node.force .+ force
end


const GRAVITY_FORCE = (0.0, 0.3)
const FRICTION_FACTOR = 0.9
const FRICTION_FACTOR_PLATFORM = 0.3


# Apply physical rules
function beat!(node::CNode)
	node.speed_vector = (node.speed_vector .+ GRAVITY_FORCE .+ node.force) .* FRICTION_FACTOR
	if node.is_on_platform
		node.speed_vector = (node.speed_vector[1] * FRICTION_FACTOR_PLATFORM , node.speed_vector[2])
	end
	translate_by!(node.shape, node.speed_vector) # Roassal function
end


# Check if the node is on a platform. If yes, then node.is_on_platform is set to true
function check_for_collision(node::CNode, platforms)
	node.is_on_platform = false
	for p in platforms
		if does_collide(node, p)
			node.speed_vector = (node.speed_vector[1], 0.0)
			adjust_node_if_necessary(node, p)
			node.is_on_platform = true
			return
		end
	end
end


## Modeling platforms
mutable struct CPlatform
	width
	height
	shape

	CPlatform() = CPlatform(80, 10)
	CPlatform(w) = CPlatform(w, 10)
	CPlatform(w, h) = new(w, h, RBox(; width = w, height = h))
end


function Base.show(io::IO, p::CPlatform)
    print(io, "CPlatform{ pos: $(pos(p.shape)), height: $(p.height) }")
end


translate_to(p::CPlatform, position) = translate_to!(p.shape, position)


# Does node collide with platform?
function does_collide(node::CNode, platform::CPlatform)
	return is_intersecting(node.shape, platform.shape)
end


function adjust_node_if_necessary(node::CNode, platform::CPlatform)
	bottom_node = bottom_center(node.shape)[2] 	 # Y-component of the center at the bottom
	top_platform = top_center(platform.shape)[2] # Y-component of the top center

	if top_platform < bottom_node
		translate_by!(node.shape, (0, (top_platform - bottom_node)))
	end
end


## Modeling Muscle
mutable struct CMuscle
	time
	time1
	time2
	length1
	length2
	strength
	shape
	node1::CNode
	node2::CNode
end


function beat!(muscle::CMuscle)
	muscle.time += 1
	if muscle.time == max_time(muscle)
		muscle.time = 0
	end
end


function muscle_length(muscle::CMuscle)
	return muscle.time < min_time(muscle) ? muscle.length1 : muscle.length2
end


function min_time(muscle::CMuscle)
	return min(muscle.time1, muscle.time2)
end


max_time(muscle::CMuscle) = max(muscle.time1, muscle.time2)


## Generating muscles
struct CMuscleGenerator
	min_strength
	delta_strength
	min_length
	delta_length
	min_time
	delta_time

	CMuscleGenerator() = CMuscleGenerator(0.01, 0.5, 10, 80, 10, 100)
	CMuscleGenerator(a, b, c, d, e, f) = new(a, b, c, d, e, f)
end


generate_length(mg::CMuscleGenerator) = mg.min_length + rand(1:mg.delta_length)


generate_strength(mg::CMuscleGenerator) = mg.min_strength + rand() * mg.delta_strength


generate_time(mg::CMuscleGenerator) = mg.min_time + rand(1:mg.delta_time)


function build_muscle(
	muscle_generator::CMuscleGenerator,
	node1::CNode,
	node2::CNode,
	color::Symbol=:gray
)
	m = CMuscle(
		0,
		generate_time(muscle_generator),
		generate_time(muscle_generator),
		generate_length(muscle_generator),
		generate_length(muscle_generator),
		generate_strength(muscle_generator),
		RLine(node1.shape, node2.shape; color),
		node1,
		node2
	)
end


function serialize(muscle::CMuscle)
	return [
		muscle.length1,
		muscle.length2,
		muscle.strength,
		muscle.time1,
		muscle.time2
	]
end


function materialize!(muscle::CMuscle, values)
	muscle.length1 = values[1]
	muscle.length2 = values[2]
	muscle.strength = values[3]
	muscle.time1 = values[4]
	muscle.time2 = values[5]
end


function value_for_index(muscle_generator, index)
	i = mod(index - 1, 5)
	(i == 0 || i == 1) && return generate_length(muscle_generator)
	i == 2 && return generate_strength(muscle_generator)
	(i == 3 || i == 4) && return generate_time(muscle_generator)
end


## Modeling creatures
mutable struct CCreature
	nodes::Vector{CNode}
	muscles::Vector{CMuscle}
	color

	CCreature() = CCreature(:gray)
	CCreature(color::Symbol) = new([], [], color)
end


function Base.show(io::IO, c::CCreature)
    print(io, "CCreature{ $(length(c.nodes)) nodes, $(length(c.muscles)) muscles }")
end


function add_muscle!(creature::CCreature, node1::CNode, node2::CNode, mg::CMuscleGenerator)
	push!(creature.muscles, build_muscle(mg, node1, node2))
end


muscle_count(creature::CCreature) = length(creature.muscles)


function beat!(creature::CCreature)
	for n in creature.nodes
		beat!(n)
	end

	for m in creature.muscles
		beat!(m)
	end
	do_physic!(creature)
end


function check_for_collision(creature::CCreature, platforms)
	for n in creature.nodes
		check_for_collision(n, platforms)
	end
end


## Creating creatures
function add_nodes!(creature::CCreature, nodes_count::Int, color::Symbol=:gray)
	for _ in 1:nodes_count
		push!(creature.nodes, CNode(color))
	end
end


function CCreature(
	nodes_count::Int,
	color::Symbol=:gray,
	muscle_generator::CMuscleGenerator=CMuscleGenerator()
)
	creature = CCreature(color)
	add_nodes!(creature, nodes_count, color)
	existing_muscles = []
	for n1 in creature.nodes
		for n2 in creature.nodes
			n1 === n2 && continue
			if !((n1 => n2) in existing_muscles)
				add_muscle!(creature, n1, n2, muscle_generator)
				push!(existing_muscles, n1 -> n2)
				push!(existing_muscles, n2 -> n1)
			end
		end
	end
	locate_nodes!(creature)
	return creature
end


function do_physic!(creature::CCreature)
	for n in creature.nodes
		n.force = (0, 0)
	end
	r(delta) = sqrt(delta[1] * delta[1] + delta[2] * delta[2])
	for m in creature.muscles
		n1 = m.node1
		n2 = m.node2
		delta = get_pos(n2) .- get_pos(n1)
		actual_length = max(1, r(delta))

		unit = delta ./ actual_length
		force = 0.1 * m.strength * (actual_length - muscle_length(m)) .* unit
		add_force!(n1, force)
		add_force!(n2, 0 .- force)
	end
end


### Serialization and materialization of a creature
serialize(creature::CCreature) = vcat([serialize(m) for m in creature.muscles]...)


function materialize!(creature::CCreature, values)
	values_per_muscle = [collect(s) for s in Iterators.partition(values, 5)]
	for (muscle, values) in zip(creature.muscles, values_per_muscle)
		materialize!(muscle, values)
	end
end


### Accessors and utility functions
function get_pos(creature::CCreature)
	return reduce(.+, map(get_pos, creature.nodes)) ./ length(creature.nodes)
end


function locate_nodes!(creature::CCreature)
	for (i,n) in enumerate(creature.nodes)
		translate_by!(n.shape, (i, i))
	end
end


function translate_to(creature::CCreature, point)
	average_center = get_pos(creature)
	delta = point .- average_center
	for n in creature.nodes
		translate_by!(n.shape, delta)
	end
end


## Defining the World
mutable struct CWorld
	creatures
	platforms
	time
	canvas
	ground_length
end

function CWorld()
	# 5000 corresponds to size of the world
	w = CWorld([], [], 0, [], 5000)
	w.canvas = RCanvas()
	add_ground!(w)
	return w
end


function add_ground!(world::CWorld)
	ground_platform = CPlatform(world.ground_length + 500)
	add_platform!(world, ground_platform)
	translate_to!(ground_platform.shape, (world.ground_length/2, 100))
end


function Base.show(io::IO, w::CWorld)
    print(io, "CWorld{ $(length(w.creatures)) creatures, $(length(w.platforms)) platforms, $(w.time) time }")
end


function add_creature!(world::CWorld, creature::CCreature)
	push!(world.creatures, creature)
	for n in creature.nodes
		add!(world.canvas, n.shape)
	end
	for m in creature.muscles
		add!(world.canvas, m.shape)
	end
	push_lines_back(world.canvas)
	return world
end


function add_platform!(world::CWorld, platform)
	push!(world.platforms, platform)
	add!(world.canvas, platform.shape)
end


function beat!(world::CWorld)
	world.time += 1
	for c in world.creatures
		beat!(c)
		check_for_collision(c, world.platforms)
	end
	refresh(w.canvas)
end


function add_pylons!(world::CWorld)
	for flag_position in 0:100:world.ground_length
		pylon = RBox(; color=:green, width=3, height=100)
		add!(world.canvas, pylon)
		translate_to!(pylon, flag_position, 50)
	end
end


function open_world(world::CWorld)
	add_pylons!(world)

	function animate!(a)
		beat!(world)
		if !isempty(world.creatures)
			p = get_pos(world.creatures[1])
			translate_to!(world.canvas, (-p[1], 0))
		end
		sleep(0.05)
	end

	add!(world.canvas, Roassal.Animation((a)->animate!(a), 180))
	rshow(world.canvas; center=false)
end


## Cold run
w = CWorld()
add_creature!(w, CCreature(10, :red))
open_world(w)


red_creature = CCreature(10, :red)
green_creature = CCreature(10, :green)
yellow_creature = CCreature(10, :yellow)

w = CWorld()
add_creature!(w, red_creature)
add_creature!(w, green_creature)
add_creature!(w, yellow_creature)

translate_to(green_creature, (100, -50))
translate_to(yellow_creature, (200, -50))
open_world(w)


## What have we seen in this chapter?
