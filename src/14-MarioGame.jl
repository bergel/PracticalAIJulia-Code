# The Platform Video Game
## Dependencies
using Roassal
using Random
using Cairo
using Downloads


## Game assets
const CELL_SIZE = 20


using Downloads
global IMAGES_CACHES = Dict{String, Any}()
function initialize_cache()
    if !isempty(IMAGES_CACHES)
        return
    end
	url = "http://raw.githubusercontent.com/bergel/PracticalAIJulia-Code/main/platform_game_assets"
	path = "/tmp"
	function get_resource(name)
		filename = joinpath(path, "$(name).png")
		isfile(filename) && return filename
		Downloads.download("$(url)/$(name).png", filename)
	end
	load_resource(name) = Cairo.read_from_png(get_resource(name))

    IMAGES_CACHES["hero"] = load_resource("hero")
    IMAGES_CACHES["monster"] = load_resource("monster")
    IMAGES_CACHES["ground"] = load_resource("ground")
    IMAGES_CACHES["tube"] = load_resource("tube")
    IMAGES_CACHES["end"] = load_resource("end")
end
initialize_cache()


## Monster definition
mutable struct MNMonster
    world
    is_going_left::Bool
    roassal_shape
    remain_on_platform::Bool
	shapes_nearby_cache

    function MNMonster(world)
        roassal_shape = RImage(
			IMAGES_CACHES["monster"];
			width = CELL_SIZE,
			height = CELL_SIZE,
			model=:monster
		)

        # Very simple heuristics to decide if the monster should remain on the platform or
        # not. It makes the game more fun if some of them are falling.
        should_remain_on_platform = isodd(length(world.monsters))
        return new(world, true, roassal_shape, should_remain_on_platform, nothing)
    end
end


get_position(character) = pos(character.roassal_shape)
get_position_y(character) = get_position(character)[2]
get_position_x(character) = get_position(character)[1]


function is_far_from_hero(monster::MNMonster)
	return abs(get_position_x(monster) - get_position_x(monster.world.hero)) > 200
end


function beat!(character::MNMonster)
    # Is the character too far from the hero? If yes, do nothing
    is_far_from_hero(character) && return

    # Check if the character is falling
    is_position_empty(character, (0, 1)) && move!(character, (0, 1))

    if character.remain_on_platform
         if character.is_going_left
            if is_position_empty(character, (-1, 0)) && !is_position_empty(character, (-1, 1))
                move!(character, (-1, 0))
            else
                character.is_going_left = false
            end
        else
            if is_position_empty(character, (1, 0))  && !is_position_empty(character, (1, 1))
                move!(character, (1, 0))
            else
                character.is_going_left = true
            end
        end
        return
    end

    if character.is_going_left
        if is_position_empty(character, (-1, 0))
            move!(character, (-1, 0))
        else
            character.is_going_left = false
        end
    else
        if is_position_empty(character, (1, 0))
            move!(character, (1, 0))
        else
            character.is_going_left = true
        end
    end
end


# Return true if the character moved, false otherwise
function move!(character, delta::Tuple{Int,Int})
    is_empty = is_position_empty(character, delta)
    is_empty && translate_by!(character.roassal_shape, delta)
	character.shapes_nearby_cache = nothing
    return is_empty
end


## Hero definition
abstract type MNAIAgent end


mutable struct MNHero
    world
    jump_counter::Int64
    is_falling::Bool
    is_jumping::Bool
    roassal_shape
    ai_agent::Union{Nothing, MNAIAgent}
	shapes_nearby_cache

    function MNHero(ai_agent=nothing)
        roassal_shape = RImage(
			IMAGES_CACHES["hero"];
			width = CELL_SIZE,
			height = CELL_SIZE,
			model=:hero
		)
        return new(nothing, 0, false, false, roassal_shape, ai_agent, nothing)
    end
end


function jump(character::MNHero)
    if !character.is_jumping && !character.is_falling
        character.is_jumping = true
        character.jump_counter = 0
    end
end


function beat!(character::MNHero)
	# Use the AI if an agent is set
    if !isnothing(character.ai_agent)
	    character.world.abstract_view= abstract_view_game(character.world)
        actions = ask_ai(character.ai_agent, character.world.abstract_view)
        (:move_right in actions) && move!(character, (1,0))
        (:move_left in actions)  && move!(character, (-1,0))
        (:jump in actions)       && jump(character)
    end

    if character.is_jumping
        if is_position_empty(character, (0, -1))
            move!(character, (0, -1))
        else
            character.is_jumping = false
            character.is_falling = true
        end
        character.jump_counter += 1
        if character.jump_counter == 130
            character.is_jumping = false
            character.is_falling = true
        end
    else
        character.is_falling = move!(character, (0, 1))
    end

	# If the hero reaches the goal
    if get_position_x(character) == character.world.goal_position[1] * CELL_SIZE
        print_in_world(character.world, "You won!")
		character.world.has_won = true
        game_over(character.world)
    end

	# Has the character hit a monster?
    for monster in character.world.monsters
        if is_intersecting(character.roassal_shape, monster.roassal_shape)
            print_in_world(character.world, "Game Over!")
            game_over(character.world)
        end
    end
end


function print_in_world(world, string::String)
	txt = RText(string)
	add!(world.canvas, txt)
	# Move the text just above the hero
	translate_to!(txt, get_position(world.hero) .- (0, 30))
end


## Modelling the World
mutable struct MNWorld
    hero::MNHero
    canvas::RCanvas
    monsters::Vector{MNMonster}
    is_game_running::Bool
    seed::Int64
    abstract_view::Matrix{Int8}
    goal_position::Tuple{Int64, Int64}
    monsters_count::Int64
    platforms_count::Int64
    tubes_count::Int64
	has_won::Bool

    function MNWorld(;
		ai_agent=nothing, monsters_count=10, platforms_count=10,
		tubes_count=5, width=100, height=20, seed=42
	)
        monsters = Vector{MNMonster}()
        hero = MNHero(ai_agent)
		canvas = RCanvas("Mini Platform"; background_color = :ciel)

        new_world = new(hero, canvas, monsters, true, seed, zeros(Int8, 20, 20), (0,0),
			monsters_count, platforms_count, tubes_count, false
		)
        generate_map!(new_world, width, height)

        # Hero setup
        hero.world = new_world
        translate_to!(hero.roassal_shape, (3 * CELL_SIZE, 2 * CELL_SIZE))
        add!(new_world.canvas, hero.roassal_shape)
        center_on_hero(new_world)

        return new_world
    end
end


function center_on_hero(world::MNWorld)
    center_on_shape!(world.canvas, world.hero.roassal_shape)
end


function game_over(game::MNWorld)
    game.is_game_running = false
end


function add_brick!(world::MNWorld, position::Tuple{Int64, Int64})
    box = RImage(
		IMAGES_CACHES["ground"];
		width = CELL_SIZE,
		height = CELL_SIZE,
		model=:obstacle
	)
    add!(world.canvas, box)
    translate_to!(box, (position[1] * CELL_SIZE, position[2] * CELL_SIZE))
end


function add_platform!(world::MNWorld, start_pos::Tuple{Int64, Int64}, length::Int64=4)
    x_start, y = start_pos
    for x in x_start:(x_start + length - 1)
        add_brick!(world, (x, y))
    end
end


function add_tube!(world::MNWorld, position::Tuple{Int64,Int64}, height::Int64=2)
    x, y_start = position
    for y in (y_start - height):y_start
        box = RImage(
			IMAGES_CACHES["tube"];
			width = CELL_SIZE + 1,
			height = CELL_SIZE + 1,
			model=:obstacle
		)

        add!(world.canvas, box)
        translate_to!(box, (x * CELL_SIZE, y * CELL_SIZE))
    end

    box = RImage(IMAGES_CACHES["tube"];
		width = CELL_SIZE,
		height = CELL_SIZE,
		model=:obstacle
	)
    add!(world.canvas, box)
    translate_to!(box, ((x-1) * CELL_SIZE, (y_start - height) * CELL_SIZE))

    box = RImage(
		IMAGES_CACHES["tube"];
		width = CELL_SIZE,
		height = CELL_SIZE,
		model=:obstacle
	)
    add!(world.canvas, box)
    translate_to!(box, ((x+1) * CELL_SIZE, (y_start - height) * CELL_SIZE))
end


function is_position_empty(character, delta::Tuple{Int64, Int64})
    w = character.roassal_shape.width
    h = character.roassal_shape.height
    return is_position_empty(
        character,
        get_position(character),
        delta,
        w,
        h
    )
end


function is_position_empty(
    character,
    position::Tuple{Int64, Int64},
    delta::Tuple{Int64, Int64},
    width::Int64,
    height::Int64
)
	if isnothing(character.shapes_nearby_cache)
		character.shapes_nearby_cache = shapes_nearby(character.roassal_shape, CELL_SIZE*2)
	end

	roassal_shape = character.roassal_shape

    for s in character.shapes_nearby_cache
        s === roassal_shape && continue
        s.model != :obstacle && continue

        if is_intersecting((position[1] + delta[1], position[2] + delta[2], width-2, height-2),
                           (pos(s)[1], pos(s)[2], s.width, s.height))
            return false
        end
    end
    return true
end


# width and height are in number of bricks
function generate_map!(world::MNWorld, width::Int64, height::Int64)
    rng = Xoshiro(world.seed)
    # Adding platforms
    for _ in 1:world.platforms_count
        x = rand(rng, 2:width)
        y = rand(rng, 3:height)
        add_platform!(world, (x, y), rand(rng, 2:5))
    end
    # Adding tubes
    for _ in 1:world.tubes_count
        x = rand(rng, 5:width)
        y = height - 1
        add_tube!(world, (x, y), rand(rng, 2:5))
    end

    # Walls and ceiling
    for y in 0:height
        add_brick!(world, (1, y))
        add_brick!(world, (width + 1, y))
    end
    for x in 1:(width + 1)
        add_brick!(world, (x, 0))
        add_brick!(world, (x, height))
    end

    # Adding monsters
    for _ in 1:world.monsters_count
        monster = MNMonster(world)
        translate_to!(monster.roassal_shape, (rand(rng, 10:width) * 20, 32))
        add!(world.canvas, monster.roassal_shape)
        push!(world.monsters, monster)
    end

    # Finish line
    world.goal_position = (width-2, height-1)
    for y in 1:(height-1)
		goal_shape = RImage(
			IMAGES_CACHES["end"];
			width = CELL_SIZE,
			height = CELL_SIZE,
			model = :goal
		)
		translate_to!(goal_shape, (world.goal_position[1] * CELL_SIZE, y * CELL_SIZE))
		add!(world.canvas, goal_shape)
    end
end


function beat!(world::MNWorld)
    world.is_game_running || return

    beat!(world.hero)
    for monster in world.monsters
        beat!(monster)
    end

    center_on_hero(world)
    refresh(world.canvas)
end


## Abstracting the game representation
function abstract_view_game(world::MNWorld)
    shapes = shapes_nearby(world.hero.roassal_shape, 200)
    result = zeros(Int8, 20, 20)
    hero_pos_abstract_map = pos_in_abstract_map(get_position(world.hero))
    for shape in vcat(shapes, [world.hero.roassal_shape])
        x_abstract_map_o, y_abstract_map_o = pos_in_abstract_map(pos(shape))
        x_abstract_map = x_abstract_map_o - hero_pos_abstract_map[1] + 10
        y_abstract_map = y_abstract_map_o - hero_pos_abstract_map[2] + 10
        if x_abstract_map>=1 && x_abstract_map<=20 && y_abstract_map>=1 && y_abstract_map<=20
            if shape.model == :hero
                result[y_abstract_map, x_abstract_map] = 4
            elseif shape.model == :goal
                result[y_abstract_map, x_abstract_map] = 3
            elseif shape.model == :monster
                result[y_abstract_map, x_abstract_map] = 2
            elseif shape.model == :obstacle
                result[y_abstract_map, x_abstract_map] = 1
            end
        end
    end
    return result
end


function pos_in_abstract_map(pos_in_world::Tuple)
    x_in_abstract_map = convert(Int, round(pos_in_world[1] / CELL_SIZE) + 1)
    y_in_abstract_map = convert(Int, round(pos_in_world[2] / CELL_SIZE) + 1)
    return (x_in_abstract_map, y_in_abstract_map)
end


## Playing the Platform game
function open_game(world::MNWorld)
    delta_x = Threads.Atomic{Int}(0)
    delta_y = Threads.Atomic{Int}(0)

    add_key_right_callback!(world.canvas,
        (event, canvas) -> begin delta_x[] = 1 end,
        (event, canvas) -> begin delta_x[] = 0 end
    )
    add_key_left_callback!(world.canvas,
        (event, canvas) -> begin delta_x[] = -1 end,
        (event, canvas) -> begin delta_x[] = 0 end
    )
    add_key_up_callback!(world.canvas,
        (event, canvas) -> begin delta_y[] = -1 end,
        (event, canvas) -> begin delta_y[] = 0 end
    )
    s = Base.Semaphore(1)
    function refreshing(animation)
	    world.is_game_running || return
        Base.acquire(s) do
            if delta_x[] == -1
                move!(world.hero, (-1,0))
            elseif delta_x[] == 1
                move!(world.hero, (1,0))
            end
            if delta_y[] == -1
                jump(world.hero)
            end

			beat!(world)
        end
    end

    add!(world.canvas, Roassal.Animation(refreshing, 1000))
    rshow(world.canvas; center = false, size = (20*CELL_SIZE, 20*CELL_SIZE))
end


world = MNWorld(; monsters_count=5, platforms_count=60, tubes_count=5, seed=420)
open_game(world)


## Plugging an agent
struct MNAIAgent_MoveRight <: MNAIAgent end

function ask_ai(::MNAIAgent_MoveRight, abstract_view)
    return [:move_right]
end


## What have we seen in this chapter?
