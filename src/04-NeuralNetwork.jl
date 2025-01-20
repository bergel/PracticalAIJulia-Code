# Neural Networks
## General architecture
## Neural layer
mutable struct Layer
    previous_layer::Union{Layer, Nothing}
    next_layer::Union{Layer, Nothing}
    neurons::Vector{Neuron}
end


function Layer(neurons_count::Integer, weights_count::Integer)
    neurons = []
    for _ in 1:neurons_count
        # Generate weights_count random numbers ranging from -2 to 2
        weights = rand(Float64, weights_count) .* 4 .- 2
		# Generate a bias, ranging from -2 to 2
        bias = rand(Float64) * 4 - 2
        push!(neurons, Neuron(weights, bias, SigmoidAF()))
    end
    return Layer(nothing, nothing, neurons)
end


function Layer(neurons_count::Integer, weights_count::Integer, next_layer::Layer)
    new_layer = Layer(neurons_count, weights_count)
    new_layer.next_layer = next_layer
    next_layer.previous_layer = new_layer
    return new_layer
end


function feed(layer::Layer, input_values::Vector{T}) where T <: Number
    current_output = map(n->feed(n, input_values), layer.neurons)
    if isoutput(layer)
        return current_output
    else
        return feed(layer.next_layer, current_output)
    end
end

isoutput(layer::Layer) = isnothing(layer.next_layer)


function output_layer(layer::Layer)
    return isoutput(layer) ? layer : output_layer(layer.next_layer)
end


@testset "Output layer" begin
    Random.seed!(42)
    nl = Layer(3, 4)
    @test isoutput(nl)
    @test output_layer(nl) == nl
    @test length(nl.neurons) == 3
    for n in nl.neurons
        @test length(n.weights) == 4
    end

    output = feed(nl, [1, 2, 3, 4])
    @test length(output) == 3
end


@testset "Layered chain" begin
    Random.seed!(42)
    nl = Layer(3, 4, Layer(4, 3))
    @test !isoutput(nl)
    @test isoutput(nl.next_layer)
    @test output_layer(nl) == nl.next_layer

    output = feed(nl, [1, 2, 3, 4])
    @test length(output) == 4
    expected = [0.6784327847793552, 0.7200854527228148, 0.7255330924454609, 0.16459558102999708]
    for (e, o) in zip(expected, output)
        @test isclose(e, o)
    end
end


## Modeling a neural network
mutable struct NNetwork
    root_layer::Layer
    errors::Vector{Number}
    precisions::Vector{Number}
    NNetwork(root_layer) = new(root_layer, [], [])
end


function feed(nn::NNetwork, input_values::Vector{T}) where T <: Number
    return feed(nn.root_layer, input_values)
end


function NNetwork(inputs_count::Int, hidden_count::Int, outputs_count::Int)
    output_layer = Layer(outputs_count, hidden_count)
    inner_layer = Layer(hidden_count, inputs_count, output_layer)
    return NNetwork(inner_layer)
end


function NNetwork(inputs_count::Int, hidden_count1::Int, hidden_count2::Int, outputs_count::Int)
    output_layer = Layer(outputs_count, hidden_count2)
    inner_layer2 = Layer(hidden_count2, hidden_count1, output_layer)
    inner_layer1 = Layer(hidden_count1, inputs_count, inner_layer2)
    return NNetwork(inner_layer1)
end


output_layer(nn::NNetwork) = output_layer(nn.root_layer)
outputs_count(nn::NNetwork) = length(output_layer(nn).neurons)


@testset "Simple network evaluation" begin
    Random.seed!(42)
    n = NNetwork(2, 2, 1)
    @test outputs_count(n) == 1

    output_as_array = feed(n, [1, 3])
    output = first(output_as_array)
    @test isclose(output, 0.538)
end


## Backpropagation
### Step 1: forward feeding
function feed(n::Neuron, inputs::Vector{T}) where T <: Number
    @assert length(n.weights) == length(inputs) "Inputs and weights should have same size"
    z = sum(map(*, n.weights, inputs)) + n.bias
    n.output = eval(n.activation_function, z)
    return n.output
end


### Step 2: error backward propagation
function backward_propagate_error(
    nn::NNetwork,
    expected_outputs::Vector{T}
) where T <: Number
    return backward_propagate_error(output_layer(nn), expected_outputs)
end


# Propagate the error computed from the expected values
function backward_propagate_error(nl::Layer, expected::Vector{T}) where T <: Number
    # This is a recursive function.
    # The back propagation begins with the output layer (i.e., the last layer)
    for (neuron, exp) in zip(nl.neurons, expected)
        the_error = exp - neuron.output
        adjust_delta_with(neuron, the_error)
    end
    # no iteration if we are at the input layer, i.e., if there is no
    # previous layer.
    isnothing(nl.previous_layer) && return

    # iterate
    backward_propagate_error(nl.previous_layer)
end


# Propagate the error previously computed
function backward_propagate_error(nl::Layer)
    # Recursive function. The back propagation begins with the output layer.
    for (i, neuron) in enumerate(nl.neurons)
        the_error = 0.0
        for next_neuron in nl.next_layer.neurons
            the_error += next_neuron.weights[i] * next_neuron.delta
        end
        adjust_delta_with(neuron, the_error)
    end
    isnothing(nl.previous_layer) && return
    backward_propagate_error(nl.previous_layer)
end


function adjust_delta_with(n::Neuron, the_error)
    n.delta = the_error * derivative(n.activation_function, n.output)
end


### Step 3: updating neurons parameters
function update_weights(network::NNetwork, initial_inputs)
    update_weights(network.root_layer, initial_inputs)
end


function update_weights(layer::Layer, inputs)
    # Update the weights of the neuron based on the set of initial input.
    # Assume that layer is the first layer of a network
    for n in layer.neurons
        adjust_weight_with_inputs(n, inputs)
        adjust_bias(n)
    end
    isnothing(layer.next_layer) && return
    update_weights(layer.next_layer)
end


function update_weights(layer::Layer)
    # Update the weights of the neuron based on the set of initial input.
    # layer is not a root layer
    inputs = map(n->n.output, layer.previous_layer.neurons)
    update_weights(layer, inputs)
end


function adjust_weight_with_inputs(neuron::Neuron, inputs, learning_rate=0.1)
    for (index, an_input) in enumerate(inputs)
        neuron.weights[index] += learning_rate * neuron.delta * an_input
    end
end


function adjust_bias(neuron::Neuron, learning_rate=0.1)
    neuron.bias += learning_rate * neuron.delta
end


function train!(network::NNetwork, inputs, desired_output)
    feed(network, inputs)
    backward_propagate_error(network, desired_output)
    update_weights(network, inputs)
end


@testset "Learning XOR" begin
    n = NNetwork(2, 3, 1)
    for _ in 1:20_000
        train!(n, [0, 0], [0])
        train!(n, [0, 1], [1])
        train!(n, [1, 0], [1])
        train!(n, [1, 1], [0])
    end
    @test first(feed(n, [0, 0])) < 0.1
    @test first(feed(n, [0, 1])) > 0.9
    @test first(feed(n, [1, 0])) > 0.9
    @test first(feed(n, [1, 1])) < 0.1
end


## What have we seen in this chapter?
