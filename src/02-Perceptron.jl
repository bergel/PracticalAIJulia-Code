# The Perceptron Model
## Perceptron as a kind of neuron
## Implementing the perceptron
mutable struct Neuron
	weights::Vector{Number}
	bias::Number
    activation_function
    delta
    output
end

Neuron(weights, bias) = Neuron(weights, bias, nothing)
Neuron(weights, bias, activation_function) = Neuron(weights, bias, activation_function, 0, 0)


function feed(n::Neuron, inputs::Vector{T}) where T <: Number
    @assert length(n.weights) == length(inputs) "Inputs and weights should have same size"
    z = sum(map(*, n.weights, inputs)) + n.bias
    return z > 0 ? 1 : 0
end


p = Neuron([1, 2], -2)
feed(p, [5, 2])


## Testing our code
using Test

@testset "Simple perceptron" begin
	p = Neuron([1, 2], -2)
	@test feed(p, [5, 2])  == 1
	@test feed(p, [-5, 2]) == 0
end


## Formulating logical expressions
@testset "AND" begin
    p = Neuron([1, 1], -1.5)
    table = [[0, 0, 0],
			 [0, 1, 0],
			 [1, 0, 0],
			 [1, 1, 1]]
    @test all(row -> feed(p, row[1:end-1]) == last(row), table)
end


@testset "OR" begin
    p = Neuron([1, 1], -0.5)
    table = [[0, 0, 0],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1]]
    @test all(row -> feed(p, row[1:end-1]) == last(row), table)
end


@testset "NOR" begin
    p = Neuron([-1, -1], 0.5)
    table = [[0, 0, 1],
             [0, 1, 0],
             [1, 0, 0],
             [1, 1, 0]]
    @test all(row -> feed(p, row[1:end-1]) == last(row), table)
end


@testset "NOT" begin
    p = Neuron([-1], 0.5)
    table = [[0, 1], [1, 0]]
    @test all(row -> feed(p, row[1:end-1]) == last(row), table)
end


## Error Handling
@testset "Wrong feeding" begin
    p = Neuron([-1], 0.5)
	@test_throws AssertionError feed(p, [1, 2])
end


## Combining perceptrons
function digital_comparator(a::Int64, b::Int64)
    @assert (a in [0, 1]) && (b in [0, 1]) "Each argument must be either 0 or 1"
    and = Neuron([1, 1], -1.5)
    not = Neuron([-1], 0.5)
    nor = Neuron([-1, -1], 0.5)

    not_a = feed(not, [a])
    not_b = feed(not, [b])

    a_less_than_b = feed(and, [not_a, b])
    a_greater_than_b = feed(and, [a, not_b])
    a_equals_to_b = feed(nor, [a_greater_than_b, a_less_than_b])
    return [a_greater_than_b, a_equals_to_b, a_less_than_b]
end


@testset "Digital comparator" begin
    @test digital_comparator(0, 0) == [0, 1, 0]
    @test digital_comparator(0, 1) == [0, 0, 1]
    @test digital_comparator(1, 0) == [1, 0, 0]
    @test digital_comparator(1, 1) == [0, 1, 0]
end


## Training a perceptron
function train!(
	n::Neuron,
	inputs::Vector{T},
	desired_output::K,
    learning_rate = 0.1
) where T <: Number where K <: Number
    output = feed(n, inputs)
    computed_error = desired_output - output
    for (index, input) in enumerate(inputs)
        n.weights[index] += learning_rate * computed_error * input
    end
    n.bias += learning_rate * computed_error
end


p = Neuron([-1, -1], 2)


p = Neuron([-1, -1], 2)
train!(p, [0, 1], 0)
feed(p, [0, 1])


p = Neuron([-1, -1], 2)
feed(p, [0, 1])  # Return 1
for _ in 1:10
    train!(p, [0, 1], 0)
end
feed(p, [0, 1])  # Return 0


@testset "Training OR" begin
    n = Neuron([-1, -1], 2)
    for epoch in 1:40
        train!(n, [0, 0], 0)
        train!(n, [0, 1], 1)
        train!(n, [1, 0], 1)
        train!(n, [1, 1], 1)
    end

    @test feed(n, [0, 0]) == 0
    @test feed(n, [0, 1]) == 1
    @test feed(n, [1, 0]) == 1
    @test feed(n, [1, 1]) == 1
end


@testset "Training NOT" begin
    n = Neuron([-1], 2)
    for epoch in 1:40
        train!(n, [0], 1)
        train!(n, [1], 0)
    end

    @test feed(n, [0]) == 1
    @test feed(n, [1]) == 0
end


## Drawing graphs
import Pkg
Pkg.add("Plots")
using Plots: plot!, scatter, plot
plot([0, 1, 5, -1])


plot([cos(x) for x in 0:0.1:3.1415*10])


using Random
Random.seed!(42)
scatter([(rand(-50:50), rand(-50:50)) for _ in 1:10])


using Random
Random.seed!(42)
scatter([(rand(0:300), rand(-5:5)) for _ in 1:10])
plot!([cos(x) for x in 0:0.1:3.1415*10])


scatter([(1, 5), (2, 4), (3, 2)], color=["red","blue","black"])


## Number of epochs and precision
table = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
precisions = []
for number_of_epochs in 1:40
    n = Neuron([-1, -1], 2)

    # Run `number_of_epochs` epochs
    for epoch in 1:number_of_epochs
        for row in table
            train!(n, row[1:end-1], last(row))
        end
    end

    # Test the perceptron
    # A prediction of 1 means the prediction is correct, else it is 0
    predictions = [feed(n, row[1:end-1]) == last(row) for row in table]
    push!(precisions, sum(predictions) / length(predictions))
end
plot(precisions, xlabel="epochs", ylabel="precision", ylim=(0,1))


## Predicting 2D points
using Random
Random.seed!(42)

some_points = [(rand(-50:50), rand(-50:50)) for _ in 1:500]
f(x) = -2x - 3
color = []
for p in some_points
    if f(p[1]) > p[2]
        push!(color, "red")
    else
        push!(color, "blue")
    end
end

scatter(some_points, color=color)
plot!(f)


Random.seed!(42)
p = Neuron([10, 2], -2)
ITERATIONS = 20

# Training the perceptron
for _ in 1:ITERATIONS
    x, y = (rand(-50:50), rand(-50:50))
    expected_output = f(x) >= y ? 1 : 0
    train!(p, [x, y], expected_output)
end

# Testing the perceptron
test_points = [(rand(-50:50), rand(-50:50)) for _ in 1:2000]
color = []
for (x,y) in some_points
    if feed(p, [x, y]) == 1
        push!(color, "red")
    else
        push!(color, "blue")
    end
end

scatter(some_points, color=color)
plot!(f)


## Measuring the precision
learning = []
f(x) = -2x - 3
for training in 0:10:2000
    Random.seed!(42)
    p = Neuron([1, 2], -1)

    for _ in 1:training
        x, y = (rand(-50:50), rand(-50:50))
        expected_output = f(x) >= y ? 1 : 0
        train!(p, [x, y], expected_output)
    end

    good = 0
    tries = 1000
    for _ in 1:1000
        x, y = (rand(-50:50), rand(-50:50))
        real_output = f(x) >= y ? 1 : 0
        if abs(feed(p, [x, y]) - real_output) < 0.2
            good += 1
        end
    end
    push!(learning, good / tries)
end

plot(learning)


## Historical perspective
## Exercises
## What have we seen?
