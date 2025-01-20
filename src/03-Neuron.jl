# Sigmoid Neuron
## Limit of a single perceptron
## Activation function
stepf(x) = x > 0 ? 1 : 0
plot(stepf, -10:0.01:10, label="StepF")


## The sigmoid neuron
sigmoid(x) = 1 / (1 + (exp(-x)))
plot(sigmoid, -10:0.01:10, label="SigmoidF")


## Implementing the activation functions
abstract type activation_function end


struct SigmoidAF <: activation_function end
eval(::SigmoidAF, z) = 1 / (1 + exp(-z))
derivative(::SigmoidAF, output) = output * (1 - output)


struct StepAF <: activation_function end
eval(::StepAF, z) = z > 0 ? 1 : 0
derivative(::StepAF, output) = 1


## Extending the neuron with the activation functions
function feed(n::Neuron, inputs::Vector{T}) where T <: Number
    @assert length(n.weights) == length(inputs) "Inputs and weights should have same size"
    z = sum(map(*, n.weights, inputs)) + n.bias
    return eval(n.activation_function, z)
end


function train!(
	n::Neuron,
	inputs::Vector{T},
	desired_output::K,
	learning_rate = 0.1
) where T <: Number where K <: Number
    output = feed(n, inputs)
    diff = desired_output - output
    delta = diff * derivative(n.activation_function, output)
    for (index, input) in enumerate(inputs)
        n.weights[index] += learning_rate * delta * input
    end
    n.bias += learning_rate * delta
end


function sigmoid!(n::Neuron)
    n.activation_function = SigmoidAF()
end

function step!(n::Neuron)
    n.activation_function = StepAF()
end


## Testing the sigmoid neuron
isclose(x, y) = isapprox(x, y; atol = 0.05)

@testset "Long OR sigmoid" begin
    n = Neuron([-1, -1], 2)
    sigmoid!(n)
    for _ in 1:20000
        train!(n, [0, 0], 0)
        train!(n, [0, 1], 1)
        train!(n, [1, 0], 1)
        train!(n, [1, 1], 1)
    end

    @test isclose(feed(n, [0, 0]), 0)
    @test isclose(feed(n, [0, 1]), 1)
    @test isclose(feed(n, [1, 0]), 1)
    @test isclose(feed(n, [1, 1]), 1)
end


function train!(n::Neuron, table::Vector, learning_rate = 0.1)
	for _ in 1:20000
		for row in table
			train!(n, row[1:end-1], row[end])
		end
    end
end
function train_and_test(n::Neuron, table::Vector)
	train!(n, table)
	return all(row -> isclose(feed(n, row[1:end-1]), row[end]), table)
end

@testset "OR sigmoid" begin
    n = Neuron([1, 1], 2)
    sigmoid!(n)
    table = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    @test train_and_test(n, table)
end


@testset "Training AND Sigmoid" begin
    n = Neuron([-1, -1], 2)
    sigmoid!(n)
    table = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
    @test train_and_test(n, table)
end

@testset "Training NOT Sigmoid" begin
    n = Neuron([-1], 2)
    sigmoid!(n)
	table = [[0, 1], [1, 0]]
    @test train_and_test(n, table)
end


## Slower to learn
table = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
result_sig = []
result_stp = []
for iteration in 1:1000
    c(x, y) = isapprox(x, y; atol = 0.1)
    n_sig = Neuron([-1, -1], 2)
    sigmoid!(n_sig)

    n_stp = Neuron([-1, -1], 2)
    step!(n_stp)

    # We run `iteration` epochs
    for _ in 1:iteration
        for row in table
            train!(n_sig, row[1:end-1], last(row))
            train!(n_stp, row[1:end-1], last(row))
        end
    end

    # We test
    test_result_sig = [c(feed(n_sig, row[1:end-1]), last(row)) for row in table]
    push!(result_sig, sum(filter(v->v, test_result_sig)) / length(table))

	test_result_stp = [c(feed(n_stp, row[1:end-1]), last(row)) for row in table]
    push!(result_stp, sum(filter(v->v, test_result_stp)) / length(table))
end
plot(label = "Sigmoid", result_sig, xlabel="epochs", ylabel="precision", ylim=(0,1))
plot!(label = "Step", result_stp, xlabel="epochs", ylabel="precision", ylim=(0,1))


## What have we seen in this chapter?
