# Neuroevolution
## Supervised, unsupervised learning, reinforcement learning
## Neuroevolution
## Two techniques for neuroevolution
## The NeuroGenetic approach
## Neural network serialization and materialization
function nn_serialize(n::NNetwork)
	result = []
	current_layer = n.root_layer
	while !isnothing(current_layer)
		for neuron in current_layer.neurons
			append!(result, neuron.weights)
			append!(result, [neuron.bias])
		end
		current_layer = current_layer.next_layer
	end
	return result
end


function nn_materialize!(n::NNetwork, values::Vector)
	index = 1
	current_layer = n.root_layer
	while !isnothing(current_layer)
		for neuron in current_layer.neurons
			new_weights = values[index:(index + length(neuron.weights) - 1)]
			neuron.weights = convert(Vector{Number}, new_weights)
			neuron.bias = values[(index + length(neuron.weights))]
			index += length(neuron.weights) + 1
		end
		current_layer = current_layer.next_layer
	end
end


@testset "Network materializing and serializing" begin
	n1 = NNetwork(2, 3, 1)
	for _ in 1:20_000
		train!(n1, [0, 0], [0])
		train!(n1, [0, 1], [1])
		train!(n1, [1, 0], [1])
		train!(n1, [1, 1], [0])
	end

	n2 = NNetwork(2, 3, 1)
	nn_materialize!(n2, nn_serialize(n1))
	@test feed(n1, [0, 0]) == feed(n2, [0, 0])
	@test feed(n1, [0, 1]) == feed(n2, [0, 1])
	@test feed(n1, [1, 0]) == feed(n2, [1, 0])
	@test feed(n1, [1, 1]) == feed(n2, [1, 1])
end


## Evolving weights and biases
data = [
	[0, 0, 0],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 0]
]


chromosome_length = length(nn_serialize(NNetwork(2, 3, 1)))


function nn_error(genes)
	n = NNetwork(2, 3, 1)
	nn_materialize!(n, genes)
	return mean([(first(feed(n, row[1:end-1])) - row[end])^2 for row in data])
end


function nn_precision(genes)
	n = NNetwork(2, 3, 1)
	nn_materialize!(n, genes)

	t(inputs) = (first(feed(n, inputs)) > 0.5) ? 1 : 0
	score_per_row = [t(row[1:end-1]) == row[end] for row in data]
	return sum(score_per_row) / length(data)
end


logs = ga_run(
	nn_error,
	(_, _) -> rand() * 10 - 5,
	chromosome_length
	;
	population_size = 500,
	termination = logs -> logs[end].best.fitness < 0.01 || logs[end].generation == 80,
	compare_fitness = <
);

ga_plot(logs)


## Converting binary values to decimal
data = [
	[0, 0, 0, 0],
	[0, 0, 1, 1],
	[0, 1, 0, 2],
	[0, 1, 1, 3],
	[1, 0, 0, 4],
	[1, 0, 1, 5],
	[1, 1, 0, 6],
	[1, 1, 1, 7],
]

# Number of input values per row
data_inputs_count = length(first(data)) - 1

# Number of different output values
data_outputs_count = length(Set([r[end] for r in data]))
mid_layer_size = data_inputs_count * 2
chromosome_length = length(nn_serialize(NNetwork(data_inputs_count, mid_layer_size, data_outputs_count)))

function nn_error(genes)
	n = NNetwork(data_inputs_count, mid_layer_size, data_outputs_count)
	nn_materialize!(n, genes)
	tmp = []
	for row in data
		computed_value = feed(n, row[1:end-1])
		expected_value = zero(1:data_outputs_count)
		expected_value[row[end] + 1] = 1
		append!(tmp, (computed_value - expected_value).^2)
	end

	return mean(tmp)
end

function nn_precision(genes)
	n = NNetwork(data_inputs_count, mid_layer_size, data_outputs_count)
	nn_materialize!(n, genes)
	score = 0
	for row in data
		computed_value = predict(n, row[1:end-1])
		expected_value = row[end]
		if expected_value == computed_value
			score += 1
		end
	end
	return score / length(data)
end

logs = ga_run(
	nn_error,
	(_, _) -> rand() * 10 - 5,
	chromosome_length
	;
	population_size = 500,
	mutation_operator=GAMutationOperation(0.001),
	termination = logs ->
					logs[end].generation == 200 ||
					nn_precision(logs[end].best.genes) == 1.0
			,
	compare_fitness = <
);

ga_plot(logs)


n = NNetwork(data_inputs_count, mid_layer_size, data_outputs_count)
nn_materialize!(n, logs[end].best.genes)
predict(n, [1, 1, 0])


## Iris dataset
using CSV, DataFrames
df = CSV.read(download("https://agileartificialintelligence.github.io/Datasets/iris.csv"), DataFrame)
data = []
for index in 1:nrow(df)
	tp = values(df[index,:])
	local hot
	if tp[end] == "setosa" hot = 0 end
	if tp[end] == "versicolor" hot = 1 end
	if tp[end] == "virginica" hot = 2 end
	push!(data, collect((tp[1:end-1]..., (hot))))
end


# Number of input values per row
data_inputs_count = length(first(data)) - 1

# Number of different output values
data_outputs_count = length(Set([r[end] for r in data]))
mid_layer_size = data_inputs_count * 2
chromosome_length = length(nn_serialize(NNetwork(data_inputs_count, mid_layer_size, data_outputs_count)))

logs = ga_run(
	nn_error,
	(_, _) -> rand() * 10 - 5,
	chromosome_length
	;
	population_size = 500,
	mutation_operator=GAMutationOperation(0.001),
	termination=logs->logs[end].best.fitness < 0.01 || logs[end].generation == 80,
	compare_fitness = <
);


## Further reading about NeuroGenetic
## What have we seen in this chapter?
