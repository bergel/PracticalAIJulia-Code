# Data Classification
## Train a network
Random.seed!(42)
n = NNetwork(2, 3, 1)
for _ in 1:20_000
	train!(n, [0, 0], [0])
	train!(n, [0, 1], [1])
	train!(n, [1, 0], [1])
	train!(n, [1, 1], [0])
end


# `n` is the network to train
# `data` is an array of array of numbers (inputs with integer as expected output)
# `epochs_count` is the number of epochs for which `n` is trained
# `learning_rate` is the learning rate, a small positive number
function train!(n::NNetwork, data::Vector, epochs_count::Int, learning_rate::Float64=0.1)
	for _ in 1:epochs_count
		error_sum = 0
		epoch_precision = 0
		precisions = []
		for row in data
			# the convertion below is just to make sure we do not end up with
			# Vector{Any} if we have float and integer values.
			input = convert(Vector{Number}, row[1:end-1])
			output = feed(n, input)
			expected_output = zero(1:outputs_count(n))
			expected_output[convert(Integer, row[end]) + 1] = 1

			# If our prediction marches the expected value, we increase the precision
			if row[end] == predict(n, input)
				epoch_precision += 1
			end

			all_errors = [(expected_output[i] - output[i])^2 for i in 1:outputs_count(n)]
			error_sum += sum(all_errors)
			backward_propagate_error(n, expected_output)
			update_weights(n, input, learning_rate)
		end
		push!(n.errors, error_sum)
		push!(n.precisions, epoch_precision / length(data))
	end
end


# Make a prediction. The function assumes that the number of output equals the number of
# different values the network can output.
function predict(n::NNetwork, input::Vector{T}) where T <: Number
	output = feed(n, input)
	return findmax(output)[2] - 1
end


Random.seed!(42)
n = NNetwork(2, 3, 2)
data = [[0, 0, 0],
		[0, 1, 1],
		[1, 0, 1],
		[1, 1, 0]]
train!(n, data, 20_000)


n = NNetwork(3, 8, 8)
data = [
	[0, 0, 0, 0],
	[0, 0, 1, 1],
	[0, 1, 0, 2],
	[0, 1, 1, 3],
	[1, 0, 0, 4],
	[1, 0, 1, 5],
	[1, 1, 0, 6],
	[1, 1, 1, 7]]
train!(n, data, 1_000)


predict(n, [0, 1, 1])


## Neural network as a hashmap
data = [
	[0, 0, 0, 0],
	[0, 0, 1, 1],
	[0, 1, 0, 2],
	[0, 1, 1, 3],
	[1, 0, 0, 4],
	[1, 0, 1, 5],
	[1, 1, 0, 6],
	[1, 1, 1, 7]]
d = Dict()
for row in data
	d[row[1:end-1]] = row[end]
end
d[[0, 1, 1]] # Return 3


n = NNetwork(3, 8, 8)
data = [
	[0, 0, 0, 0],
	[0, 0, 1, 1],
	[0, 1, 0, 2],
	[0, 1, 1, 3],
	[1, 0, 0, 4],
	[1, 0, 1, 5],
	[1, 1, 0, 6],
	[1, 1, 1, 7]]
train!(n, data, 1_000)
predict(n, [0.4, 0.7, 0.6]) # Return 3


## Visualizing the error and the precision
using Plots: twinx
function nn_plot(n::NNetwork)
	plot(
		n.errors,
		label="error",
		xlabel = "epoch",
		legend=:topleft,
		y_foreground_color_border=:blue,
		y_foreground_color_text=:blue,
		y_guidefontcolor=:blue,
		ylims=(0, :auto)
	)
	plot!(
		twinx(),
		n.precisions,
		color=:red,
		xticks=:none,
		label="precision",
		legend=:topright,
		y_foreground_color_border=:red,
		y_foreground_color_text=:red,
		y_guidefontcolor=:red,
		ylims=(0, 1)
	)
end


Random.seed!(42)
n = NNetwork(2, 3, 2)
data = [[0, 0, 0],
		[0, 1, 1],
		[1, 0, 1],
		[1, 1, 0]]
train!(n, data, 5_000)
nn_plot(n)


## Contradictory data
Random.seed!(42)
n = NNetwork(2, 3, 2)
data = [[0, 0, 0],
		[0, 0, 1]]
train!(n, data, 1_000)
nn_plot(n)


## Classifying data and one hot encoding
n = NNetwork(2, 3, 2)
data = [[0, 0, 0],
		[0, 1, 1],
		[1, 0, 1],
		[1, 1, 0]]
train!(n, data, 5_000)


## Iris dataset
using CSV, DataFrames
CSV.read(download("https://agileartificialintelligence.github.io/Datasets/iris.csv"), DataFrame)


using CSV, DataFrames
df = CSV.read(download("https://agileartificialintelligence.github.io/Datasets/iris.csv"), DataFrame)
iris_data = []
for index in 1:nrow(df)
	tp = values(df[index,:])
	local hot
	if tp[end] == "setosa" hot = 0 end
	if tp[end] == "versicolor" hot = 1 end
	if tp[end] == "virginica" hot = 2 end
	push!(iris_data, collect((tp[1:end-1]..., (hot))))
end


## Training a network with the iris dataset
n = NNetwork(4, 6, 3)
train!(n, iris_data, 1000)
nn_plot(n)


## Effect of the learning curve
learning_rates = [0.001, 0.01, 0.1, 0.2, 0.3]
EPOCH_COUNT = 1000
pl = plot()
for learning_rate in learning_rates
	local learning_curve = []

	Random.seed!(42)
	local p = Neuron([-1, -1], 2)
	sigmoid!(p)

	for epoch_count in 1:EPOCH_COUNT
		train!(p, [0, 0], 0, learning_rate)
		train!(p, [0, 1], 0, learning_rate)
		train!(p, [1, 0], 0, learning_rate)
		train!(p, [1, 1], 1, learning_rate)

		local result = abs(feed(p, [0, 0])- 0) +
		 		 abs(feed(p, [0, 1])- 0) +
		 		 abs(feed(p, [1, 0])- 0) +
		 		 abs(feed(p, [1, 1])- 1)
		push!(learning_curve, result / 4)
	end
	plot!(pl, learning_curve, label = string(learning_rate))
end
plot(pl)


# Repeat the script with a different learning rate
n = NNetwork(4, 6, 3)
train!(n, iris_data, 1000, 0.1)
nn_plot(n)


## Test and validation
cut = 0.8
data_training_length = round(Int, length(iris_data) * cut)
data_test_length = round(Int, length(iris_data) - data_training_length)
data_training = iris_data[1:data_training_length]
data_test = iris_data[end - data_test_length + 1:end]


n = NNetwork(4, 6, 3)
train!(n, data_training, 1000)
nn_plot(n)


cut = 0.8
data_training_length = round(Int, length(iris_data) * cut)
data_test_length = round(Int, length(iris_data) - data_training_length)
data_training = iris_data[1:data_training_length]
data_test = iris_data[end - data_test_length + 1:end]

n = NNetwork(4, 6, 3)
train!(n, data_training, 1000)

correct_ratio = sum([predict(n, row[1:end-1]) == row[end] for row in data_test]) / data_test_length
round(correct_ratio, digits = 2)


cut = 0.6
data_training_length = round(Int, length(iris_data) * cut)
data_test_length = round(Int, length(iris_data) - data_training_length)
data_training = iris_data[1:data_training_length]
data_test = iris_data[end - data_test_length + 1:end]

n = NNetwork(4, 6, 3)
train!(n, data_training, 1000)

correct_ratio = sum([predict(n, row[1:end-1]) == row[end] for row in data_test]) / data_test_length
round(correct_ratio, digits = 2)


Random.seed!(42)
shuffled_iris_data = shuffle(iris_data)
cut = 0.6
data_training_length = round(Int, length(shuffled_iris_data) * cut)
data_test_length = round(Int, length(shuffled_iris_data) - data_training_length)
data_training = shuffled_iris_data[1:data_training_length]
data_test = shuffled_iris_data[end - data_test_length + 1:end]

n = NNetwork(4, 6, 3)
train!(n, data_training, 1000)

correct_ratio = sum([predict(n, row[1:end-1]) == row[end] for row in data_test]) / data_test_length
round(correct_ratio, digits = 2)


## Normalization
data_max = []
data_min = []
for index in 1:4
	push!(data_max, max(map(row -> row[index], iris_data)...))
	push!(data_min, min(map(row -> row[index], iris_data)...))
end
collect(zip(data_min, data_max))


n = NNetwork(3, 8, 8)
data = [
	[0, 0, 0, 0],
	[0, 0, 1, 1],
	[0, 1, 0, 2],
	[0, 1, 1, 3],
	[1, 0, 0, 4],
	[1, 0, 1, 5],
	[1, 1, 0, 6],
	[1, 1, 1, 7]
]
train!(n, data, 1000)
nn_plot(n)


n = NNetwork(3, 8, 8)
data = [
	[0, 0, 0, 0],
	[0, 0, 1, 1],
	[0, 1000, 0, 2],
	[0, 1000, 1, 3],
	[0.1, 0, 0, 4],
	[0.1, 0, 1, 5],
	[0.1, 1000, 0, 6],
	[0.1, 1000, 1, 7]
]
train!(n, data, 1000)
nn_plot(n)


function normalize_data(training_data::Vector)
	@assert length(training_data) >= 2 "Not enough data to normalize"

	# Exclude expected output
	column_count = length(first(training_data)) - 1

	data_min = []
	data_max = []
	for index in 1:column_count
		values = [row[index] for row in training_data]
		push!(data_min, min(values...))
		push!(data_max, max(values...))
	end
	return normalize_data(training_data, data_min, data_max)
end


function normalize_data(training_data::Vector, data_min::Vector, data_max::Vector)
	@assert length(training_data) >= 2 "Not enough data to normalize"
	# Exclude expected output
	column_count = length(first(training_data)) - 1

	normalized_data = []
	for row in training_data
		new_row = []
		for index in 1:column_count
			v = row[index]
			min_at_index = data_min[index]
			max_at_index = data_max[index]
			push!(new_row, (v - min_at_index)/(max_at_index - min_at_index))
		end
		push!(new_row, row[end])
		push!(normalized_data, new_row)
	end
	return normalized_data
end


@testset "Simple normalization" begin
	data = [[10, 5, 1], [2, 6, 0]]
	normalized_data = normalize_data(data)
	@test normalized_data == [[1.0, 0.0, 1], [0.0, 1.0, 0]]
end


@testset "Erroneous normalization" begin
	@test_throws AssertionError normalize_data([[10, 5, 1]])
	@test_throws AssertionError normalize_data([])
end


## Integrating the normalization into NNetwork
function train!(
	n::NNetwork,
	raw_data::Vector,
	epochs_count::Int,
	learning_rate::Float64=0.1,
	should_normalize::Bool=false
)
	data = should_normalize ? normalize_data(raw_data) : raw_data
	for _ in 1:epochs_count
		error_sum = 0
		epoch_precision = 0
		precisions = []
		for row in data
			input = convert(Vector{Number}, row[1:end-1])
			output = feed(n, convert(Vector{Number}, input))
			expected_output = zero(1:outputs_count(n))
			expected_output[convert(Integer, row[end]) + 1] = 1
			if row[end] == predict(n, input)
				epoch_precision += 1
			end

			all_errors = [(expected_output[i] - output[i])^2 for i in 1:outputs_count(n)]
			error_sum += sum(all_errors)
			backward_propagate_error(n, expected_output)
			update_weights(n, input, learning_rate)
		end
		push!(n.errors, error_sum)
		push!(n.precisions, epoch_precision / length(data))
	end
end


n = NNetwork(3, 8, 8)
data = [
	[0, 0, 0, 0],
	[0, 0, 1, 1],
	[0, 1000, 0, 2],
	[0, 1000, 1, 3],
	[0.1, 0, 0, 4],
	[0.1, 0, 1, 5],
	[0.1, 1000, 0, 6],
	[0.1, 1000, 1, 7]
]
train!(n, data, 1000, 0.1, true)
nn_plot(n)


## What have we seen in this chapter?
