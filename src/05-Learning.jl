# Theory on Learning
## Loss function
points = [(1, 3), (3, 5.2), (2, 4.1), (4, 7.5)]
scatter(points; xlims=(0, :auto), ylims=(0, :auto))


points = [(1, 3), (3, 5.2), (2, 4.1), (4, 7.5)]
scatter(points; xlims=(0, :auto), ylims=(0, :auto))

a = 0.5
b = 3
f(x) = a*x + b
plot!(f)


points = [(1, 3), (3, 5.2), (2, 4.1), (4, 7.5)]
a = 0.5
b = 3
f(x) = a*x + b
j = sum([(yi - f(xi))^2 for (xi, yi) in points]) / length(points)


## Gradient descent
points = [(1, 3), (3, 5.2), (2, 4.1), (4, 7.5)]

a = 0.5
b = 3
f(x) = a*x + b

alpha = 0.01
pts_count = length(points)
for _ in 1:1000
	deriMSEa = (-2 / pts_count) * sum([xi*(yi - f(xi)) for (xi, yi) in points])
	deriMSEb = (-2 / pts_count) * sum([(yi - f(xi)) for (xi, yi) in points])
	global a = a - (alpha * deriMSEa)
	global b = b - (alpha * deriMSEb)
end

scatter(points; xlims=(0, :auto), ylims=(0, :auto))
plot!(f)


## Parameter update
points = [(1, 3), (3, 5.2), (2, 4.1), (4, 7.5)]

a = 0.5
b = 3
f(x) = a*x + b

alpha = 0.01
pts_count = length(points)
result = []
for _ in 1:1000
	deriMSEa = (-2 / pts_count) * sum([xi*(yi - f(xi)) for (xi, yi) in points])
	deriMSEb = (-2 / pts_count) * sum([(yi - f(xi)) for (xi, yi) in points])
	global a = a - (alpha * deriMSEa)
	global b = b - (alpha * deriMSEb)
	mse = sum([(yi - f(xi))^2 for (xi, yi) in points]) / pts_count
	push!(result, mse)
end
plot!(result)


## What have we seen in this chapter?
## Further reading
