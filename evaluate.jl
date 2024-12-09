include("felzenszwalb.jl")
using  Statistics
using  Plots

using NNlib: tanh_fast, tanh

dims = (128, 128)
image = load_sample_image("data/astronaut.png", dims)
g = rag_from_image(image)

ks = [0.1,1,5,10,15,25,50,100,150,200,250,500]
μs = [5,10]

tol = 1e-2

for (k, μ) in Iterators.product(ks, μs)
    print("k: $k, μ: $μ")
    time = @elapsed S = felzenszwalb(g, k=k, μ=μ, tol=tol)
    print(", time: $time")
    prob_sum = sum.(eachrow(S))
    print(", mean: $(mean(prob_sum)), std: $(std(prob_sum))")
    println()
    save_masked_image(dims, S, "results/felzenszwalb_k$(k)_mu$(μ).png")
end

function plot_tanh(μ::Float64)
    x = -1:0.001:1
    y = tanh_fast.(x * μ)
    plot(x, y, label="μ=$μ")
end

function plot_tanh(μs::AbstractVector{<:Number})
    x = -1:0.001:1
    p = plot()
    for μ in μs
        plot!(x, tanh.(x * μ), label="μ=$(μ)")
    end
    return p
end