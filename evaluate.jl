using  Statistics
using  Plots

using NNlib: tanh_fast, tanh
using GraphNeuralNetworks

import Images 
import Graphs

# include("felzenszwalb.jl")
include("node_felzenszwalb.jl")

function compute_edge_weights(g::GNNGraph)
    src, dst = edge_index(g)
    w = mean(sqrt.((g.x[:, src] .- g.x[:, dst]).^2), dims=1)
end

function rag(dims::Tuple{Int, Int})
    pixel_index(x, y, width) = (y-1)*width + x

    neighbor_offsets = [
        (1, 0),  # right
        (-1, 1),
        (0, 1),
        (1, 1)   # Diagonal down-left and down-right
    ]
    
    N = dims[1]*dims[2]
    E = Int(4*N - 3*(dims[1] + dims[2]) + 2)
    # E = 2*N - dims[1] - dims[2]
    src, dst = Vector{Int64}(undef, E), Vector{Int64}(undef, E)

    edge_index = 1
    for y in 1:dims[1]
        for x in 1:dims[2]
            current_pixel = pixel_index(x, y, dims[2])
            for (dx, dy) in neighbor_offsets
                nx, ny = x + dx, y + dy
                if 1 <= nx <= dims[2] && 1 <= ny <= dims[1]
                    neighbor_pixel = pixel_index(nx, ny, dims[2])
                    src[edge_index] = current_pixel
                    dst[edge_index] = neighbor_pixel
                    edge_index += 1
                end
            end
        end
    end
         
    return GNNGraph(src, dst, num_nodes=N)
end

function rag(dims:: Tuple{Int, Int})
    grid = Graphs.grid(dims)
    return to_unidirected(GNNGraph(grid))
end

function rag_from_image(img)
    dims = size(img)
    g = rag(dims)
    x = Images.channelview(img)
    x = reshape(x, (3, dims[1]*dims[2]))
    g.ndata.x = Float64.(x)
    g.edata.e = compute_edge_weights(g)
    return g
end

function load_sample_image(img_path="data/astronaut.png", dims=(32,32))
    img = Images.load(img_path)
    img = Images.RGB.(img)
    img = Images.imresize(img, dims)
    return img
end

function save_masked_image(dims, S, path="data/result_mask.png")
    segments = argmax.(eachrow(S))
    segments = reshape(segments, dims)
    unique_segments = unique(segments)
    color_map = Dict([(seg, Images.ColorTypes.RGB(rand(), rand(), rand())) for seg in unique_segments])
    mask = [color_map[seg] for seg in segments]
    mask = reshape(mask, dims)
    Images.save(path, mask)    
end

dims = (128, 128)
image = load_sample_image("data/astronaut.png", dims)
g = rag_from_image(image)

ks = [0.1,1,5,10,15,25,50,100,150,200,250,500]
μs = [5,10]

tol = 1e-2

function evaluate(ks, μs, tol)
    for (k, μ) in Iterators.product(ks, μs)
        print("k: $k, μ: $μ")
        time = @elapsed S = felzenszwalb(g, k=k, μ=μ, tol=tol)
        print(", time: $time")
        prob_sum = sum.(eachrow(S))
        print(", mean: $(mean(prob_sum)), std: $(std(prob_sum))")
        println()
        save_masked_image(dims, S, "results/felzenszwalb_k$(k)_mu$(μ).png")
    end
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