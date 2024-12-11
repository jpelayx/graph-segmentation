using SparseArrays
using GraphNeuralNetworks
using NNlib: σ, tanh, tanh_fast
using Statistics

import Images
import Graphs

get_segments(S::SparseMatrixCSC{Float64, Int64}, cs::AbstractVector) = @view S[:, cs]

function get_segments_from_vertices(S::SparseMatrixCSC{Float64, Int64}, vi::Int) 
     get_segments(S, S[vi,:].nzind), S[vi,:].nzind
end

adjust_u!(C_off, C, u) = C_off[u] *= C[u]
adjust_v!(C_off, v) = C_off[v] = 0.0


multsp(A, B) = A .* B

function merge_segments!(
    S, segment_size, internal_diff, 
    P, V, V_I, U, U_I, v, u, w, N
)
    Ci = eachcol(V)
    Cj = eachcol(U)

    Cj_off = Cj .* sum(P, dims=2)
    adjust_u!.(Cj_off, Cj, u)

    get_offset(MtoC) = sparsevec(vcat(rowvals.(Cj)...), vcat(MtoC .* nonzeros.(Cj)...), N)
    notC(C, I) = sparsevec(
        vcat(I, rowvals(C[I])), 
        vcat(ones(Float64, length(I)), -nonzeros(C[I])), 
        N
    )
    toCi = get_offset.(eachcol(P))
    notCi = notC.(Ci, rowvals.(toCi))
    Ci_off = multsp.(notCi, toCi)
    adjust_v!.(Ci_off, v)

    Mi = sum(P, dims=1)'
    internal_diff[V_I] = (1 .- Mi) .* internal_diff[V_I] + Mi .* w
    segment_size[V_I] .+= [sum(col .* segment_size[U_I]) for col in eachcol(P)]

    Ci .+= Ci_off
    Cj .-= Cj_off 
end

function merge_probability(
    V_I, V, U_I, U, 
    internal_diff, 
    segment_size, 
    v, u, w, 
    k, μ = 2.0, min_prob=1e-2
)
    τ(V_I, k) = k./segment_size[V_I]
    MInt = minimum.(
        Iterators.product(internal_diff[U_I] .+ τ(U_I, k), 
                          internal_diff[V_I] .+ τ(V_I, k)))
    Mij_conditional = tanh_fast.((MInt .- w) .* μ)

    intersections = V_I ∩ U_I
    for absolute_index in intersections
        i = findfirst(x -> x == absolute_index, V_I)
        j = findfirst(x -> x == absolute_index, U_I)
        Mij_conditional[j,i] = 0.0
    end
    V_K = nonzeros(V[v,:])
    U_K = nonzeros(U[u,:])
    Mij = max.(Mij_conditional .* (U_K * V_K'), 0.0)
    Mij[Mij .< min_prob] .= 0.0
    return Mij
end

function compute_edge_weights(g::GNNGraph)
    src, dst = edge_index(g)
    w = mean(abs.(g.x[:, src] .- g.x[:, dst]), dims=1)
end

function felzenszwalb(g::GNNGraph; tol=1e-6, μ=1.0, k=1.5)
    edge_weights = compute_edge_weights(g)
    sorted_edges = sortperm(edge_weights, dims=2)
    src, dst = edge_index(g)
    E = length(sorted_edges)
    N = g.num_nodes

    Is = [i for i in 1:N]
    Vs = [1.0 for _ in 1:N]
    S = sparse(Is, Is, Vs, N, N)

    segment_size = ones(Float64, N)
    internal_diff = zeros(Float64, N)

    for edge_num in 1:E
        if edge_num % 100 == 0
            println("Edge $edge_num/$E")
        end
        i = sorted_edges[edge_num]
        v, u = src[i], dst[i]
        w = g.e[i]
        V, V_I = get_segments_from_vertices(S, v)
        U, U_I = get_segments_from_vertices(S, u)
        P = merge_probability(V_I, V, U_I, U, internal_diff, segment_size, v, u, w, k, μ)
        merge_segments!(S, segment_size, internal_diff, P, V, V_I, U, U_I, v, u, w, N)
        droptol!(S, tol)
    end
    return S
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

dims = (28, 28)
image = load_sample_image("data/astronaut.png", dims)
g = rag_from_image(image)


edge_weights = compute_edge_weights(g)
sorted_edges = sortperm(edge_weights, dims=2)
src, dst = edge_index(g)
E = length(sorted_edges)
N = g.num_nodes

Is = [i for i in 1:N]
Vs = [1.0 for i in 1:N]
S = sparse(Is, Is, Vs, N, N)

segment_size = ones(Float64, N)
internal_diff = zeros(Float64, N)

edge_num = 1
function loop(;k=100, μ=1.0, tol=1e-6)
    global  edge_num
    print("Edge $edge_num/$E: ")

    i = sorted_edges[edge_num]
    v, u = src[i], dst[i]
    w = g.e[i]
    println("v: $v, u: $u, w: $w")

    V, V_I = get_segments_from_vertices(S, v)
    U, U_I = get_segments_from_vertices(S, u)
    @assert all([v in rowvals(C) for C in eachcol(V)])
    @assert all([u in rowvals(C) for C in eachcol(U)])
    
    P = merge_probability(V_I, V, U_I, U, internal_diff, segment_size, v, u, w, k, μ)
    try
        @assert all(P .>= 0.0)
        @assert all(P .<= 1.0)
    catch e
        println(e)
        println(P)
    end
    
    merge_segments!(S, segment_size, internal_diff, P, V, V_I, U, U_I, v, u, w,N)
    edge_num += 1
    try
        @assert isapprox(sum(S[v,:]), 1.0)
    catch e
        # println(e)
        # println(S[v,:])
        # println()
    end
    try
        @assert isapprox(sum(S[u,:]), 1.0)
    catch e
        # println(e)
        # println(S[u,:])
        # println()
    end
    
    droptol!(S, tol)
    return v, u, V, V_I, U, U_I, P
end