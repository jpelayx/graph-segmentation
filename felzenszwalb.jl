using SparseArrays
using GraphNeuralNetworks
using NNlib: σ

import Images

function initialize_structures(N::Int) 
    S = Vector{SparseVector{Float64, Int64}}(undef, N) 
    segment_hash = Dict{Int64, Set{Int64}}(zip(1:N, Set([i]) for i in 1:N))
    segment_size = ones(Float64, N)
    internal_diff = zeros(Float64, N)
    for i in 1:N
        S[i] = sparsevec([i], [1.0], N)
    end
    return S, segment_hash, segment_size, internal_diff
end

get_segments(S::Vector{SparseVector{Float64, Int64}}, cs::AbstractVector) = @view S[cs]

get_segments_from_vertices(S::Vector{SparseVector{Float64, Int64}}, vi::Int) = @inbounds [c for c in 1:length(S) if S[c][vi] > 0.0]

function get_segments_from_vertices(S::Vector{SparseVector{Float64, Int64}}, segment_hash::Dict{Int64, Vector{Int64}}, vi::Int) 
    V_I = @inbounds segment_hash[vi]
    V_K = [S[i][vi] for i in V_I]
    return V_I, V_K
end

function get_segments_from_vertices(S::Vector{SparseVector{Float64, Int64}}, segment_hash::Dict{Int64, Set{Int64}}, vi::Int) 
    V_I = @inbounds collect(segment_hash[vi])
    V_K = [S[i][vi] for i in V_I]
    return V_I, V_K
end

adjust_offset!(C, C_off, u) = C_off[u] *= C[u]

not(C) = dropzeros(sparsevec(C.nzind, 1 .- C.nzval, N))
spmult(A, B) = sparsevec(cat(A.nzind, B.nzind, dims=1), cat(A.nzval, B.nzval, dims=1), N, *)

function add_to_segment_hash!(segment::Int, C::SparseVector{Float64, Int64})
    for i in C.nzind
        push!(segment_hash[i], segment)
    end
end

function  remove_from_segment_hash!(segment::Int, C::SparseVector{Float64, Int64}, C_new::SparseVector{Float64, Int64})
    for i in setdiff(C.nzind, C_new.nzind)    
        delete!(segment_hash[i], segment)
    end
end

function merge_segments!(
    S::Vector{SparseVector{Float64, Int64}}, 
    segment_hash::Dict{Int64, Set{Int64}}, 
    segment_size::Vector{Float64}, 
    internal_diff::Vector{Float64}, 
    P::Matrix{Float64}, w::Float64, 
    V_I, U_I,
    u, v
)
    Ci = @view S[V_I]
    Cj = @view S[U_I]

    Cj_off = -Cj .* sum(P, dims=2)
    adjust_offset!.(Cj, Cj_off, U_I)

    Ci_off = spmult.(not.(Ci), [sum(col .* Cj) for col in eachcol(P)])

    Mi = sum(P, dims=1)'
    internal_diff[V_I] = (1 .- Mi) .* internal_diff[V_I] + Mi .* w
    
    segment_size[V_I] .+= [sum(col .* segment_size[U_I]) for col in eachcol(P)]

    S[V_I] .+= Ci_off
    S[U_I] .+= Cj_off

    add_to_segment_hash!.(V_I, Ci_off)
    remove_from_segment_hash!.(U_I, Cj_off, S[U_I])
end

function merge_probability(Vi_I, Vi_K, Vj_I, Vj_K, internal_diff,segment_size, w, k)
    τ(V_I, k) = k./segment_size[V_I]
    MInt = minimum.(
        Iterators.product(internal_diff[Vj_I] .+ τ(Vj_I, k), 
                          internal_diff[Vi_I] .+ τ(Vi_I, k)))
    Mij_conditional = σ.(MInt .- w)
    Mij = Mij_conditional .* (Vj_K * Vi_K')
    return Mij
end

function rag(dims::Tuple{Int, Int})
    pixel_index(x, y, width) = (y-1)*width + x

    neighbor_offsets = [
        (-1, 0), (1, 0),  # Left and right
        (0, -1), (0, 1),  # Up and down
        (-1, -1), (1, -1),  # Diagonal up-left and up-right
        (-1, 1), (1, 1)   # Diagonal down-left and down-right
    ]
    
    N = dims[1]*dims[2]
    E = 8*N - 6*(dims[1] + dims[2]) + 4
    src, dst = Vector{Int64}(undef, E), Vector{Int64}(undef, E)

    edge_index = 1
    @inbounds for y in 1:dims[1]
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


function compute_edge_weights(g::GNNGraph)
    src, dst = edge_index(g)
    x_dim = size(g.x, 1)

    w = @views sum(abs.(g.x[:, src] .- g.x[:, dst]), dims=1) ./ x_dim

    return w
end

function rag_from_image(img)
    dims = size(img)
    g = rag(dims)
    x = Images.channelview(img)
    x = reshape(x, (3, dims[1]*dims[2]))
    g.ndata.x = x
    g.edata.e = compute_edge_weights(g)
    return g
end

function felzenszwalb(g::GNNGraph)
    sorted_edges = sortperm(g.e', dims=1)
    src, dst = edge_index(g)

    for i in sorted_edges
        v, u = src[i], dst[i]
        w = g.e[i]
        V_I, V_K = get_segments_from_vertices(S, segment_hash, v)
        U_I, U_K = get_segments_from_vertices(S, segment_hash, u)
        P = merge_probability(V_I, V_K, U_I, U_K, internal_diff, segment_size, w, 300/255)
        merge_segments!(S, segment_hash, segment_size, internal_diff, P, w, V_I, U_I, u, v)
    end
end

function load_sample_image()
    img = Images.load("data/astronaut.png")
    img = Images.RGB.(img)
    img = Images.imresize(img, (128, 128))
    return img, (128, 128)
end

img, dims = load_sample_image()
N = dims[1]*dims[2]
g = rag_from_image(img)
S, segment_hash, segment_size, internal_diff = initialize_structures(N)

felzenszwalb(g)