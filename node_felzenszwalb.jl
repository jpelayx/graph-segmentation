"""
This is the NODE-like implementation of the Felzenszwalb algorithm.
"""

using Zygote
using Zygote: @adjoint
using GraphNeuralNetworks
using NNlib: σ, tanh, tanh_fast
using LinearAlgebra

"""
d/dt h(t) = f(h(t), t, θ)
h(t+1) = h(t) + 1 * f(h(t), t, θ)
"""

k = 25.0
μ = 3.0
min_prob = 1e-3

multsp(A, B) = A .* B

get_segments(S::SparseMatrixCSC{Float64,Int64}, cs::AbstractVector) = @view S[:, cs]

get_segments_from_vertices(S::SparseMatrixCSC{Float64,Int64}, vi::Int) =
    get_segments(S, S[vi, :].nzind), S[vi, :].nzind

function clear_intersections!(P, Vi, Ui)
    intersections = Vi ∩ Ui
    if !isempty(intersections)
        is = indexin(intersections, Vi)
        js = indexin(intersections, Ui)
        P[js, is] .= 0.0
    end
    return P
end

@adjoint clear_intersections!(P, Vi, Ui) = clear_intersections!(P, Vi, Ui), Δ -> begin
    δP = Δ[1]
    intersections = Vi ∩ Ui
    if !isempty(intersections)
        is = indexin(intersections, Vi)
        js = indexin(intersections, Ui)
        δP[js, is] .= 0.0
    end
    return (δP, nothing, nothing)
end

function merge_probability(
    Vi, V, Ui, U,
    internal_diff,
    segment_size,
    v, u, weight
)
    τ(Vi, k) = k ./ segment_size[Vi]
    MInt = minimum.(
        Iterators.product(internal_diff[Ui] .+ τ(Ui, k),
            internal_diff[Vi] .+ τ(Vi, k)))
    Mij_conditional = tanh_fast.((MInt .- weight) .* μ)

    if !isempty(Vi ∩ Ui)
        clear_intersections!(Mij_conditional, Vi, Ui)
    end

    Mij = max.(Mij_conditional .* (U[u] * V[v]'), 0.0)
    return Mij
end

function adjust_u!(dU, U, i) 
    dU[i, :] .*= U[i, :]
    return dU
end 

@adjoint adjust_u!(dU, U, i) = adjust_u!(dU, U, i), Δ -> begin
    δdU = Δ[1]
    δU = zeros(size(δdU))
    δU[i,:] .= dU[i,:] .* Δ[1][i,:]
    δdU[i,:] .*= U[i,:]
    return (δdU, δU, nothing)
end

function adjust_v!(dV, i) 
    dV[i,:] .= 0.0
    return dV
end

@adjoint adjust_v!(dV, i) = adjust_v!(dV, i), Δ -> begin
    δdV = Δ[1]
    δdV[i] .= 0.0
    return (δdV, nothing)
end

function f(S, internal_diff, segment_size, t, w, E)
    weight = w[t]
    v, u = E[t]
    Vi = findall(x -> x > min_prob, S[v, :])
    Ui = findall(x -> x > min_prob, S[u, :])
    V = @view S[:, Vi]
    U = @view S[:, Ui]

    P = merge_probability(Vi, V, Ui, U, internal_diff, segment_size, v, u, weight)

    dU = S[:, Ui] .* sum(P, dims=2)
    adjust_u!(dU, U, u)

    dV = (1 .- V) .* (U * P')
    adjust_v!(dV, v)

    dV_I = rowvals.(dV)
    dV_J = fill.(Vi, length.(dV_I))
    dU_I = rowvals.(dU)
    dU_J = fill.(Ui, length.(dU_I))
    dS = sparse(
        vcat(dV_I..., dU_I...),
        vcat(dV_J..., dU_J...),
        vcat(nonzeros.(dV)..., -nonzeros.(dU)...),
        S.n, S.n
    )

    Mi = sum(P, dims=1)'
    internal_diff_offset = zeros(size(S)[1])
    internal_diff_offset[Vi] = (1 .- Mi) .* internal_diff[Vi] + Mi .* weight

    segment_size_offset = zeros(size(S)[1])
    segment_size_offset[Vi] .= [sum(col .* segment_size[Ui]) for col in eachcol(P)]

    return dS, internal_diff_offset, segment_size_offset
end

function felzenszwalb_solve(g::GNNGraph)
    src, dst = edge_index(g)
    w = mean(sqrt.((g.x[:, src] .- g.x[:, dst]) .^ 2), dims=1)
    edge_order = sortperm(w, dims=2)
    w = w[edge_order]
    src, dst = src[edge_order], dst[edge_order]
    E = collect(zip(src, dst))
    N = g.num_nodes
    S = Matrix{Float64}(I, N, N)
    internal_diff = zeros(Float64, N)
    segment_size = ones(Float64, N)

    for t in 1:length(E)
        dS, internal_diff_offset, segment_size_offset = f(S, internal_diff, segment_size, t, w, E)
        S += dS
        internal_diff += internal_diff_offset
        segment_size += segment_size_offset

        if t % 100 == 0
            println("Iteration $t/$(length(E))")
        end
    end
    return S
end