"""
This is the NODE-like implementation of the Felzenszwalb algorithm.
"""

using Zygote
using Zygote: @adjoint
using SparseArrays
using GraphNeuralNetworks
using NNlib: σ, tanh, tanh_fast

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
    return (δP, Δ[2], Δ[3])
end

function merge_probability(
    Vi, V, Ui, U,
    internal_diff,
    segment_size,
    v, u, weight
)
    τ(V_I, k) = k ./ segment_size[V_I]
    MInt = minimum.(
        Iterators.product(internal_diff[Ui] .+ τ(Ui, k),
                          internal_diff[Vi] .+ τ(Vi, k)))
    Mij_conditional = tanh_fast.((MInt .- weight) .* μ)

    if !isempty(Vi ∩ Ui)
        clear_intersections!(Mij_conditional, Vi, Ui)
    end

    V_K = nonzeros(V[v, :])
    U_K = nonzeros(U[u, :])
    Mij = max.(Mij_conditional .* (U_K * V_K'), 0.0)
    return Mij
end

function f(S, internal_diff, segment_size, t, w, E)
    weight = w[t]
    v, u = E[t]
    Vi = S[v, :].nzind
    Ui = S[u, :].nzind
    V = @view S[:, Vi]
    U = @view S[:, Ui]

    P = merge_probability(Vi, V, Ui, U, internal_diff, segment_size, v, u, weight)

    V, U = eachcol(V), eachcol(U)

    dU = eachcol(S[:, Ui]) .* sum(P, dims=2)
    adjust_u!(dUi, Ui, u) = dUi[u] *= Ui[u]
    adjust_u!.(dU, U, u)

    get_offset(MtoC) = sparsevec(
        vcat(rowvals.(U)...),
        vcat(MtoC .* nonzeros.(U)...), S.n
    )

    notC(C, Ci) = sparsevec(
        vcat(Ci, rowvals(C[Ci])),
        vcat(ones(Float64, length(Ci)), -nonzeros(C[Ci])),
        S.n
    )
    merged_to_V = get_offset.(eachcol(P))
    notV = notC.(V, rowvals.(merged_to_V))
    dV = multsp.(notV, merged_to_V)
    adjust_v!(dVi, v) = dVi[v] = 0.0
    adjust_v!.(dV, v)

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
    internal_diff_offset = zeros(S.n)
    internal_diff_offset[Vi] = (1 .- Mi) .* internal_diff[Vi] + Mi .* weight

    segment_size_offset = zeros(S.n)
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
    S = sparse(1:N, 1:N, ones(Float32, N), N, N)
    internal_diff = zeros(Float32, N)
    segment_size = ones(Float32, N)

    for t in 1:length(E)
        dS, internal_diff_offset, segment_size_offset = f(S, internal_diff, segment_size, t, w, E)
        S += dS
        internal_diff += internal_diff_offset
        segment_size += segment_size_offset

        droptol!(S, min_prob)
        if t % 100 == 0
            println("Iteration $t/$(length(E))")
        end
    end
    return S
end