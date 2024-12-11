"""
This is the NODE-like implementation of the Felzenszwalb algorithm.
"""

using Zygote 
using SparseArrays
using GraphNeuralNetworks
using NNlib: σ, tanh, tanh_fast

"""
d/dt h(t) = f(h(t), t, θ)
h(t+1) = h(t) + 1 * f(h(t), t, θ)
"""

k = 2.0
μ = 2.0
min_prob = 1e-2

multsp(A, B) = A .* B

get_segments(S::SparseMatrixCSC{Float64, Int64}, cs::AbstractVector) = @view S[:, cs]

get_segments_from_vertices(S::SparseMatrixCSC{Float64, Int64}, vi::Int) =
     get_segments(S, S[vi,:].nzind), S[vi,:].nzind

function merge_probability(
    V_I, V, U_I, U, 
    internal_diff, 
    segment_size, 
    v, u, w 
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

function f(S, internal_diff, segment_size, t, w, E)
    weigth = w[t]
    v, u = E[t]
    Vi = S[v, :].nzind
    Ui = S[u, :].nzind
    V = eachcol(@view S[:, Vi])
    U = eachcol(@view S[:, Ui])
    
    P = merge_probability(Vi, V, Ui, U, internal_diff, segment_size, v, u, weigth)
    
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