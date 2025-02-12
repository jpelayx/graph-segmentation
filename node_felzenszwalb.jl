"""
This is the NODE-like implementation of the Felzenszwalb algorithm.
"""

using ChainRulesCore
using Zygote 
using Zygote: @adjoint
using GraphNeuralNetworks
using NNlib: σ, tanh, tanh_fast, relu
using LinearAlgebra
using SparseArrays
using DataStructures: Stack, pop!, push! 

"""
d/dt h(t) = f(h(t), t, θ)
h(t+1) = h(t) + 1 * f(h(t), t, θ)
"""

k = 25.0
μ = 3.0
min_prob = 1e-2

include("evaluate.jl")

struct FelzenszwalbStep 
    t::Int
    δS::SparseMatrixCSC{Float64, Int}
end 

struct FelzenszwalbTape 
    stack::Stack{FelzenszwalbStep}   
end

FelzenszwalbTape() = FelzenszwalbTape(Stack{FelzenszwalbStep}())

function record!(tape::FelzenszwalbTape, δS, t)
    push!(tape.stack, FelzenszwalbStep(t, sparse(δS)))
end

function apply!(S, tape::FelzenszwalbTape)
    step = pop!(tape.stack)
    I, J, V = findnz(step.δS)
    S[CartesianIndex.(zip(I,J))] .-= V
    return step.t
end

struct Segmentation
    S::Matrix{Float64}
    internal_diff::Vector{Float64}
    segment_size::Vector{Float64}
end

Segmentation(N::Int) = Segmentation(Matrix{Float64}(I,N,N), zeros(N), ones(N))

function clear_intersections!(P, Vi, Ui)
    intersections = Vi ∩ Ui
    if !isempty(intersections)
        is = indexin(intersections, Vi)
        js = indexin(intersections, Ui)
        P[js, is] .= 0.0
    end
    return P
end

function ChainRulesCore.rrule(::typeof(clear_intersections!), P, Vi, Ui)
    function pullback(δP)
        intersections = Vi ∩ Ui
        if !isempty(intersections)
            is = indexin(intersections, Vi)
            js = indexin(intersections, Ui)
            δP[js, is] .= 0.0
        end
        return (NoTangent(), δP, NoTangent(), NoTangent())
    end
  
    return clear_intersections!(P, Vi, Ui), pullback
end


function merge_probability(
    Vi, V, Ui, U,
    internal_diff,
    segment_size,
    v, u, weight
)
    τ(Vi, k) = k ./ segment_size[Vi]
    MInt = minimum.(
        Iterators.product(
            internal_diff[Ui] .+ τ(Ui, k),
            internal_diff[Vi] .+ τ(Vi, k)
        )
    )
    Mij_conditional = tanh.((MInt .- weight) .* μ)

    clear_intersections!(Mij_conditional, Vi, Ui)

    Mij = relu.(Mij_conditional .* (U[u,:] * V[v,:]'))
    return Mij
end


function adjust_u!(dU, U, i) 
    dU[i, :] .*= U[i, :]
    return dU
end 

function ChainRulesCore.rrule(::typeof(adjust_u!), dU, U, i)
    function adjust_u!_pullback(ΔU)
        ΔU = unthunk(ΔU)

        δU = zeros(size(ΔU))
        δU[i,:] .= ΔU[i,:] .* dU[i,:]
        
        δdU = ΔU
        δdU[i,:] .*= U[i,:]
        
        return(NoTangent(), δdU, δU, NoTangent())        
    end
    return adjust_u!(dU, U, i), adjust_u!_pullback
end


function adjust_v!(dV, i) 
    dV[i,:] .= 0.0
    return dV
end

function ChainRulesCore.rrule(::typeof(adjust_v!), dV, i)
    function adjust_v!_pullback(ΔV)
        δV = unthunk(ΔV)
        δV[i,:] .= 0.0 
        return (NoTangent(), δV, NoTangent())
    end
    return adjust_v!(dV, i), adjust_v!_pullback
end


function fill_S!(dS, I, dI)
    dS[:, I] .+= dI
    return dS
end

function ChainRulesCore.rrule(::typeof(fill_S!), dS, I, dI)
    function fill_S!_pullback(ΔS)
        δdS = unthunk(ΔS)
        δdI = δdS[:, I]
        return (NoTangent(), δdS, NoTangent(), δdI)
    end
    return fill_S!(dS, I, dI), fill_S!_pullback    
end

function make_dS(dV, Vi, dU, Ui)
    N = size(dV)[1]
    dS = zeros(N,N)
    dS[:, Vi] .= dV
    dS[:, Ui] .+= dU
    return dS
end

function ChainRulesCore.rrule(::typeof(make_dS), dV, Vi, dU, Ui)
    function make_dS_pullback(Δ)
        ΔdS = unthunk(Δ)
        δdV = ΔdS[:, Vi]
        δdU = ΔdS[:, Ui]
        return (NoTangent(), δdV, NoTangent(), δdU, NoTangent())
    end
    return make_dS(dV, Vi, dU, Ui), make_dS_pullback
end

function fillvec!(v, I, dI)
    v[I] .= dI
    return v
end

function ChainRulesCore.rrule(::typeof(fillvec!), v, I, dI)
    function fillvec!_pullback(Δ)
        δv = unthunk(Δ)
        δdI = δv[I]
        return (NoTangent(), δv, NoTangent(), δdI)
    end
    return fillvec!(v, I, dI), fillvec!_pullback
end


function f(S, internal_diff, segment_size, t, w, E, tape=nothing)
    weight = w[t]
    v, u = E[t]
    Vi = findall(x -> x > 0.0, S[v, :])
    Ui = findall(x -> x > 0.0, S[u, :])
    V = @view S[:, Vi]
    U = @view S[:, Ui]

    P = merge_probability(Vi, V, Ui, U, internal_diff, segment_size, v, u, weight)
    
    dU = U * Diagonal(vec(sum(P, dims=2)))
    adjust_u!(dU, U, u)

    dV = (1 .- V) .* (U * P)
    adjust_v!(dV, v)

    dS = make_dS(dV, Vi, -dU, Ui)
    if tape !== nothing
        record!(tape, dS, t)
    end

    Mi = sum(P, dims=1)'
    internal_diff_offset = zeros(size(S)[1])
    fillvec!(internal_diff_offset, 
        Vi, 
        ((1 .- Mi) .* internal_diff[Vi] .+ Mi .* weight) - internal_diff[Vi]
    )

    segment_size_offset = zeros(size(S)[1])
    fillvec!(segment_size_offset, 
        Vi, 
        [sum(col .* segment_size[Ui]) for col in eachcol(P)]
    )

    return dS, internal_diff_offset, segment_size_offset
end

# S, segment_hash, segment_size, internal_diff = initialize_structures(N)
function felzenszwalb_solve(G::GNNGraph)
    src, dst = edge_index(G)
    
    w = mean(sqrt.((G.x[:, src] .- G.x[:, dst]) .^ 2), dims=1)
    edge_order = sortperm(w, dims=2)
    w = w[edge_order]

    src, dst = src[edge_order], dst[edge_order]    
    E = collect(zip(src, dst))

    num_nodes = N = G.num_nodes
    num_segments = num_nodes 
    S = Matrix{Float64}(I, num_nodes, num_segments)
                        #  rows       cols
    
    internal_diff = zeros(Float64, N)
    segment_size = ones(Float64, N)

    tape = FelzenszwalbTape()

    for t in 1:length(E)
        dS, internal_diff_offset, segment_size_offset = f(S, internal_diff, segment_size, t, w, E, tape)
        S += dS
        internal_diff += internal_diff_offset
        segment_size += segment_size_offset
        if t % 100 == 0
            println("Iteration $t/$(length(E))")
        end
    end
    return Segmentation(S, internal_diff, segment_size), tape
end

function felzenszwalb_reverse(
    G::GNNGraph,
    S::Segmentation, 
    tape::FelzenszwalbTape,
    ΔS::Matrix{Float64} 
)
    src, dst = edge_index(G)
    
    w = mean(sqrt.((G.x[:, src] .- G.x[:, dst]) .^ 2), dims=1)
    edge_order = sortperm(w, dims=2)
    w = w[edge_order]

    src, dst = src[edge_order], dst[edge_order]    
    E = collect(zip(src, dst))

    num_nodes = N = G.num_nodes
    num_segments = num_nodes 
     
    t = length(E) 
    while t > 1 
        # S(t-1) = S(t) - δS(t)
        t = apply!(S.S, tape)
        δS, _, _, δt, δw, _, _ = gradient(
            f, 
            S.S,
            S.internal_diff,
            S.segment_size,
            t,
            w,
            E
        )
    end


end
