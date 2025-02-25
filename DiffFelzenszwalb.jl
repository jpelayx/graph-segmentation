module DiffFelzenszwalb

using ChainRulesCore
using Flux
using Zygote: pullback
using GraphNeuralNetworks
using SparseArrays
using LinearAlgebra
using DataStructures: Stack, pop!, push!

export Felzenszwalb

"""
Records one step of the Felzenszwalb algorithm
"""
struct FelzenszwalbStep 
    t::Int
    ΔS::SparseMatrixCSC{Float64, Int}
    ΔInt::SparseVector{Float64, Int}
    Δsize::SparseVector{Float64, Int}
end

"""
Recording of an execution of the Felzenszwalb algorithm
"""
struct FelzenszwalbTape 
    stack::Stack{FelzenszwalbStep}   
end

FelzenszwalbTape() = FelzenszwalbTape(Stack{FelzenszwalbStep}())

import Base: isempty
function Base.isempty(tape::FelzenszwalbTape)
    return isempty(tape.stack)
end

function record!(tape::FelzenszwalbTape, ΔS, ΔInt, Δsize, t)
    ΔInt = sparsevec(ΔInt)
    Δsize = sparsevec(Δsize) 
    ΔS = sparse(ΔS)
    push!(tape.stack, FelzenszwalbStep(t, ΔS, ΔInt, Δsize))
end

function apply!(
    S::Matrix{Float64},
    internal_diff::Vector{Float64},
    segment_size::Vector{Float64},
    tape::FelzenszwalbTape
)
    step = pop!(tape.stack)
    I, J, V = findnz(step.ΔS)
    S[CartesianIndex.(zip(I,J))] .-= V
    I, V = findnz(step.ΔInt)
    internal_diff[I] .-= V
    I, V = findnz(step.Δsize)
    segment_size[I] .-= V
    return step.t    
end

"""
Felzenszwalb layer for a graph neural network
"""
struct Felzenszwalb <: GNNLayer
    k::Float64
    μ::Float64
    tol::Float64
    tape::FelzenszwalbTape
end

Flux.@layer Felzenszwalb

function Felzenszwalb(k::Float64, μ::Float64 = 1.0, tol::Float64 = 0.0)
    return Felzenszwalb(k, μ, tol, FelzenszwalbTape())
end

(l::Felzenszwalb)(g, x) = felzenszwalb_forward(l, g, x)

function compute_edge_weights(x, src, dst)
    return norm.(eachcol(x[:, src] .- x[:, dst]))
end

function felzenszwalb_forward(l::Felzenszwalb, g::GNNGraph, x::Matrix{Float64})
    src, dst = edge_index(g)

    w = compute_edge_weights(x, src, dst)
    order = sortperm(w)
    w = w[order]

    src, dst = src[order], dst[order]
    edges = collect(zip(src, dst))

    n = g.num_nodes

    S = Matrix{Float64}(I, n, n)
    internal_diff = zeros(n)
    segment_size = ones(n)

    S, _, _ = felzenszwalb_loop!(l, S, internal_diff, segment_size, w, edges)
    pooled_g, pooled_x = apply_segmentation(g, x, S)
    return pooled_g, pooled_x
end

function felzenszwalb_loop!(
    l::Felzenszwalb,
    S::Matrix{Float64},
    internal_diff::Vector{Float64},
    segment_size::Vector{Float64},
    w::Vector{Float64},
    edges::Vector{Tuple{Int, Int}}
)
    for t in 1:length(edges)
        ΔS, Δinternal_diff, Δsize = felzenszwalb_step(S, internal_diff, segment_size, t, w, edges, l)
        if ΔS === nothing
            continue
        end
        S += ΔS
        internal_diff += Δinternal_diff
        segment_size += Δsize

        record!(l.tape, ΔS, Δinternal_diff, Δsize, t)
        if t % 100 == 0
            println("Iteration $t/$(length(edges))")
        end
    end
    return S, internal_diff, segment_size
end

function ChainRulesCore.rrule(::typeof(felzenszwalb_loop!), 
    l::Felzenszwalb, 
    S::Matrix{Float64},
    internal_diff::Vector{Float64},
    segment_size::Vector{Float64},
    w::Vector{Float64},
    edges::Vector{Tuple{Int, Int}}
)
    function back(∇)
        return (
            NoTangent(), 
            felzenszwalb_reverse!(l, S, internal_diff, segment_size, w, edges, ∇)...
        )
    end
    return felzenszwalb_loop!(l,S,internal_diff,segment_size,w,edges), back
end

function felzenszwalb_reverse!(
    l::Felzenszwalb,
    S::Matrix{Float64},
    internal_diff::Vector{Float64},
    segment_size::Vector{Float64},
    w::Vector{Float64},
    edges::Vector{Tuple{Int, Int}},
    ∇::Any
)
    ∇S, _, _ = ∇
    ∇Int = zeros(size(internal_diff)...)
    ∇size = zeros(size(segment_size)...)
    ∇w = zeros(size(w)...)

    tape = l.tape
    t = length(edges)
    while !isempty(tape)
        t = apply!(S, internal_diff, segment_size, tape)
        _, back = pullback(felzenszwalb_step, S, internal_diff, segment_size, t, w, edges, l)
        δS, δInt, δsize, _, δw, _, _ = back((∇S, ∇Int, ∇size))
        ∇w .+= -δw
        δS, δInt, _ = δS
        ∇S .+= -δS
        ∇Int .+= -δInt
        ∇size .+= -δsize
        if t % 100 == 0
            println("Iteration $t/$(length(edges))")
        end
    end
    return (NoTangent(), ∇S, ∇Int, NoTangent(), ∇w, NoTangent())
end

function felzenszwalb_step(S, internal_diff, segment_size, t, w, edges,l)
    weight = w[t]
    v, u = edges[t]
    Vi = findall(x -> x > l.tol, S[v, :])
    Ui = findall(x -> x > l.tol, S[u, :])

    V = @view S[:, Vi]
    U = @view S[:, Ui]

    P = merge_probability(Vi, V, Ui, U, internal_diff, segment_size, v, u, weight, l)
    @ignore_derivatives if !any(p -> p > l.tol, P)
        return nothing, nothing, nothing
    end    
    
    dU = U * Diagonal(vec(sum(P, dims=2)))
    adjust_u!(dU, U, u)

    dV = (1 .- V) .* (U * P)
    adjust_v!(dV, v)

    dS = make_dS(dV, Vi, -dU, Ui)

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

function apply_segmentation(g, x, S)
    S = S[:, vec(sum(S, dims=1)) .> 0.0]
    pooled_x = S' * x'
    A = sparse(S' * adjacency_matrix(g) * S)
    pooled_g = GNNGraph(A)
    return pooled_g, pooled_x
end

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
    v, u, weight,
    l
)
    τ(Vi) = l.k ./ segment_size[Vi]
    MInt = minimum.(
        Iterators.product(
            internal_diff[Ui] .+ τ(Ui),
            internal_diff[Vi] .+ τ(Vi)
        )
    )
    Mij_conditional = tanh.((MInt .- weight) .* l.μ)

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
end 
