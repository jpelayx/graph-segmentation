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
using CUDA
using CUDA.CUSPARSE
using Flux
using Adapt: @adapt_structure

"""
d/dt h(t) = f(h(t), t, θ)
h(t+1) = h(t) + 1 * f(h(t), t, θ)
"""

k = 25.0
μ = 3.0
min_prob = 1e-2

include("evaluate.jl")

mutable struct Segmentation
    S::Matrix{Float64}
    internal_diff::Vector{Float64}
    segment_size::Vector{Float64}
end

@adapt_structure Segmentation

Segmentation(N::Int) = Segmentation(Matrix{Float64}(I,N,N), zeros(N), ones(N))

struct FelzenszwalbStep 
    t::Int
    ΔS
    ΔInt
    Δsize
end 

struct FelzenszwalbTape 
    stack::Stack{FelzenszwalbStep}   
end

FelzenszwalbTape() = FelzenszwalbTape(Stack{FelzenszwalbStep}())

import Base: isempty
function Base.isempty(tape::FelzenszwalbTape)
    return isempty(tape.stack)
end

function record!(tape::FelzenszwalbTape, ΔS, ΔInt, Δsize, t)
    @CUDA.allowscalar ΔInt = sparsevec(ΔInt)
    if isempty(ΔInt.nzval)
        return 
    end 
    @CUDA.allowscalar Δsize = sparsevec(Δsize) 
    ΔInt = CuSparseVector(ΔInt)
    Δsize = CuSparseVector(Δsize)
    ΔS = CuSparseMatrixCSR(ΔS)
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

function clear_intersections!(P, Vi, Ui)
    intersections = Vi ∩ Ui
    if !isempty(intersections)
        is = something.(indexin(intersections, Vi))
        js = something.(indexin(intersections, Ui))
        CUDA.@allowscalar P[js, is] .= 0.0
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

function merge_probability(S, internal_diff, segment_size, v, u, weight, k, μ, ϵ=1e-6, auxmem=nothing)
    ignore_derivatives() do 
        P = auxmem.P
        V = auxmem.V
        U = auxmem.U
        condition = auxmem.τ
    end 
    
    condition = internal_diff .+ (k ./ segment_size)
    V = - condition * auxmem.N1'
    U = - auxmem.N1 * condition'
    
    P = (V .+ U .+ sqrt.((V .- U) .^ 2 .+ ϵ)) ./ 2
    P = tanh.((P .- weight) .* μ)
    P = P .* (S[u,:] * S[v,:]')
      
    return P
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
    dS = zeros(N,N) |> cu
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


function f(
    S, 
    internal_diff, 
    segment_size,
    t, w, E,
    tape=nothing
    auxmem=nothing
)
    weight = w[t]
    v, u = E[t]

    P = merge_probability(S, internal_diff, segment_size, v, u, weight, k, μ, auxmem)
    
    U = S * Diagonal(vec(sum(P, dims=2)))
    adjust_u!(U, S, u)

    V = (1 .- S) .* (S * P)
    adjust_v!(V, v)

    dS = U .+ V

    Mi = sum(P, dims=1)'
    internal_diff_offset = (1 .- Mi) .* internal_diff .+ Mi .* weight - internal_diff
    # internal_diff_offset = zeros(size(S)[1]) |> cu 
    # fillvec!(internal_diff_offset, 
    #     Vi, 
    #     ((1 .- Mi) .* internal_diff[Vi] .+ Mi .* weight) - internal_diff[Vi]
    # )

    segment_size_offset = [sum(col .* segment_size) for col in eachcol(P)] |> cu
    # segment_size_offset = zeros(size(S)[1]) |> cu
    # fillvec!(segment_size_offset, 
    #     Vi, 
    #     [sum(col .* segment_size[Ui]) for col in eachcol(P)] |> cu
    # )
    
    @ignore_derivatives if tape !== nothing
        record!(tape, dS, internal_diff_offset, segment_size_offset, t)
    end

    return dS, internal_diff_offset, segment_size_offset
end

global P = CuArray{Float64}(undef, N, N)
global V = CuArray{Float64}(undef, N, N)
global U = CuArray{Float64}(undef, N, N)
global NOnes = CUDA.ones(Float64, N)
global tau = CuArray{Float64}(undef, N)



function felzenszwalb_solve(G::GNNGraph)
    src, dst = edge_index(G)
    
    w = mean(sqrt.((G.x[:, src] .- G.x[:, dst]) .^ 2), dims=1)
    edge_order = sortperm(w, dims=2)
    w = w[edge_order] |> cpu 

    src, dst = src[edge_order], dst[edge_order]    
    E = collect(zip(src, dst))

    N = G.num_nodes
    S = CuArray{Float64}(I,N,N) 
    internal_diff = CUDA.zeros(N) 
    segment_size = CUDA.ones(N) 

    # allocating memory for ops 
    auxmem = (
        dS = CuArray{Float64}(undef, N, N),
        internal_diff_offset = CuArray{Float64}(undef, N),
        segment_size_offset = CuArray{Float64}(undef, N),
        P = CuArray{Float64}(undef, N, N),
        V = CuArray{Float64}(undef, N, N),
        U = CuArray{Float64}(undef, N, N),
        N1 = CUDA.ones(Float64, N),
        τ = CuArray{Float64}(undef, N)
    )

    tape = FelzenszwalbTape() |> cu

    for t in 1:length(E)
        dS, internal_diff_offset, segment_size_offset = f(S, internal_diff, segment_size, t, w, E, tape, auxmem)
        S += dS
        internal_diff += internal_diff_offset
        segment_size += segment_size_offset
        if t % 100 == 0
            println("Iteration $t/$(length(E))")
        end
    end
    return S, internal_diff, segment_size, tape
end

function felzenszwalb_reverse(
    G::GNNGraph,
    S::Matrix{Float64},
    internal_diff::Vector{Float64}, 
    segment_size::Vector{Float64},
    tape::FelzenszwalbTape,
    ∇::Any
)
    src, dst = edge_index(G)
    
    w = mean(sqrt.((G.x[:, src] .- G.x[:, dst]) .^ 2), dims=1)
    edge_order = sortperm(w, dims=2)
    w = w[edge_order]
    Δw = zeros(size(w)) 

    src, dst = src[edge_order], dst[edge_order]    
    E = collect(zip(src, dst))
     
    t = length(E) 
    ∇S, ∇Int, ∇size = ∇
    while !isempty(tape) 
        t = apply!(S, internal_diff, segment_size, tape)
        print(t)
        _, back = pullback(f, S, internal_diff, segment_size, t, w, E)
        δS, δInt, δsize,  _, δw, _   = back(∇)
        Δw .+= -δw
        ∇S .+= -δS
        ∇Int .+= -δInt
        ∇size .+= -δsize
        ∇ = (∇S, ∇Int, nothing)
    end
    return Δw
end

function compute_edge_weights(x, edge_index)
    w = sum(sqrt.((x[:, edge_index[1]] .- x[:, edge_index[2]]) .^ 2), dims=1)./ndims(x)
    return w
end

G = G |> cu
S, internal_diff, segment_size, tape = felzenszwalb_solve(G);
S = S  |> cu
internal_diff = internal_diff |> cu
segment_size = segment_size |> cu
tape = tape |> cu
N, _ = size(S)
∇ = (rand(N,N) |> cu, rand(N) |> cu, rand(N) |> cu)
Δw = felzenszwalb_reverse(G, S, internal_diff, segment_size, tape, ∇)