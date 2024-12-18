include("node_felzenszwalb.jl")

using ChainRulesTestUtils

ChainRulesCore.debug_mode() = true

# Test the clear_intersections! rule

P = rand(10,10)
I = collect(1:10)
J = collect(5:14)

test_rrule(clear_intersections!, P, I, J)

# Test adjust_u! rule
dU = rand(2,3)
U = rand(2,3)
i = 1

test_rrule(adjust_u, dU, U, i)