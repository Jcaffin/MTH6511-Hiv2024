# using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, SparseArrays, QRMumps
# using JSOSolvers, SolverBenchmark, Plots, NLSProblems, SparseMatricesCOO
# using Dates, DataFrames, PrettyTables

include("LM.jl")
include("AuxiliaryFunctions.jl")

nls = mgh01()
x = nls.meta.x0

Fx = residual(nls, x)
Jx = jac_residual(nls, x)
Hx = hess(nls, x)
g = Jx' * Fx

λ = 1

A = sparse(Hx + λ*I)
b = -g

QR = qr(A)
d = QR\(b)

M = Ma97(A)