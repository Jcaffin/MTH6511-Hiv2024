using ADNLPModels, JSOSolvers, NLSProblems, SolverBenchmark
include("LM.jl")
include("solvers.jl")

FH(x) = [x[2]+x[1].^2-11, x[1]+x[2].^2-7]
x0H = [10., 20.]
himmelblau_nls = ADNLSModel(FH,x0H,2)

stats = lm_param(himmelblau_nls, himmelblau_nls.meta.x0, 1e-6, 1e-6)
@test stats.status == :first_order


