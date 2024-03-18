using ADNLPModels, JSOSolvers, NLSProblems, SolverBenchmark, Plots
include("LM.jl")

############################ Test Himmelblau #############################

# FH(x) = [x[2]+x[1].^2-11, x[1]+x[2].^2-7]
# x0H = [10., 20.]
# himmelblau_nls = ADNLSModel(FH,x0H,2)

# objectif, gradient, stats = LM_D(himmelblau_nls, himmelblau_nls.meta.x0, 1e-10, 1e-10)
# @test stats.status == :first_order


###################### Test sur un problème unique #######################

problems_names = setdiff(names(NLSProblems), [:NLSProblems])
problems = (eval((problem))() for problem ∈ problems_names)
pb = collect(problems)
pb_sc = filter(problem -> problem.meta.ncon == 0, pb)
pb_test1 = filter(problem -> problem.meta.name == "tp272", pb)

n = 15
pb_test = pb_sc[n]
pb_testD = pb_sc[n]
pb_testZ = pb_sc[n]
pb_testA = pb_sc[n]

stats, obj, grad = LM_D(pb_test1[1]; ApproxD = false)
rangs = 1:length(grad)
# stats_D, obj_D, grad_D = LM_D(pb_test1[1]; fctDk = SPG)
# rangs_D = 1:length(grad_D)
# stats_Z, obj_Z, grad_Z = LM_D(pb_test1[1]; fctDk = Zhu)
# rangs_Z = 1:length(grad_Z)
stats_A, obj_A, grad_A = LM_D(pb_test1[1]; fctDk = Andrei)
rangs_A = 1:length(grad_A)

plot(rangs, grad, xlabel="k", ylabel="‖JᵀF‖",yaxis =:log10, label="LM classique")
# plot!(rangs_D, grad_D, label="LM avec D (SPG)")
# plot!(rangs_Z, grad_Z, label="LM avec D (Z)")
plot!(rangs_A, grad_A, label="LM avec D (A)")
# savefig("obj_mgh03.png")


######################## Profils de performance #########################


# problems_names = setdiff(names(NLSProblems), [:NLSProblems])
# problems = (eval((problem))() for problem ∈ problems_names)

# solvers = Dict(
#   :LM => model -> LM_D(model; ApproxD = false),
#   :LM_SPG => model -> LM_D(model; fctDk = SPG),
#   :LM_Andrei => model -> LM_D(model; fctDk = Andrei)
# )
# stats = bmark_solvers(solvers, problems)

# cols = [:name, :status, :objective, :elapsed_time, :iter]
# for solver ∈ keys(solvers)
#   pretty_stats(stats[solver][!, cols])
# end
# cost(df) = (df.status .!= :first_order) * Inf + df.iter
# performance_profile(stats, cost)

# # #savefig("performance_profile.png")