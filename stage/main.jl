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


pb_test1 = pb_sc[1]
pb_test2 = pb_sc[7]


obj, grad, stats = LM(pb_test1)
rangs = 1:length(grad)
obj_D, grad_D, stats_D = LM_D(pb_test1)
rangs_D = 1:length(grad_D)

norm_obj = [norm(objectif) for objectif ∈ obj]
norm_obj_D = [norm(objectif) for objectif ∈ obj_D]

plot(rangs, norm_obj, xlabel="k", ylabel="‖J'.F‖",yaxis =:log10, label="LM classique")
plot!(rangs_D, norm_obj_D, label="LM avec D")


######################## Profils de performance #########################


# problems_names = setdiff(names(NLSProblems), [:NLSProblems])
# problems = (eval((problem))() for problem ∈ problems_names)

# solvers = Dict(:LM_D => LM_D)
# stats = bmark_solvers(solvers, problems)








