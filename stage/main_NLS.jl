# using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, SparseArrays, QRMumps
# using JSOSolvers, SolverBenchmark, Plots
# using NLSProblems

include("LM.jl")
include("AuxiliaryFunctions.jl")
include("test.jl")




dict_solvers = Dict(
    # :LM_wo_D => LM_wo_D,
    :LM_test => LM_test,
    # :LM_D_y_diese => LM_D_y_diese,
    # :LM_D_y_tilde => LM_D_y_tilde,
    :LM_SPG => LM_SPG,
    # :LM_Zhu => LM_Zhu,
    :LM_Andrei => LM_Andrei,
    :LM_SPG_alt => LM_SPG_alt,
    # :LM_Zhu_alt => LM_Zhu_alt,
    :LM_Andrei_alt => LM_Andrei_alt
    )

    
problems_names = setdiff(names(NLSProblems), [:NLSProblems])
problems = (eval((problem))() for problem ∈ problems_names)
pb = collect(problems)
pb_sc = filter(problem -> problem.meta.ncon == 0, pb)
# pb_sc = filter(problem -> problem.meta.name == "mgh10", pb)

###################### Test sur un problème unique #######################

# compare_solvers(pb_sc[1], dict_solvers; type = "obj", save = true)

######################## Profils de performance #########################

# pp(dict_solvers, problems; save = true)



####################### Générer tous les graphes ########################

for k =1:length(pb_sc)
    compare_solvers(pb_sc[k], dict_solvers; type = "obj", save = true)
end
