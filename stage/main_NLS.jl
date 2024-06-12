using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, SparseArrays, QRMumps
using JSOSolvers, SolverBenchmark, Plots
using NLSProblems

# include("LM.jl")
include("LM.jl")
include("AuxiliaryFunctions.jl")

dict_solvers = Dict(
    # :LM_test => LM_test,
    # :LM_SPG => LM_SPG,
    # :LM_Zhu => LM_Zhu,
    :LM_Andrei => LM_Andrei,
    :LM_Andrei_qrmumps => LM_Andrei_qrmumps,
    # :LM_SPG_alt => LM_SPG_alt,
    # :LM_Zhu_alt => LM_Zhu_alt,
    # :LM_Andrei_alt => LM_Andrei_alt,
    # :LM_Andrei_alt_qrmumps => LM_Andrei_alt_qrmumps,
    # :LM_SPG_quasi_nul_lin => LM_SPG_quasi_nul_lin,
    # :LM_Zhu_quasi_nul_lin => LM_Zhu_quasi_nul_lin,
    # :LM_Andrei_quasi_nul_lin => LM_Andrei_quasi_nul_lin,
    # :LM_Andrei_quasi_nul_lin_qrmumps => LM_Andrei_quasi_nul_lin_qrmumps,
    );
    
problems_names = setdiff(names(NLSProblems), [:NLSProblems]);
problems = (eval((problem))() for problem ∈ problems_names);
pb = collect(problems);
# pb_sc = filter(problem -> (problem.meta.ncon == 0), pb);
pb_sc = filter(problem -> (problem.meta.name == "tp293"), pb);
# LM_D(pb_sc[1]; verbose = true)

###################### Test sur un problème unique #######################

compare_solvers(pb_sc[1], dict_solvers; type = "obj", save = false)

######################## Profils de performance #########################

# pp(dict_solvers, problems; save = false)



####################### Générer tous les graphes ########################

# for k in eachindex(pb_sc)
#     compare_solvers(pb_sc[k], dict_solvers; type = "obj", save = true)
#     sleep(1.5)
# end

# 0.00376163450836752
# 0.00016248756554137