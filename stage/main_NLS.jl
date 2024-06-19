using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, SparseArrays, QRMumps
using JSOSolvers, SolverBenchmark, Plots
using NLSProblems

# include("LM.jl")
include("LM.jl")
include("AuxiliaryFunctions.jl")

all_mgh = ["mgh02", "mgh06", "mgh08", "mgh10", "mgh15", "mgh16", "mgh17",  "mgh19", "mgh23", "mgh24", "mgh26", "mgh30", "mgh31", "mgh32", "mgh33", "mgh34"]

dict_solvers = Dict(
    # :LM_test => LM_test,
    # :LM_SPG => LM_SPG,
    # :LM_Zhu => LM_Zhu,
    # :LM_Andrei => LM_Andrei,
    # :LM_Andrei_λD => LM_Andrei_λD,
    # :LM_SPG_alt => LM_SPG_alt,
    # :LM_Zhu_alt => LM_Zhu_alt,
    :LM_Andrei_alt => LM_Andrei_alt,
    # :LM_Andrei_alt_λD => LM_Andrei_alt_λD,
    # :LM_SPG_quasi_nul_lin => LM_SPG_quasi_nul_lin,
    # :LM_Zhu_quasi_nul_lin => LM_Zhu_quasi_nul_lin,
    # :LM_Andrei_quasi_nul_lin => LM_Andrei_quasi_nul_lin,
    # :LM_Andrei_quasi_nul_lin_λD => LM_Andrei_quasi_nul_lin_λD,
    );
    
problems_names = setdiff(names(NLSProblems), [:NLSProblems]);
problems = (eval((problem))() for problem ∈ problems_names);
pb = collect(problems);
pb_sc = filter(problem -> problem.meta.ncon == 0, pb);
# pb_sc = filter(problem -> (problem.meta.name in all_mgh), pb);

###################### Test sur un problème unique #######################

compare_solvers(pb_sc[1], dict_solvers; type = "obj", save = false)

######################## Profils de performance #########################

# pp(dict_solvers, pb_sc; save = false)



####################### Générer tous les graphes ########################

# for k in eachindex(pb_sc)
#     compare_solvers(pb_sc[k], dict_solvers; type = "obj", save = true)
#     sleep(1.5)
# end

