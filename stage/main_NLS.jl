# using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, NLPModelsIpopt, SparseArrays, QRMumps
# using JSOSolvers, SolverBenchmark, Plots, NLSProblems, SparseMatricesCOO
# using Dates, DataFrames, PrettyTables, JLD2
# using OptimizationProblems, OptimizationProblems.ADNLPProblems, NLPModelsModifiers

include("LM.jl")
include("AuxiliaryFunctions.jl")

dict_solvers = Dict(
    # :hess_exact => hess_exact,
    # :LM => LM,
    :LM_SPG => LM_SPG,
    # :LM_Zhu => LM_Zhu,
    # :LM_Andrei => LM_Andrei,
    :LM_SPG_λD => LM_SPG_λD,
    # :LM_Zhu_λD => LM_Zhu_λD,
    # :LM_Andrei_λD => LM_Andrei_λD,
    # :LM_SPG_alt => LM_SPG_alt,
    # :LM_Zhu_alt => LM_Zhu_alt,
    :LM_Andrei_alt => LM_Andrei_alt,
    :LM_SPG_alt_λD => LM_SPG_alt_λD,
    # :LM_Zhu_alt_λD => LM_Zhu_alt_λD,
    # :LM_Andrei_alt_λD => LM_Andrei_alt_λD,
    # :LM_SPG_quasi_nul_lin => LM_SPG_quasi_nul,
    # :LM_Zhu_quasi_nul_lin => LM_Zhu_quasi_nul,
    # :LM_Andrei_quasi_nul_lin => LM_Andrei_quasi_nul,
    # :LM_SPG_quasi_nul_λD => LM_SPG_quasi_nul_λD,
    # :LM_Zhu_quasi_nul_λD => LM_Zhu_quasi_nul_λD,
    # :LM_Andrei_quasi_nul_λD => LM_Andrei_quasi_nul_λD,
    # :test_SPG => test_SPG,
    # :test_Zhu => test_Zhu,
    # :test_Andrei => test_Andrei,
    # :test_SPG_λD => test_SPG_λD,
    # :test_Zhu_λD => test_Zhu_λD,
    # :test_Andrei_λD => test_Andrei_λD,
    # :test_SPG_alt => test_SPG_alt,
    # :test_Zhu_alt => test_Zhu_alt,
    # :test_Andrei_alt => test_Andrei_alt,
    # :test_SPG_alt_λD => test_SPG_alt_λD,
    # :test_Zhu_alt_λD => test_Zhu_alt_λD,
    # :test_Andrei_alt_λD => test_Andrei_alt_λD,
    );
    
# problems_names = setdiff(names(NLSProblems), [:NLSProblems]);
# problems = (eval((problem))() for problem ∈ problems_names);
# pb = collect(problems);
# pb_sc = filter(problem -> problem.meta.ncon == 0, pb);
# pb_mgh = [NLSProblems.mgh02(), NLSProblems.mgh06(), NLSProblems.mgh08(), NLSProblems.mgh10(), NLSProblems.mgh15(), NLSProblems.mgh16(), NLSProblems.mgh17(), NLSProblems.mgh19(), NLSProblems.mgh23(), NLSProblems.mgh23(20), NLSProblems.mgh24(10), NLSProblems.mgh26(), NLSProblems.mgh30(), NLSProblems.mgh31(), NLSProblems.mgh31(20), NLSProblems.mgh32(), NLSProblems.mgh33(), NLSProblems.mgh33(20), NLSProblems.mgh34(), NLSProblems.mgh34(20)];

# function get_model_function(path::String)
#     parts = split(path, '.')
#     mod = @eval Main
#     for part in parts
#         mod = getproperty(mod, Symbol(part))
#     end
#     return mod
# end
# eq_problem_names = OptimizationProblems.meta[OptimizationProblems.meta.has_equalities_only, :name];
# OP_problem_names = ["OptimizationProblems.ADNLPProblems.$name" for name in eq_problem_names];
# eq_problems      = [get_model_function(problem)() for problem in OP_problem_names];
# eq_nls_problems  = [FeasibilityResidual(problem) for problem in eq_problems];

###################### Test sur un problème unique #######################
# compare_solvers(pb_mgh[1], dict_solvers; type = "obj", save = false)

######################## Profils de performance #########################
# pp(dict_solvers, eq_nls_problems; save_stats = false)

####################### Générer tous les graphes ########################
# for k in eachindex(pb_sc)
#     compare_solvers(pb_sc[k], dict_solvers; type = "obj", save = true)
#     sleep(1.5)
# end


write_solver_df_to_jld2(mgh02(), "hoho"; is_LM = true)
