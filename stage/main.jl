# using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, NLPModelsIpopt, SparseArrays, QRMumps
# using JSOSolvers, SolverBenchmark, Plots, NLSProblems, SparseMatricesCOO
# using Dates, DataFrames, PrettyTables, JLD2
# using OptimizationProblems, OptimizationProblems.ADNLPProblems, NLPModelsModifiers, BenchmarkProfiles

include("LM.jl")
include("AuxiliaryFunctions.jl")  # sert pour ma fonction pp (performance profile)

dict_solvers = Dict(
    :hess_exact => hess_exact,
    :LM => LM,
    # :LM_SPG => LM_SPG,
    # :LM_Zhu => LM_Zhu,
    # :LM_Andrei => LM_Andrei,
    # :LM_SPG_λD => LM_SPG_λD,
    # :LM_Zhu_λD => LM_Zhu_λD,
    # :LM_Andrei_λD => LM_Andrei_λD,
    :LM_SPG_alt => LM_SPG_alt,
    # :LM_Zhu_alt => LM_Zhu_alt,
    # :LM_Andrei_alt => LM_Andrei_alt,
    # :LM_SPG_alt_λD => LM_SPG_alt_λD,
    # :LM_Zhu_alt_λD => LM_Zhu_alt_λD,
    # :LM_Andrei_alt_λD => LM_Andrei_alt_λD,
    :LM_SPG_quasi_nul => LM_SPG_quasi_nul,
    # :LM_Zhu_quasi_nul => LM_Zhu_quasi_nul,
    # :LM_Andrei_quasi_nul => LM_Andrei_quasi_nul,
    # :LM_SPG_quasi_nul_λD => LM_SPG_quasi_nul_λD,
    # :LM_Zhu_quasi_nul_λD => LM_Zhu_quasi_nul_λD,
    # :LM_Andrei_quasi_nul_λD => LM_Andrei_quasi_nul_λD,
    );
    
# problems_names = setdiff(names(NLSProblems), [:NLSProblems]);
# problems = (eval((problem))() for problem ∈ problems_names);
# pb = collect(problems);
# pb_sc = filter(problem -> problem.meta.ncon == 0, pb);

pb_mgh = [NLSProblems.mgh02(), NLSProblems.mgh06(), NLSProblems.mgh08(), NLSProblems.mgh10(), NLSProblems.mgh15(), NLSProblems.mgh16(), NLSProblems.mgh17(), NLSProblems.mgh19(), NLSProblems.mgh23(), NLSProblems.mgh23(20), NLSProblems.mgh24(10), NLSProblems.mgh26(), NLSProblems.mgh30(), NLSProblems.mgh31(), NLSProblems.mgh31(20), NLSProblems.mgh32(), NLSProblems.mgh33(), NLSProblems.mgh33(20), NLSProblems.mgh34(), NLSProblems.mgh34(20)];

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
# eq_nls_problems_2  = [FeasibilityFormNLS(problem) for problem in eq_nls_problems];


pp(dict_solvers, pb_mgh; save_stats = true)