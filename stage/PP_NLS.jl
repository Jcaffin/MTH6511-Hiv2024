# using Pkg

# Pkg.add("NLPModels")
# Pkg.add("Printf")
# Pkg.add("Logging")
# Pkg.add("SolverCore")
# Pkg.add("Test")
# Pkg.add("ADNLPModels")
# Pkg.add("SparseArrays")
# Pkg.add("QRMumps")
# Pkg.add("JSOSolvers")
# Pkg.add("SolverBenchmark")
# Pkg.add("Plots")
# Pkg.add("NLSProblems")
# Pkg.add("SparseMatricesCOO")
# Pkg.add("Dates")
# Pkg.add("DataFrames")
# Pkg.add("PrettyTables")


# using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, SparseArrays, QRMumps, JSOSolvers, SolverBenchmark, Plots, NLSProblems, SparseMatricesCOO, Dates, DataFrames, PrettyTables

include("LM.jl")
include("AuxiliaryFunctions.jl")

dict_solvers = Dict(
    :LM_GN => LM_GN,
    :LM_SPG => LM_SPG,
    :LM_Zhu => LM_Zhu,
    :LM_Andrei => LM_Andrei,
    :LM_SPG_λD => LM_SPG_λD,
    :LM_Zhu_λD => LM_Zhu_λD,
    :LM_Andrei_λD => LM_Andrei_λD,
    :LM_SPG_alt => LM_SPG_alt,
    :LM_Zhu_alt => LM_Zhu_alt,
    :LM_Andrei_alt => LM_Andrei_alt,
    :LM_SPG_alt_λD => LM_SPG_alt_λD,
    :LM_Zhu_alt_λD => LM_Zhu_alt_λD,
    :LM_Andrei_alt_λD => LM_Andrei_alt_λD,
    # :LM_SPG_quasi_nul_lin => LM_SPG_quasi_nul_lin,
    # :LM_Zhu_quasi_nul_lin => LM_Zhu_quasi_nul_lin,
    # :LM_Andrei_quasi_nul_lin => LM_Andrei_quasi_nul_lin,
    # :LM_Andrei_quasi_nul_lin_λD => LM_Andrei_quasi_nul_lin_λD,
    );

pb_mgh = [mgh02(), mgh06(), mgh08(), mgh10(), mgh15(), mgh16(), mgh17(), mgh19(), mgh23(), mgh23(20), mgh24(10), mgh26(), mgh30(), mgh31(), mgh31(20), mgh32(), mgh33(), mgh33(20), mgh34(), mgh34(20)]
pp(dict_solvers, pb_mgh; save_stats = false, verbose = true)