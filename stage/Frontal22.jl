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
# Pkg.add("BundleAdjustmentModels")


# using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, SparseArrays, QRMumps, JSOSolvers, SolverBenchmark, Plots, NLSProblems, SparseMatricesCOO, Dates, DataFrames, PrettyTables, BundleAdjustmentModels

include("LM.jl")
include("AuxiliaryFunctions.jl")

df = problems_df()
list_names = df[!,:name]
function solve(nls; kwargs...)
    filename = "Archives/result_"*nls.meta.name*".txt"
    write_solvers_df_to_doc(nls, filename; verbose = true, kwargs...)
end
pb_name = ""; 
solve(BundleAdjustmentModel(pb_name))
