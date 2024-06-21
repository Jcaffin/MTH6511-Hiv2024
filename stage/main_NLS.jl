using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, SparseArrays, QRMumps
using JSOSolvers, SolverBenchmark, Plots, NLSProblems
using Dates, DataFrames

# include("LM.jl")
include("LM.jl")
include("AuxiliaryFunctions.jl")

dict_solvers = Dict(
    :LM_test => LM_test,
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
    
problems_names = setdiff(names(NLSProblems), [:NLSProblems]);
problems = (eval((problem))() for problem ∈ problems_names);
pb = collect(problems);
pb_sc = filter(problem -> problem.meta.ncon == 0, pb);
pb_mgh = [mgh02(), mgh06(), mgh08(), mgh10(), mgh15(), mgh16(), mgh17(), mgh19(), mgh23(), mgh23(20), mgh24(10), mgh26(), mgh30(), mgh31(), mgh31(20), mgh32(), mgh33(), mgh33(20), mgh34(), mgh34(20)]

###################### Test sur un problème unique #######################
# compare_solvers(pb_mgh[10], dict_solvers; type = "obj", save = false)

######################## Profils de performance #########################
pp(dict_solvers, pb_mgh; save = true)

####################### Générer tous les graphes ########################
# for k in eachindex(pb_sc)
#     compare_solvers(pb_sc[k], dict_solvers; type = "obj", save = true)
#     sleep(1.5)
# end

########################## Écriture dataframe ###########################
# nlp = pb_sc[1]
# alt_mod = true
# app_qnl = false
# fct = Andrei
# filename = generate_filename(nlp, fct, alt_mod, app_qnl)
# stats, df = LM_D(pb_sc[1]; fctD = fct, alternative_model = alt_mod, approxD_quasi_nul_lin = app_qnl, save_df = true, verbose = true)
# write_dataframe_to_doc(filename, df)