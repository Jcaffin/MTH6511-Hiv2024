using BundleAdjustmentModels
include("AuxiliaryFunctions.jl")
include("LM.jl")

df = problems_df()
names = df[!,:name]
nls = BundleAdjustmentModel(names[1])

dict_solvers = Dict(
    # :LM_wo_D => LM_wo_D,
    # :LM_test => LM_test,
    # :LM_SPG => LM_SPG,
    # :LM_Zhu => LM_Zhu,
    :LM_Andrei => LM_Andrei,
    # :LM_SPG_alt => LM_SPG_alt,
    # :LM_Zhu_alt => LM_Zhu_alt,
    :LM_Andrei_alt => LM_Andrei_alt,
    # :LM_SPG_quasi_nul_lin => LM_SPG_quasi_nul_lin,
    # :LM_Zhu_quasi_nul_lin => LM_Zhu_quasi_nul_lin,
    :LM_Andrei_quasi_nul_lin => LM_Andrei_quasi_nul_lin,
    )

filename = "result_"*nls.meta.name*"_"*string(now())*".txt"
write_solvers_df_to_doc(nls, filename)
# compare_solvers(model, dict_solvers; type = "obj", save = true)