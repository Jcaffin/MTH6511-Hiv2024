using BundleAdjustmentModels
include("AuxiliaryFunctions.jl")
include("test.jl")

df = problems_df()
filter_df = df[ ( df.nvar .≤ 34000 ), :]

name = filter_df[1, :name]

model = BundleAdjustmentModel(name)

dict_solvers = Dict(
    # :LM_wo_D => LM_wo_D,
    :LM_test => LM_test,
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

compare_solvers(model, dict_solvers; type = "obj", save = true)