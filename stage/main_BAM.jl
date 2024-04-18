using BundleAdjustmentModels

df = problems_df()
filter_df = df[ ( df.nvar .≤ 34000 ), :]

name = filter_df[1, :name]

model = BundleAdjustmentModel(name)

dict_solvers = Dict(
    :LM => LM,
    :LM_SPG => LM_SPG,
    #:LM_Zhu => LM_Zhu,
    :LM_Andrei => LM_Andrei,
    :LM_SPG_alt => LM_SPG_alt,
    #:LM_Zhu_alt => LM_Zhu_alt,
    :LM_Andrei_alt => LM_Andrei_alt
    )

compare_solvers(model, dict_solvers; type = "grad", save = false)