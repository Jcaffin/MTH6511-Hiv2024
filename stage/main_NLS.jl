using NLSProblems

include("LM.jl")
include("AuxiliaryFunctions.jl")



dict_solvers = Dict(
    #:LM => LM,
    #:LM_SPG => LM_SPG,
    #:LM_Zhu => LM_Zhu,
    :LM_Andrei => LM_Andrei,
    #:LM_SPG_alt => LM_SPG_alt,
    #:LM_Zhu_alt => LM_Zhu_alt,
    #:LM_Andrei_alt => LM_Andrei_alt
    )

    
problems_names = setdiff(names(NLSProblems), [:NLSProblems])
problems = (eval((problem))() for problem ∈ problems_names)
pb = collect(problems)
#pb_sc = filter(problem -> problem.meta.ncon == 0, pb)
pb_sc = filter(problem -> problem.meta.name == "mgh10", pb)

###################### Test sur un problème unique #######################

compare_solvers(pb_sc[1], dict_solvers; type = "obj", save = false)

######################## Profils de performance #########################

# pp(dict_solvers, problems; save = true)



####################### Générer tous les graphes ########################

# for k =1:length(pb_sc)
#     compare_solvers(pb_sc[k], dict_solvers; type = "obj", save = true)
# end
