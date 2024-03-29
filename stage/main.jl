include("LM.jl")
include("AuxiliaryFunctions.jl")


# DÉPLACER LES 4 FCTS DÉFINIES IMPLICITEMENT DANS LM ICI, RAJOUTER LES η etc EN ARGUMENTS DE CETTE FONCTION ET 
# DANS LES ARGUMENTS DE pp(), et le seul endroit ou elle va servir dans pp() c'est pour nommer le fichier .png

dict_solvers = Dict(
    :LM => LM,
    :LM_SPG => LM_SPG,
    :LM_Zhu => LM_Zhu,
    :LM_Andrei => LM_Andrei
    )

problems_names = setdiff(names(NLSProblems), [:NLSProblems])
problems = (eval((problem))() for problem ∈ problems_names)


###################### Test sur un problème unique #######################

pb = collect(problems)
pb_sc = filter(problem -> problem.meta.ncon == 0, pb)

# compare_solvers(8, dict_solvers, pb_sc; type = "grad", save = false)


######################## Profils de performance #########################

# pp(dict_solvers, problems; save = false)


for k = 1:10
    compare_solvers(k, dict_solvers, pb_sc; type = "grad", save = true)
end
