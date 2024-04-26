using NLSProblems

include("LM.jl")
include("AuxiliaryFunctions.jl")


# DÉPLACER LES 4 FCTS DÉFINIES IMPLICITEMENT DANS LM ICI, RAJOUTER LES η etc EN ARGUMENTS DE CETTE FONCTION ET 
# DANS LES ARGUMENTS DE pp(), et le seul endroit ou elle va servir dans pp() c'est pour nommer le fichier .png
dict_solvers = Dict(
    :LM => LM,
    :LM_SPG => LM_SPG,
    #:LM_Zhu => LM_Zhu,
    :LM_Andrei => LM_Andrei,
    :LM_SPG_alt => LM_SPG_alt,
    #:LM_Zhu_alt => LM_Zhu_alt,
    :LM_Andrei_alt => LM_Andrei_alt
    )

    
problems_names = setdiff(names(NLSProblems), [:NLSProblems])
problems = (eval((problem))() for problem ∈ problems_names)
pb = collect(problems)
pb_sc = filter(problem -> problem.meta.ncon == 0, pb)
#pb_sc = filter(problem -> problem.meta.nvar == 2, pb)

################################# Tests ##################################
# nlp = pb_sc[8]
# x  = nlp.meta.x0
# rows, cols = jac_structure_residual(nlp)
# vals = jac_coord_residual(nlp, x)
# Jx_sparse = sparse(rows, cols, vals)
# Jx_full = jac_residual(nlp, x)


###################### Test sur un problème unique #######################

compare_solvers(pb_sc[1], dict_solvers; type = "obj", save = false)

######################## Profils de performance #########################

# pp(dict_solvers, problems; save = true)



####################### Générer tous les graphes ########################

# for k =1:length(pb_sc)
#     compare_solvers(pb_sc[k], dict_solvers; type = "obj", save = true)
# end
