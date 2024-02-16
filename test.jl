df = OptimizationProblems.meta

# On extrait tous les noms des problèmes de moindres carrés non linéaire (sans contrainte)
names = df[(df.objtype .== :least_squares) .& (df.contype .== :unconstrained), :name]
popat!(names,length(names))

# On stocke tous ces problèmes dans ad_problems
ad_problems = (eval(Meta.parse(problem))(use_nls = true) for problem ∈ names)

# Exemple d'un problème stocké dans ad_problems
nls = first(ad_problems)
typeof(nls)

# dictionnaire des solveurs : 4 variantes de TRUNK
solvers = Dict(
  :trunk_cgls => model -> trunk(model, subsolver_type = CglsSolver),
  :trunk_crls => model -> trunk(model, subsolver_type = CrlsSolver),
  :trunk_lsqr => model -> trunk(model, subsolver_type = LsqrSolver),
  :trunk_lsmr => model -> trunk(model, subsolver_type = LsmrSolver)
)

# exécute les solvers sur les problèmes test
stats = bmark_solvers(solvers, ad_problems)


# on crée un DataFrame pour améliorer les résultats 
first_order(df) = df.status .== :first_order
unbounded(df) = df.status .== :unbounded
solved(df) = first_order(df) .| unbounded(df)
costnames = ["time"]
costs = [df -> .!solved(df) .* Inf .+ df.elapsed_time]

# On trace les résultats
gr()
profile_solvers(stats, costs, costnames)

"Quelques modif pour un premier commit"
