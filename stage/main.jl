using ADNLPModels, JSOSolvers, NLSProblems, SolverBenchmark, Plots
include("LM.jl")

problems_names = setdiff(names(NLSProblems), [:NLSProblems])
problems = (eval((problem))() for problem ∈ problems_names)


###################### Test sur un problème unique #######################

pb = collect(problems)
pb_sc = filter(problem -> problem.meta.ncon == 0, pb)
pb_test1 = filter(problem -> problem.meta.name == "tp272", pb)

solvers = [lm,
        LM_SPG,
        LM_Zhu,
        LM_Andrei]
solvers_names = Dict(lm => "LM", LM_SPG => "LM_SPG", LM_Zhu => "LM_zhu", LM_Andrei => "LM_Andrei")
pb_test = pb_sc[7]

function comparaison_solvers(;type :: String ="grad" , save :: Bool = false)
    for k = 1:length(solvers) 
        solver = solvers[k]
        name = solvers_names[solver]

        stats, obj, grad = solver(pb_test; bool=true)
        to_plot = (type == "grad") ? grad : obj
        y_label = (type == "grad") ? "‖JᵀF‖" : "‖F‖"
        rangs = 1:length(grad)
        if k == 1
            plot(rangs, to_plot, xlabel="k", ylabel=y_label,yaxis =:log10, label=name, title="problème : "*pb_test.meta.name)
        else
            plot!(rangs, to_plot, label=name)
        end
        reset!(pb_test)
    end
    display(current())
    save && savefig(type*"_"*pb_test.meta.name*".png")
end
# savefig("obj_mgh03.png")


######################## Profils de performance #########################



function pp(; save :: Bool = false)
    solvers = Dict(
    :LM => LM,
    :LM_SPG => LM_SPG,
    :LM_Zhu => LM_Zhu,
    :LM_Andrei => LM_Andrei
    )
    stats = bmark_solvers(solvers, problems)

    cols = [:name, :status, :objective, :elapsed_time, :iter]
    for solver ∈ keys(solvers)
        pretty_stats(stats[solver][!, cols])
    end
    cost(df) = (df.status .!= :first_order) * Inf + df.iter
    performance_profile(stats, cost)
    display(current())
    save && savefig("performance_profile.png")
end

# 