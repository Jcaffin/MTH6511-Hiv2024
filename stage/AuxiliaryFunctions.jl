using NLSProblems, JSOSolvers, SolverBenchmark, Plots

function compare_solvers(pb_sc,
    dict_solvers;
    type :: String ="grad",
    save :: Bool = false)

    solvers_names = Dict(LM => "LM", LM_SPG => "LM_SPG", LM_Zhu => "LM_zhu", LM_Andrei => "LM_Andrei")
    solvers = collect(values(dict_solvers))

    for k = 1:length(solvers) 
        solver = solvers[k]
        name = solvers_names[solver]

        stats, obj, grad = solver(pb_sc; bool=true)
        to_plot = (type == "grad") ? grad : obj
        #y_label = (type == "grad") ? "‖JᵀF‖" : "‖F‖"
        rangs = 1:length(grad)
        if k == 1
            plot(rangs, to_plot, xlabel="k", ylabel=type,yaxis =:log10, label=name, title="problème : "*pb_sc.meta.name)
        else
            plot!(rangs, to_plot, label=name)
        end
        reset!(pb_sc)
    end
    display(current())
    save && savefig("Pictures/Comparaison_new/"*type*"_"*pb_test.meta.name*".png")
end


function pp(dict_solvers, 
    problems; 
    save :: Bool = false
    )

    stats = bmark_solvers(dict_solvers, problems, skipif = problem -> (problem.meta.ncon == 0) ? false : true)
    #stats = bmark_solvers(dict_solv, problems)

    cols = [:name, :status, :objective, :elapsed_time, :iter]
    for solver ∈ keys(solvers)
        pretty_stats(stats[solver][!, cols])
    end
    cost(df) = (df.status .!= :first_order) * Inf + df.iter
    performance_profile(stats, cost)
    display(current())
    save && savefig("Pictures/Performance_profiles/η.png")
end