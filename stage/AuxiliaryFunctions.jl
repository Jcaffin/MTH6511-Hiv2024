function compare_solvers(pb_sc,
    dict_solvers;
    type :: String ="grad",
    save :: Bool = false)

    solvers_names = Dict(
        LM_test => "LM_test", 
        LM_wo_D => "LM_wo_D", 
        LM_D_y_diese => "LM_D_y_diese",
        LM_D_y_tilde => "LM_D_y_tilde",
        LM_SPG => "LM_SPG", 
        LM_Zhu => "LM_zhu", 
        LM_Andrei => "LM_Andrei", 
        LM_SPG_alt => "LM_SPG_alt", 
        LM_Zhu_alt => "LM_Zhu_alt", 
        LM_Andrei_alt => "LM_Andrei_alt" 
        )
    solvers = collect(values(dict_solvers))

    for k = 1:lastindex(solvers) 
        solver = solvers[k]
        name = solvers_names[solver]

        stats, obj, grad = solver(pb_sc; bool_grad_obj=true)
        to_plot = (type == "grad") ? grad : obj
        rangs = 1:length(grad)
        if k == 1
            plot(rangs, to_plot, xlabel="k", ylabel=type,yaxis =:log10, label=name, title="problème : "*pb_sc.meta.name)
        else
            plot!(rangs, to_plot, label=name)
        end
        reset!(pb_sc)
    end
    display(current())
    save && savefig("Pictures/Comparaison/"*pb_sc.meta.name*"_"*type*".svg")
end


function pp(dict_solvers, 
    problems; 
    save :: Bool = false
    )

    stats = bmark_solvers(dict_solvers, problems, skipif = problem -> (problem.meta.ncon == 0) ? false : true)

    cols = [:name, :status, :objective, :elapsed_time, :iter]
    for solver ∈ keys(dict_solvers)
        pretty_stats(stats[solver][!, cols])
    end
    cost(df) = (df.status .!= :first_order) * Inf + df.iter
    performance_profile(stats, cost)
    display(current())
    save && savefig("Pictures/Performance_profiles/LMD_Andrei_ALT.svg")
end