function write_msg_file(msg::String, filename::String)
    open(filename, "a") do file  # Ouvrir le fichier en mode append
        println(file, msg)
    end
end

function write_dense_vector(v::Vector, filename::String, var_name::String)
    open(filename, "a") do file  # Ouvrir le fichier en mode append
        print(file, var_name, " : [")
        n = length(v)
        for (i, value) in enumerate(v)
            print(file, value)
            if i < n  # Si ce n'est pas le dernier élément
                print(file, ", ")
            end
        end
        print(file, "]")
        println(file)  # Saut de ligne à la fin du vecteur
        println(file)  # Saut de ligne à la fin du vecteur
    end
end

function write_sparse_matrix_as_dense(A::SparseMatrixCSC, filename::String, var_name::String)
    open(filename, "a") do file
        rows, cols, vals = findnz(A)
        n = length(rows)
        print(file, var_name," : [")
        for (i, r) in enumerate(rows)
            print(file, r)
            if i < n  # Si ce n'est pas le dernier élément
                print(file, ", ")
            end
        end
        print(file, "]")
        println(file)
        print(file, repeat(" ", length(var_name)),"   [")
        for (j, c) in enumerate(cols)
            print(file, c)
            if j < n  # Si ce n'est pas le dernier élément
                print(file, ", ")
            end
        end
        print(file, "]")
        println(file)
        print(file, repeat(" ", length(var_name)),"   [")
        for (k, v) in enumerate(vals)
            print(file, v)
            if k < n  # Si ce n'est pas le dernier élément
                print(file, ", ")
            end
        end
        print(file, "]")
        println(file)
        println(file)
    end
end

function compare_solvers(pb_sc,
    dict_solvers;
    type    :: String ="grad",
    save    :: Bool = false,
    verbose :: Bool = true)

    solvers_names = Dict(
        LM_test => "LM_test", 
        LM_SPG => "LM_SPG", 
        LM_Zhu => "LM_zhu", 
        LM_Andrei => "LM_Andrei", 
        LM_Andrei_qrmumps => "LM_Andrei_qrmumps",
        LM_SPG_alt => "LM_SPG_alt", 
        LM_Zhu_alt => "LM_Zhu_alt", 
        LM_Andrei_alt => "LM_Andrei_alt",
        LM_Andrei_alt_qrmumps => "LM_Andrei_alt_qrmumps",
        LM_SPG_quasi_nul_lin => "LM_SPG_quasi_nul_lin",
        LM_Zhu_quasi_nul_lin => "LM_Zhu_quasi_nul_lin",
        LM_Andrei_quasi_nul_lin => "LM_Andrei_quasi_nul_lin",
        LM_Andrei_quasi_nul_lin_qrmumps => "LM_Andrei_quasi_nul_lin_qrmumps"
        )
    solvers = collect(values(dict_solvers))

    for k = 1:lastindex(solvers) 
        solver = solvers[k]
        name = solvers_names[solver]
        @show name

        stats, obj, grad = solver(pb_sc; bool_grad_obj=true, bool_verbose = verbose)
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
    save && savefig("Pictures/Performance_profiles/LMD_SPG_Zhu_Andrei.svg")
end