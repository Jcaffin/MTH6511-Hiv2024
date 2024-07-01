function write_msg_to_doc(msg::String, filename::String)
    open(filename, "a") do file  # Ouvrir le fichier en mode append
        println(file, msg)
    end
end

function write_dense_vector_to_doc(v::Vector, filename::String, var_name::String)
    open(filename, "a") do file  # Ouvrir le fichier en mode append
        print(file, var_name, " = [")
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

function write_sparse_matrix_to_doc(A::SparseMatrixCSC, filename::String, var_name::String)
    open(filename, "a") do file
        rows, cols, vals = findnz(A)
        n = length(rows)
        print(file, var_name," = [")
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

function write_dataframe_to_doc(df::DataFrame, filename::String)
    open(filename, "a") do file
        pretty_table(file, df, tf = tf_unicode_rounded) 
    end
end

function write_solvers_df_to_doc(nls, filename::String)
    write_msg_to_doc("MODÈLE : Gauss-Newton", filename)
    stats1, df1 = LM_tst(nls; save_df = true)
    rename!(df1, :iter => "itérations", :nf => "évaluations", :F => "‖F(x)‖", :G => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
    write_dataframe_to_doc(df1, filename)
    write_msg_to_doc("Raison d'arrêt : "*String(stats1.status), filename)
    write_msg_to_doc("", filename)
    reset!(nls)

    write_msg_to_doc("MODÈLE : JᵀJ + D     -     FONCTION : SPG", filename)
    stats2, df2 = LM_D(nls; fctD = SPG!,    alternative_model = false, approxD_quasi_nul_lin = false, save_df = true)
    rename!(df2, :iter => "itérations", :nf => "évaluations", :F => "‖F(x)‖", :G => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
    write_dataframe_to_doc(df2, filename)
    write_msg_to_doc("Raison d'arrêt : "*String(stats2.status), filename)
    write_msg_to_doc("", filename)
    reset!(nls)

    write_msg_to_doc("MODÈLE : JᵀJ + D     -     FONCTION : Zhu", filename)
    stats3, df3 = LM_D(nls; fctD = Zhu!,    alternative_model = false, approxD_quasi_nul_lin = false, save_df = true)
    rename!(df3, :iter => "itérations", :nf => "évaluations", :F => "‖F(x)‖", :G => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
    write_dataframe_to_doc(df3, filename)
    write_msg_to_doc("Raison d'arrêt : "*String(stats3.status), filename)
    write_msg_to_doc("", filename)
    reset!(nls)

    write_msg_to_doc("MODÈLE : JᵀJ + D     -     FONCTION : Andrei", filename)
    stats4, df4 = LM_D(nls; fctD = Andrei!, alternative_model = false, approxD_quasi_nul_lin = false, save_df = true)
    rename!(df4, :iter => "itérations", :nf => "évaluations", :F => "‖F(x)‖", :G => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
    write_dataframe_to_doc(df4, filename)
    write_msg_to_doc("Raison d'arrêt : "*String(stats4.status), filename)
    write_msg_to_doc("", filename)
    reset!(nls)

    write_msg_to_doc("MODÈLE : Alternatif     -     FONCTION : SPG", filename)
    stats5, df5 = LM_D(nls; fctD = SPG!,    alternative_model = true, approxD_quasi_nul_lin = false, save_df = true)
    rename!(df5, :iter => "itérations", :nf => "évaluations", :F => "‖F(x)‖", :G => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
    write_dataframe_to_doc(df5, filename)
    write_msg_to_doc("Raison d'arrêt : "*String(stats5.status), filename)
    write_msg_to_doc("", filename)
    reset!(nls)

    write_msg_to_doc("MODÈLE : Alternatif     -     FONCTION : Zhu", filename)
    stats6, df6 = LM_D(nls; fctD = Zhu!,    alternative_model = true, approxD_quasi_nul_lin = false, save_df = true)
    rename!(df6, :iter => "itérations", :nf => "évaluations", :F => "‖F(x)‖", :G => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
    write_dataframe_to_doc(df6, filename)
    write_msg_to_doc("Raison d'arrêt : "*String(stats6.status), filename)
    write_msg_to_doc("", filename)
    reset!(nls)

    write_msg_to_doc("MODÈLE : Alternatif     -     FONCTION : Andrei", filename)
    stats7, df7 = LM_D(nls; fctD = Andrei!, alternative_model = true, approxD_quasi_nul_lin = false, save_df = true)
    rename!(df7, :iter => "itérations", :nf => "évaluations", :F => "‖F(x)‖", :G => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
    write_dataframe_to_doc(df7, filename)
    write_msg_to_doc("Raison d'arrêt : "*String(stats7.status), filename)
    write_msg_to_doc("", filename)
end

function generate_filename(nlp      :: AbstractNLSModel,
    fctD                    :: Function,
    alternative_model       :: Bool,
    approxD_quasi_nul_lin   :: Bool,
    )
    parts = ["result", nlp.meta.name, string(nameof(fctD))]
    if alternative_model
        push!(parts, "AltMod")
    end
    if approxD_quasi_nul_lin
        push!(parts, "AppQnl")
    end
        push!(parts, string(now()))
        filename = join(parts, "_") * ".txt"
    return filename
end

function compare_solvers(pb_sc,
    dict_solvers;
    type    :: String ="grad",
    save    :: Bool = false,
    verbose :: Bool = true)
    
    solvers = collect(values(dict_solvers))
    k = 1
    for solver ∈ keys(dict_solvers)
        name = String(solver)

        stats, df = eval(solver)(pb_sc; bool_df=true, bool_verbose = verbose)
        obj  = df[!,:F]
        grad = df[!,:G]
        to_plot = (type == "grad") ? grad : obj
        rangs = 1:length(grad)
        if k == 1
            plot(rangs, to_plot, xlabel="k", ylabel=type,yaxis =:log10, label=name, title="problème : "*pb_sc.meta.name)
        else
            plot!(rangs, to_plot, label=name)
        end
        k+=1
        reset!(pb_sc)
    end
    display(current())
    save && savefig("Archives/Comparaisons/"*pb_sc.meta.name*"_"*type*".svg")
end

function pp(dict_solvers, 
    problems; 
    save :: Bool = false
    )
    indicateur = "testtesttest"
    file_txt = "Archives/Performance_profiles/"*indicateur*".txt"
    file_svg = "Archives/Performance_profiles/"*indicateur*".svg"
    
    stats = bmark_solvers(dict_solvers, problems, skipif = problem -> (problem.meta.ncon == 0) ? false : true)

    cols = [:name, :status, :objective, :elapsed_time, :iter]
    for solver ∈ keys(dict_solvers)
        pretty_stats(stats[solver][!, cols])
        @show String(solver)
        save && write_msg_to_doc("Solver : "*String(solver), file_txt)
        save && write_dataframe_to_doc(stats[solver][!, cols], file_txt)
    end
    cost(df) = (df.status .!= :first_order) * Inf + df.iter
    performance_profile(stats, cost)
    display(current())
    save && savefig(file_svg)
end