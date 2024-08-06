function write_solver_df_to_jld2(nls, indicateur_filename::String; kwargs...)
    formatted_date = Dates.format(today(), "dd-mm-yyyy")
    file_jld2 = "Archives/Performance_profiles/"*nls.meta.name*"_"*indicateur_filename*"_"*formatted_date*".jld2"
    dict = Dict()
    stats, df = LM_D(nls; save_df = true, kwargs...)
    # rename!(df, :iter => "itérations", :nf => "évaluations", :F => "‖F(x)‖", :G => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
    dict[nls.meta.name*indicateur_filename] = df
    dict[indicateur_filename*"_status"] = stats.status
    save(file_jld2, dict)
    reset!(nls)
end

function fill_pp_file_to_df(file_path::String)
    # Lire le contenu du fichier
    file_content = read(file_path, String)

    # Extraire les lignes du contenu du fichier
    lines = split(file_content, "\n")

    dataframes = []
    current_table = []
    header = []
    column_types = []
    current_solver = ""
    in_table = false
    is_type_line = false

    for line in lines
        if startswith(line, "Solver :")
            # Enregistrer le solveur actuel
            current_solver = strip(line[9:end])
        elseif startswith(line, "│") && !startswith(line, "├") && !startswith(line, "╭") && !startswith(line, "╰")
            if isempty(header)
                # La première ligne après les bordures est l'entête
                header = split(strip(line, '│'), "│")
                header = [strip(h) for h in header]
                is_type_line = true
            elseif is_type_line
                # La deuxième ligne après les bordures est les types de colonnes
                column_types = split(strip(line, '│'), "│")
                column_types = [strip(t) for t in column_types]
                is_type_line = false
            else
                # Ajouter les lignes de données au tableau actuel
                push!(current_table, split(strip(line, '│'), "│"))
            end
        elseif in_table && startswith(line, "╰")
            # Fin de la table actuelle
            if !isempty(current_table)
                # Convertir les colonnes aux types appropriés
                data = Dict{Symbol, Vector}()
                for (i, h) in enumerate(header)
                    col_data = [strip(row[i]) for row in current_table]
                    if column_types[i] == "String"
                        data[Symbol(h)] = col_data
                    elseif column_types[i] == "Symbol"
                        data[Symbol(h)] = Symbol.(col_data)
                    elseif column_types[i] == "Float64"
                        data[Symbol(h)] = parse.(Float64, col_data)
                    elseif column_types[i] == "Int64"
                        data[Symbol(h)] = parse.(Int64, col_data)
                    else
                        data[Symbol(h)] = col_data  # Par défaut, laisser en String
                    end
                end
                df = DataFrame(data)
                df = df[:, Symbol.(header)]  # Assurer l'ordre des colonnes
                push!(dataframes, (df, current_solver))
                current_table = []
                header = []
                column_types = []
                in_table = false
            end
        elseif startswith(line, "╭")
            # Début d'une nouvelle table
            in_table = true
        end
    end

    return dataframes
end

function jdl2_to_pretty_tables_txt(jdl2::String, filename::String)
    d = load(jdl2)
    open(filename, "w") do file
        for (name, df) in d
            println(file, "Table: $name")
            println(file, "-" ^ (length("Table: $name")))
            pretty_table(file, df)
            println(file)  # Ajoute une ligne vide entre les tables
        end
    end
end

function compare_solvers(pb_sc,
    dict_solvers;
    type    :: String = "grad",
    save    :: Bool = false,
    kwargs...)

    solvers = collect(values(dict_solvers))
    k = 1
    for solver ∈ keys(dict_solvers)
        name = String(solver)

        stats, df = eval(solver)(pb_sc; save_df=true, kwargs...)  # comment s'assurer que save_df reste sur true (pour que le solveur retourne le df) tout en ayant kwargs
        obj  = df[!,:F]
        grad = df[!,:G]
        to_plot = (type == "grad") ? grad : obj
        rangs = 1:lastindex(grad)
        if to_plot[end] == 0
            pop!(to_plot)
            rangs = 1:lastindex(grad)-1
        end
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
    save_stats :: Bool = false,
    kwargs...)

    indicateur = "OPes_pp_SPG_Zhu_Andrei_ITER"
    formatted_date = Dates.format(today(), "dd-mm-yyyy")
    file_jld2 = "Archives/Performance_profiles/"*indicateur*"_"*formatted_date*".jld2"
    file_svg = "Archives/Performance_profiles/"*indicateur*"_"*formatted_date*".svg"

    stats = bmark_solvers(dict_solvers, problems, skipif = problem -> (problem.meta.ncon == 0) ? false : true; kwargs...)
    cols = [:name, :status, :objective, :elapsed_time, :iter, :neval_residual]
    save_stats && (dict = Dict())
    for solver ∈ keys(dict_solvers)
        pretty_stats(stats[solver][!, cols])
        @show String(solver)
        save_stats && (dict[String(solver)] = stats[solver][!, cols])
    end
    # cost(df) = (df.status .!= :first_order) * Inf + df.neval_jac_residual
    cost(df) = (df.status .!= :first_order) * Inf + df.iter

    
    color_map = Dict(
        "LM_SPG" => RGB(1.0, 0.0, 0.0),
        "LM_SPG_λD" => RGB(0.9, 0.0, 0.0),
        "LM_SPG_alt" => RGB(1.0, 0.6, 0.6),
        "LM_SPG_alt_λD" => RGB(1.0, 0.4, 0.4),
        "LM_SPG_quasi_nul" => RGB(0.8, 0.0, 0.0),
        "LM_SPG_quasi_nul_λD" => RGB(0.6, 0.0, 0.0),

        "LM_Zhu" => RGB(0.0, 1.0, 0.0),
        "LM_Zhu_λD" => RGB(0.0, 0.9, 0.0),
        "LM_Zhu_alt" => RGB(0.6, 1.0, 0.6),
        "LM_Zhu_alt_λD" => RGB(0.4, 1.0, 0.4),
        "LM_Zhu_quasi_nul" => RGB(0.0, 0.8, 0.0),
        "LM_Zhu_quasi_nul_λD" => RGB(0.0, 0.6, 0.0),

        "LM_Andrei" => RGB(0.0, 0.0, 1.0),
        "LM_Andrei_λD" => RGB(0.0, 0.0, 0.9),
        "LM_Andrei_alt" => RGB(0.6, 0.6, 1.0),
        "LM_Andrei_alt_λD" => RGB(0.4, 0.4, 1.0),
        "LM_Andrei_quasi_nul" => RGB(0.0, 0.0, 0.8),
        "LM_Andrei_quasi_nul_λD" => RGB(0.0, 0.0, 0.6),

        "LM" => :grey,
        "hess_exact" => :black,
    )
    p = performance_profile(stats, cost; b = PlotsBackend())
    series = p.series_list
    for s in series
        if haskey(color_map, s[:label])
            s[:linecolor] = color_map[s[:label]]
        end
    end
    display(p)
    save_stats && save(file_jld2, dict)
    save_stats && savefig(file_svg)
end
