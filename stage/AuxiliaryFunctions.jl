function write_dataframe_to_doc(df::DataFrame, filename::String)
    open(filename, "a") do file
        pretty_table(file, df, tf = tf_unicode_rounded) 
    end
end

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

function fill_solver_comp_file_to_df(file_path::String)
    # Lire le contenu du fichier
    file_content = read(file_path, String)

    # Extraire les lignes du contenu du fichier
    lines = split(file_content, "\n")

    dataframes = []
    current_table = []
    header = []
    column_types = []
    current_solver = ""
    current_model_function = ""
    current_stop_reason = ""
    in_table = false
    is_type_line = false

    for line in lines
        if startswith(line, "Solver :")
            # Enregistrer le solveur actuel
            current_solver = strip(line[9:end])
        elseif startswith(line, "MODÈLE :")
            # Enregistrer le modèle et la fonction
            current_model_function = strip(line[9:end])
        elseif startswith(line, "Raison d'arrêt :")
            # Enregistrer la raison d'arrêt
            current_stop_reason = strip(line[17:end])
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
                push!(dataframes, (df, current_model_function, current_stop_reason))
                current_table = []
                header = []
                column_types = []
                in_table = false
                current_stop_reason = ""  # Réinitialiser la raison d'arrêt pour la prochaine table
            end
        elseif startswith(line, "╭")
            # Début d'une nouvelle table
            in_table = true
        end
    end

    return dataframes
end

function comp_df_to_plot(dataframes,
    nls_name :: String; 
    save :: Bool = false,
    is_obj :: Bool = true,
    is_iteration :: Bool =true)
    # Créer une figure vide
    if is_iteration
        plt = plot(title=nls_name, xlabel = "Itération", ylabel = is_obj ? "‖F(x)‖" : "‖J'.F‖", yscale=:log10)
    else
        plt = plot(title=nls_name, xlabel = "Evaluation", ylabel = is_obj ? "‖F(x)‖" : "‖J'.F‖", xscale=:log10, yscale=:log10)
    end
    

    # Ajouter les courbes pour chaque DataFrame
    for (df, model_function, stop_reason) in dataframes
        # Renommer les colonnes pour éviter les caractères spéciaux
        if "‖F(x)‖" in names(df)
            rename!(df, Symbol("‖F(x)‖") => :F_x)
        end
        if "‖J'.F‖" in names(df)
            rename!(df, Symbol("‖J'.F‖") => :G_x)
        end
        if "itérations" in names(df)
            rename!(df, Symbol("itérations") => :iterations)
        end
        if "évaluations" in names(df)
            rename!(df, Symbol("évaluations") => :evaluations)
        end

        x = is_iteration ? df[:, :iterations] : df[:, :evaluations]
        y = is_obj ? df[:, :F_x] : df[:, :G_x]
        plot!(plt, x, y, label=model_function)
    end

    # Afficher la figure
    save && savefig("Archives/Comparaisons/comp_"*nls_name*"_grad_eval.svg")
    display(plt)
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

    indicateur = "OPeq_pp_SPGs_ITER"
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

    # Générer le profil de performance avec des couleurs personnalisées
    p = performance_profile(stats, cost; b = PlotsBackend())
    series = p.series_list
    for s in series
        if s[:label] == "LM_SPG_alt"
            s[:linecolor] = :red
        elseif s[:label] == "LM_Zhu_alt 2"
            s[:linecolor] = :blue
        elseif s[:label] == "LM_Andrei_alt 3"
            s[:linecolor] = :green
        end
    end
    display(p)
    save_stats && save(file_jld2, dict)
    save_stats && savefig(file_svg)
end

# ING: Method definition cost(Any) in module Main at /Users/jules/Desktop/MTH6511/2-MTH6511-Hiv2024/stage/AuxiliaryFunctions.jl:361 overwritten at /Users/jules/Desktop/MTH6511/2-MTH6511-Hiv2024/stage/AuxiliaryFunctions.jl:363.

# function pp(dict_solvers,
#     problems; 
#     save_stats :: Bool = false,
#     is_iter :: Bool = true,
#     kwargs...)

#     indicateur = "testtesttest"
#     file_txt = "Archives/Performance_profiles/"*indicateur*".txt"
#     file_svg = "Archives/Performance_profiles/"*indicateur*".svg"

#     stats = bmark_solvers(dict_solvers, problems, skipif = problem -> (problem.meta.ncon == 0) ? false : true; kwargs...)
#     cols = [:name, :status, :objective, :elapsed_time, :iter, :neval_residual]
#     for solver ∈ keys(dict_solvers)
#         pretty_stats(stats[solver][!, cols])
#         @show String(solver)
#         save_stats && write_msg_to_doc("Solver : "*String(solver), file_txt)
#         save_stats && write_dataframe_to_doc(stats[solver][!, cols], file_txt)
#     end
#     if is_iter
#         cost(df) = (df.status .!= :first_order) * Inf + df.iter
#     else
#         cost(df) = (df.status .!= :first_order) * Inf + df.neval_residual
#     end
#     performance_profile(stats, cost)
#     display(current())
#     save_stats && savefig(file_svg)
# end