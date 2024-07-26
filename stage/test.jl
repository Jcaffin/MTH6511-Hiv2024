# pb_name = "problem-182-116770_ERROR"
# pb_name = "problem-237-154414_ERROR"
# pb_name = "problem-262-169354_ERROR"
# pb_name = "problem-273-176305_ERROR"
# pb_name = "problem-287-182023_ERROR"
pb_name = "problem-308-195089_ERROR"


dfs = fill_solver_comp_file_to_df("/Users/jules/Desktop/MTH6511/Archives/Frontal22/2024-07-03/2-ERROR/"*pb_name*".txt")
plt = plot()

# Ajouter chaque courbe au plot
for (df, string1, string2) in dfs
    rename!(df, Dict("itérations" => "iterations", "évaluations" => "evaluations", "‖F(x)‖" => "F", "‖J'.F‖" => "G"))
    # plot!(plt, df.evaluations, df.F, title = pb_name, xlabel = "#F", ylabel = "‖F(x)‖", label=string1, xscale=:log10, yscale=:log10)
    plot!(plt, df.evaluations, df.G, title = pb_name, xlabel = "#F", ylabel = "‖J'.F‖", label=string1, xscale=:log10, yscale=:log10)
end

# Afficher le plot
display(plt)
# savefig(pb_name*"_OBJ.svg")
savefig(pb_name*"_GRAD.svg")