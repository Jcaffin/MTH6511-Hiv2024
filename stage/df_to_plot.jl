using DataFrames, Plots
include("AuxiliaryFunctions.jl")

# problem-16-22106
# problem-21-11315
# problem-39-18060
# problem-50-20431
pb_string = "problem-16-22106"

# Exemple d'utilisation
comp_path = "/Users/jules/Desktop/"*pb_string*".txt"
dataframes_comp = fill_solver_comp_file_to_df(comp_path)




comp_df_to_plot(dataframes_comp, pb_string; save = true, is_obj = false, is_iteration = false)