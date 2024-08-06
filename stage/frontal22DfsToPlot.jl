using DataFrames, Plots

function frontal22_prettytables_to_dfs(file_path::String)
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
    current_st