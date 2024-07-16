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
