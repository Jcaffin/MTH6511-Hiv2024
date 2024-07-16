pb_mgh = [mgh02(), mgh06(), mgh08(), mgh10(), mgh15(), mgh16(), mgh17(), mgh19(), mgh23(), mgh23(20), mgh24(10), mgh26(), mgh30(), mgh31(), mgh31(20), mgh32(), mgh33(), mgh33(20), mgh34(), mgh34(20)]
nlp = pb_mgh[2]
m, n, nnzj = nlp.nls_meta.nequ, nlp.meta.nvar, nlp.nls_meta.nnzj
@show nnzj
x0 = nlp.meta.x0

Jrows        = Vector{Int}(undef, nnzj)
Jcols        = Vector{Int}(undef, nnzj)
Jvals        = Vector{eltype(x0)}(undef, nnzj)

jac_structure_residual!(nlp, Jrows, Jcols)
jac_coord_residual!(nlp, x0, Jvals)

Jx = SparseMatrixCOO(m, n, Jrows, Jcols, Jvals)
Jx₋₁ = similar(Jx)



function create_start_indices(Jcols)
    n = maximum(Jcols)
    a = Vector{Union{Int, Missing}}(undef, n)
    fill!(a, missing)
    current_line = -1
    for i in 1:lastindex(Jcols)
        if Jcols[i] != current_line
            current_line = Jcols[i]
            a[current_line] = i
        end
    end
    return a
end

function create_end_indices(Jcols)
    n = maximum(Jcols)
    a = Vector{Union{Int, Missing}}(undef, n)
    fill!(a, missing)
    for i in 1:lastindex(Jcols)
        a[Jcols[i]] = i
    end
    return a
end



vecteur = [1, 5, 5, 5, 5]