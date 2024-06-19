using SparseArrays, QRMumps

m,n  = 5 , 13

Fx = [-80.61017305526643, 10.606248341154838, 2.8284271247461903, 23.607675141438072, -52.32590180780452]

Jrows = [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5]
Jcols = [1, 2, 3, 2, 3, 4, 5, 6, 7, 7, 8, 9, 11, 11, 12, 13, 5, 6, 10]
Jvals = [4.242640687119286, 0.0, 0.0, 2.8284271247461903, 2.8284271247461903, 16.970562748477143, -0.7067534044254388, 0.7067534045431721, 1.4142135623730951, 1.4142135623730951, 1.4142135623730951, -2.8284271247461903, 1.4142135623730951, 1.4142135623730951, 1.4142135623730951, -7.0710678118654755, 1.4142135623730951, 2.8284271247461903, 15.556349186104047]
nnzj  = 19
λ = 1e-6

b = zeros(Float64, m+n)
b[1:m]        .= Fx
b .*= -1


###################################################
Jx        = sparse(Jrows, Jcols, Jvals, m, n)
A1        = spzeros(m+n, n)
sqrt_DλI1 = spdiagm(0 => ones(n))
D1 = SparseArrays.Diagonal(ones(n))
for i = 1:n
    sqrt_DλI1[i,i] = sqrt(D1[i,i] + λ)
end
A1[1:m, :]     .= Jx
A1[m+1:end, :] .= sqrt_DλI1

# spmat1 = qrm_spmat_init(A1)
# spfct1 = qrm_analyse(spmat1)
# qrm_factorize!(spmat1, spfct1)
# z1 = qrm_apply(spfct1, b, transp='t')
# d1 = qrm_solve(spfct1, z1, transp='n')
    

###################################################
A2rows              = zeros(Int64, nnzj + n)
A2rows[1:nnzj]     .= Jrows 
A2rows[nnzj+1:end] .= [k for k = m+1:m+n]
A2cols              = zeros(Int64, nnzj + n)
A2cols[1:nnzj]     .= Jcols 
A2cols[nnzj+1:end] .= [k for k = 1:n]
A2vals              = zeros(Float64, nnzj + n)


sqrt_DλI2 = ones(n)
D2 = ones(n)
for i = 1:n
    sqrt_DλI2[i] = sqrt(D2[i] + λ)
end
A2vals[1:nnzj]       .= Jvals
A2vals[nnzj + 1:end] .= sqrt_DλI2
A2                  = sparse(A2rows, A2cols, A2vals)

@show A1 == A2

# spmat2 = qrm_spmat_init(m, n, A2rows, A2cols, A2vals)
# spfct2 = qrm_analyse(spmat2)
# qrm_factorize!(spmat2, spfct2)
# z2 = qrm_apply(spfct2, b, transp='t')
# d2 = qrm_solve(spfct2, z2, transp='n')

