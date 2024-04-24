m = 10
n = 5
p = 5
T = Float64

transp = (T <: Real) ? 't' : 'c'

A = sprand(T, m, n, 0.3)
b = rand(T, m)


spmat = qrm_spmat_init(A)
spfct = qrm_analyse(spmat)
qrm_factorize!(spmat, spfct)


z = qrm_apply(spfct, b, transp=transp)
x = qrm_solve(spfct, z, transp='n')
r = b - A * x




