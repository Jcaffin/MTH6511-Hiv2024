# m = 10
# n = 5
# p = 5
# T = Float64

# transp = (T <: Real) ? 't' : 'c'

# A = sprand(T, m, n, 0.3)
# b = rand(T, m)


# spmat = qrm_spmat_init(A)
# spfct = qrm_analyse(spmat)
# qrm_factorize!(spmat, spfct)


# z = qrm_apply(spfct, b, transp=transp)
# x = qrm_solve(spfct, z, transp='n')
# r = b - A * x


irn = [1, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7]
jcn = [1, 3, 5, 2, 3, 5, 1, 4, 4, 5, 2, 1, 3]
val = [1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 4.0, 1.0, 5.0, 1.0, 3.0, 6.0, 1.0]

A = sparse(irn, jcn, val, 7, 5)
b = [22.0, 5.0, 13.0, 8.0, 25.0, 5.0, 9.0]
x_star = [1.0, 2.0, 3.0, 4.0, 5.0]



function argmin_q(A, b)
    QR = qr(A)
    x = QR\(b)
    return x
end

function test(A, b)
    
    return x
end



