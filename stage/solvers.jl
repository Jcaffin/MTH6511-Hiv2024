using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels


function quasi_newton_bfgs(
    nlp :: AbstractNLPModel; # Only mandatory argument, notice the ;
    max_time :: Float64 = 30.0, # maximum allowed time
    max_iter :: Int = 100 # maximum allowed iterations
  )
    
    xk = copy(nlp.meta.x0)
    fk = obj(nlp, xk)
    gk = grad(nlp, xk)
    Hk = I(length(xk))
    gnorm = gnorm0 = norm(gk)
    k=0
    
    k=0
    t₀ = time()
    Δt = time() - t₀
    status = :unknown
    tired = Δt ≥ max_time > 0 || k ≥ max_iter > 0
    solved = gnorm ≤ 1.0e-6 + 1.0e-6 * gnorm0

    while !(solved || tired) # && fk > -1e15
        dk = -Hk * gk
        α = armijo(xk, dk, fk, gk, x -> obj(nlp, x)) 
        sk = α*dk
        xk += sk
        new_fk = obj(nlp, xk)
        new_gk = grad(nlp, xk)
        yk = new_gk - gk
        if yk' * sk > 0
            ρk = 1 / dot(yk, sk)
            Hk = (I - ρk * sk * yk') * Hk * (I - ρk * yk * sk') + ρk * sk * sk' 
        end
        fk, gk = new_fk, new_gk
        gnorm = norm(gk)

        k += 1
        Δt = time() - t₀
        tired = Δt ≥ max_time > 0 || k ≥ max_iter > 0
        solved = gnorm ≤ 1.0e-6 + 1.0e-6 * gnorm0
    end
    if solved
        status = :first_order
      elseif tired
        if Δt ≥ max_time > 0
          status = :max_time
        elseif k ≥ max_iter > 0
          status = :max_iter
        end
    end
    return GenericExecutionStats(nlp, status=status, solution=xk, objective=obj(nlp, xk), iter=k, elapsed_time=Δt)
end