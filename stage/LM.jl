using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels

function lm_param(nlp        :: AbstractNLSModel, 
                x        :: AbstractVector, 
                ϵₐ       :: AbstractFloat,
                ϵᵣ       :: AbstractFloat;
                η₁       :: AbstractFloat = 1e-3, 
                η₂       :: AbstractFloat = 0.66, 
                σ₁       :: AbstractFloat = 10.0, 
                σ₂       :: AbstractFloat = 0.5,
                max_eval :: Int = 1000, 
                max_time :: AbstractFloat = 60.,
                max_iter :: Int = typemax(Int64)
                )

    ################ On évalue F(x₀) et J(x₀) ################
    Fx = residual(nlp, x)
    Jx = jac_residual(nlp, x)

    ################## On calcule leur norme #################
    normFx   = norm(Fx)
    normGx₀ = norm(Jx' * Fx)
    normGx = normGx₀

    iter = 0    
    λ = 0.0
    λ₀ = 1e-6

    iter_time = 0.0
    tired   = neval_residual(nlp) > max_eval || iter_time > max_time
    status  = :unknown
    start_time = time()
    optimal    = min(normFx, normGx) ≤ ϵₐ + ϵᵣ*normGx₀

    #################### Tracé des graphes ###################

    objectif = []
    gradient = []

    @info log_header(
        [:iter, :nf, :obj, :grad, :status, :nd, :λ],
        [Int, Int, Float64, Float64, String, Float64, Float64],
        hdr_override=Dict(
            :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :nd => "‖d‖", :λ => "λ")
        )

    while !(optimal || tired)
        ################# Calcul de d par factorisation QR #################
        p = size(Jx, 2)
        A = [Jx; λ * I(p)]
        b = [Fx; zeros(p)]
        
        QR = qr(A)
        d = QR\(-b)

        ######################## Calcul du ratio ρ ########################
        xp      = x + d
        Fxp     = residual(nlp, xp)
        normFxp = norm(Fxp)

        ρ = (normFx^2 - normFxp^2) / (normFx^2 - norm(Jx * d + Fx)^2)
        
        if ρ < η₁
            λ = max(λ₀, σ₁ * λ)
            status = :increase_λ
        else
            x   = xp
            Fx  = residual(nlp, x)
            Jx  = jac_residual(nlp, x)
            normFx   = norm(Fx)
            normGx = norm(Jx' * Fx)
            status = :success
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        push!(objectif,Fxp)
        push!(gradient, normGx)

        @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, status, norm(d), λ])

        iter_time    = time() - start_time
        iter        += 1

        many_evals   = neval_residual(nlp) > max_eval
        iter_limit   = iter > max_iter
        tired        = many_evals || iter_time > max_time || iter_limit
        optimal      = min(normFx, normGx) ≤ ϵₐ + ϵᵣ*normGx₀
    end


    status = if optimal 
        :first_order
        elseif tired
            if neval_residual(nlp) > max_eval
                :max_eval
            elseif iter_time > max_time
                :max_time
            elseif iter > max_iter
                :max_iter
            else
                :unknown_tired
            end
        else
            :unknown
        end
    
    return objectif, 
    gradient,
    GenericExecutionStats(nlp; 
                    status, 
                    solution = x,
                    objective = normFx^2 / 2,
                    dual_feas = normGx,
                    iter = iter, 
                    elapsed_time = iter_time)
end
