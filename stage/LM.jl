using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels

function LM(nlp          :: AbstractNLSModel; 
                x        :: AbstractVector = nlp.meta.x0, 
                ϵₐ       :: AbstractFloat = 10^(-8),
                ϵᵣ       :: AbstractFloat = 10^(-8),
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
        n = size(Jx, 2)
        A = [Jx; λ * I(n)]
        b = [Fx; zeros(n)]
        
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


function LM_D(nlp        :: AbstractNLSModel; 
    x        :: AbstractVector = nlp.meta.x0, 
    ϵₐ       :: AbstractFloat = 10^(-8),
    ϵᵣ       :: AbstractFloat = 10^(-8),
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
    Gx = Jx' * Fx

    m,n = size(Jx)

    x₋₁    = zeros(n)
    Fx₋₁   = zeros(m)
    φk₋₁   = zeros(n)
    yk₋₁   = zeros(n)
    sk₋₁   = zeros(n)
    Dk     = I(n)


    ################## On calcule leur norme #################
    normFx   = norm(Fx)
    normGx₀ = norm(Gx)
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
        [:iter, :nf, :obj, :grad, :status, :sty, :nd, :λ],
        [Int, Int, Float64, Float64, String, Float64, Float64, Float64],
        hdr_override=Dict(
        :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :sty => "sᵀy", :nd => "‖d‖", :λ => "λ")
        )

    while !(optimal || tired)
        ########################### Calcul de D ############################
        yk₋₁ = Jx' * Fx₋₁ - φk₋₁
        sk₋₁ = x - x₋₁
        sty  = sk₋₁' * yk₋₁

        if sty > 0  ###### à l'itération 1, sty = 0 donc Dk = I(n)
            σk = sty /(sk₋₁' * sk₋₁)
            Dk = σk * I(n)
        end
            

        ################# Calcul de d par factorisation QR #################
        A = [Jx; (Dk + λ * I(n))^(1/2)]
        b = [Fx; zeros(n)]

        QR = qr(A)
        d = QR\(-b)

        ######################## Calcul du ratio ρ #########################
        xp      = x + d
        Fxp     = residual(nlp, xp)
        normFxp = norm(Fxp)

        ρ = (normFx^2 - normFxp^2) / (normFx^2 - norm(Jx * d + Fx)^2)

        if ρ < η₁
            λ = max(λ₀, σ₁ * λ)
            status = :increase_λ
        else
            ############### Stockage des anciennes valeurs ###############
            x₋₁  = x
            Fx₋₁ = Fx
            φk₋₁ = Gx


            ######################## Mise à jour #########################
            x    = xp
            Fx   = Fxp
            Jx   = jac_residual(nlp, x)
            Gx   = Jx' * Fx
            normFx   = norm(Fx)
            normGx = norm(Gx)
            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        push!(objectif,Fxp)
        push!(gradient, normGx)

        @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, status, sty, norm(d), λ])

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

