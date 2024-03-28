using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels


function LM_D(nlp        :: AbstractNLSModel;
    fctDk          :: Function =  Andrei,
    x0             :: AbstractVector = nlp.meta.x0, 
    ϵₐ             :: AbstractFloat = 1e-8,
    ϵᵣ             :: AbstractFloat = 1e-8,
    η₁             :: AbstractFloat = 1e-3, 
    η₂             :: AbstractFloat = 0.66, 
    σ₁             :: AbstractFloat = 10.0, 
    σ₂             :: AbstractFloat = 0.5,
    ApproxD        :: Bool = true,
    Disp_grad_obj  :: Bool = false,
    max_eval       :: Int = 1000, 
    max_time       :: AbstractFloat = 60.,
    max_iter       :: Int = typemax(Int64)
    )

    ################ On évalue F(x₀) et J(x₀) ################
    x = copy(x0)
    Fx = residual(nlp, x)
    Jx = jac_residual(nlp, x)
    Gx = Jx' * Fx

    m,n = size(Jx)

    yk₋₁   = zeros(n)
    sk₋₁   = zeros(n)
    
    if ApproxD
        Dk = Diagonal(ones(n))
    else
        Dk = Diagonal(zeros(n))
    end

    ################## On calcule leur norme #################
    normFx₀ = norm(Fx)
    normGx₀ = norm(Gx)
    normGx  = normGx₀
    normFx  = normFx₀

    iter = 0    
    λ = 0.0
    λ₀ = 1e-6

    iter_time = 0.0
    tired   = neval_residual(nlp) > max_eval || iter_time > max_time
    status  = :unknown
    start_time = time()
    optimal    = normGx ≤ ϵₐ + ϵᵣ*normGx₀ || normFx ≤ ϵₐ + ϵᵣ*normFx₀

    #################### Tracé des graphes ###################
    objectif = [normFx]
    gradient = [normGx]

    @info log_header(
        [:iter, :nf, :obj, :grad, :status, :nd, :nD, :λ],
        [Int, Int, Float64, Float64, String, Float64, Float64, Float64],
        hdr_override=Dict(
        :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :nd => "‖d‖", :nD => "‖D‖∞", :λ => "λ")
        )

    while !(optimal || tired)
        ################# Calcul de d par factorisation QR #################
        A = [Jx; (Dk + λ * I)^(1/2)]
        b = [Fx; zeros(n)]
        b .= -b

        QR = qr(A)
        d = QR\(b)

        ######################## Calcul du ratio ρ #########################
        xp      = x + d
        Fxp     = residual(nlp, xp)
        normFxp = norm(Fxp)

        ρ = (normFx^2 - normFxp^2) / (normFx^2 - norm(Jx * d + Fx)^2 - d'*Dk*d)

        if ρ < η₁
            λ = max(λ₀, σ₁ * λ)
            status = :increase_λ
        else
            ############### Stockage des anciennes valeurs ###############
            x₋₁  = x
            Fx₋₁ = Fx
            Jx₋₁ = Jx
            φk₋₁ = Gx

            ######################## Mise à jour #########################
            x    = xp
            Fx   = Fxp
            Jx   = jac_residual(nlp, x)
            Gx   = Jx' * Fx
            normFx   = norm(Fx)
            normGx = norm(Gx)

            ######################## Calcul de D #########################
            # yk₋₁ = Jx' * Fx₋₁ - φk₋₁
            yk₋₁ = Jx' * Fx - Jx₋₁' * Fx
            sk₋₁ = x - x₋₁

            if ApproxD
                Dk = fctDk(Dk, sk₋₁, yk₋₁)
            end

            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        push!(objectif,normFx)
        push!(gradient, normGx)

        @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, status, norm(d), norm(Dk,Inf), λ])

        iter_time    = time() - start_time
        iter        += 1

        many_evals   = neval_residual(nlp) > max_eval
        iter_limit   = iter > max_iter
        tired        = many_evals || iter_time > max_time || iter_limit
        optimal      = normGx ≤ ϵₐ + ϵᵣ*normGx₀ || normFx ≤ ϵₐ + ϵᵣ*normFx₀
        
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
    
    if Disp_grad_obj
        return GenericExecutionStats(nlp; 
            status, 
            solution = x,
            objective = normFx^2 / 2,
            dual_feas = normGx,
            iter = iter, 
            elapsed_time = iter_time), objectif, gradient
    else
        return GenericExecutionStats(nlp; 
            status, 
            solution = x,
            objective = normFx^2 / 2,
            dual_feas = normGx,
            iter = iter, 
            elapsed_time = iter_time) 
    end
end



function SPG(D, s, y; ϵ = 0.01)
    n   = size(D,1)
    sty = s' * y
    ss  = s' * s
    if sty > ϵ
        σ  = sty /ss
        for i = 1:n
            D[i,i]  = σ
        end
        return D
    else
        return D
    end
end

function Zhu(D, s, y; ϵ = 0.01)
    n    = size(D,1)
    tr   = sum(si^4 for si ∈ s)
    sy   = s' * y
    sDs = sum(s[i]^2 * D[i, i] for i = 1 : n)
    frac  = (sy - sDs) / tr
    for i = 1:n
        Di = D[i,i] + frac * s[i]^2
        if Di > ϵ
            D[i,i] = Di
        else
            D[i,i] = 1
        end
    end
    return D
end 

function Andrei(D, s, y; ϵ = 0.01)
    n    = size(D)[1]
    tr   = sum(si^4 for si ∈ s)
    sy   = s' * y 
    ss   = s' * s
    sDs  = s' * D * s
    frac = (sy + ss - sDs)/tr

    for i = 1:n
        Di = D[i,i] + frac * s[i]^2 - 1
        if Di > ϵ
            D[i,i] = Di
        else
            D[i,i] = 1
        end
    end
    return D
end 


LM        = (nlp ; bool=false) -> LM_D(nlp; ApproxD = false, Disp_grad_obj = bool)
LM_SPG    = (nlp ; bool=false) -> LM_D(nlp; fctDk = SPG    , Disp_grad_obj = bool)
LM_Zhu    = (nlp ; bool=false) -> LM_D(nlp; fctDk = Zhu    , Disp_grad_obj = bool)
LM_Andrei = (nlp ; bool=false) -> LM_D(nlp; fctDk = Andrei , Disp_grad_obj = bool)