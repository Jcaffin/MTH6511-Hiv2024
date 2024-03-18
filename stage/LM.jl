using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels


function LM_D(nlp        :: AbstractNLSModel;
    fctDk    :: Function =  Andrei,
    x0       :: AbstractVector = nlp.meta.x0, 
    ϵₐ       :: AbstractFloat = 10^(-8),
    ϵᵣ       :: AbstractFloat = 10^(-8),
    η₁       :: AbstractFloat = 1e-3, 
    η₂       :: AbstractFloat = 0.66, 
    σ₁       :: AbstractFloat = 10.0, 
    σ₂       :: AbstractFloat = 0.5,
    ApproxD  :: Bool = true,
    max_eval :: Int = 1000, 
    max_time :: AbstractFloat = 60.,
    max_iter :: Int = typemax(Int64)
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
        Dk = I(n)
    else
        Dk = zeros(n,n)
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
        [:iter, :nf, :obj, :grad, :status, :nd, :λ],
        [Int, Int, Float64, Float64, String, Float64, Float64],
        hdr_override=Dict(
        :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :nd => "‖d‖", :λ => "λ")
        )

    while !(optimal || tired)
        ################# Calcul de d par factorisation QR #################
        A = [Jx; (Dk + λ * I(n))^(1/2)]
        b = [Fx; zeros(n)]

        QR = qr(A)
        d = QR\(-b)

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
            φk₋₁ = Gx

            ######################## Mise à jour #########################
            x    = xp
            Fx   = Fxp
            Jx   = jac_residual(nlp, x)
            Gx   = Jx' * Fx
            normFx   = norm(Fx)
            normGx = norm(Gx)

            ######################## Calcul de D #########################
            yk₋₁ = Jx' * Fx₋₁ - φk₋₁
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

    return GenericExecutionStats(nlp; 
            status, 
            solution = x,
            objective = normFx^2 / 2,
            dual_feas = normGx,
            iter = iter, 
            elapsed_time = iter_time), objectif, gradient
end



function SPG(Dk₋₁, sk₋₁, yk₋₁)
    n   = size(Dk₋₁)[1]
    
    sty = sk₋₁' * yk₋₁
    if sty > 0.01
        σk  = sty /(sk₋₁' * sk₋₁)
        Dk  = σk * I(n)
        return Dk
    else
        return Dk₋₁
    end
end

function Zhu(Dk₋₁, sk₋₁, yk₋₁)
    Sk₋₁ = diagm(sk₋₁)
    tr   = sum(Sk₋₁^4)

    frac  = sk₋₁'*(yk₋₁ - Dk₋₁ * sk₋₁)/tr
    if frac > 0.01
        Dk   = Dk₋₁ + frac * (Sk₋₁^2)
        return Dk
    else
        return Dk₋₁
    end
end

# function Andrei(Dk₋₁, sk₋₁, yk₋₁)
#     n    = size(Dk₋₁)[1]
#     Sk₋₁ = diagm(sk₋₁)
#     min  = minimum(sk₋₁)^2
#     tr   = sum(Sk₋₁^4)

#     frac = sk₋₁'*(yk₋₁ + sk₋₁ - Dk₋₁ * sk₋₁)/tr
#     if frac * min > 1.01
#         Dk   = Dk₋₁ + frac * (Sk₋₁^2) - I(n)
#         @show min(diag(Dk))
#         return Dk
#     else
#         return Dk₋₁
#     end
# end 


function Andrei(Dk₋₁, sk₋₁, yk₋₁; ϵ = 0.01)
    n    = size(Dk₋₁)[1]
    tr   = sum(sk₋₁ .^ 4)
    frac = sk₋₁'*(yk₋₁ + sk₋₁ - Dk₋₁ * sk₋₁)/tr
    Dk   = zeros(n,n)

    for i = 1:n
        Di = Dk₋₁[i,i] + frac * sk₋₁[i]^2 - 1
        if Di > ϵ
            Dk[i,i] = Di
        else
            Dk[i,i] = 1
        end
    end
    return Dk
end 