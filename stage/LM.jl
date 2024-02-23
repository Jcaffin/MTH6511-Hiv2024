using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels

function linear_system_QR(A ::AbstractMatrix, b:: AbstractVector)
    Q, R = qr(A)
    y = Q' * b

    dsize = size(A, 2)
    d = zeros(dsize)

    ######### initialisation : dernière composante de d #########
    d[end] = y[end]/R[end,end]

    #################### autres composantes #####################
    for i = 1 : dsize-1
        k = dsize - i
        somme = 0
        for j = k+1 : dsize
            somme += R[k,j]*d[j]
        end
        d[k] = (y[k] - somme)/R[k,k]
    end
    
    return d
end


function dsol(Fx, Jx, λ)
    Jx_mat = Matrix(Jx)     # car Jx matrice creuse type spécial : SparseArrays.SparseMatrixCSC
    p = size(Jx, 2)

    A = [Jx_mat; λ * I(p)]
    b = [Fx; zeros(p)]
    
    d = linear_system_QR(A,b)
    return d
end


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
    Jx = jac_residual(nlp, x)   #jac_op_residual pas fou fou pour QR

    ################## On calcule leur norme #################
    normFx   = norm(Fx)
    normdual = norm(Jx' * Fx)
    normdual₀ = normdual

    iter = 0    
    λ = 0.0
    λ₀ = 1e-6

    iter_time = 0.0
    tired   = neval_residual(nlp) > max_eval || iter_time > max_time
    status  = :unknown

    start_time = time()
    optimal    = min(normFx, normdual) ≤ ϵₐ + ϵᵣ*normdual₀

    @info log_header([:iter, :nf, :primal, :dual, :status, :nd, :λ],
    [Int, Int, Float64, Float64, String, Float64, Float64],
    hdr_override=Dict(:nf => "#F", :primal => "‖F(x)‖", :dual => "‖∇F(x)‖", :nd => "‖d‖"))

    while !(optimal || tired)
        ################# Calcul de d par factorisation QR #################
        d = dsol(Fx, Jx, λ)

        ######################## Calcul du ratio ρ ########################
        xp      = x + d
        Fxp     = residual(nlp, xp)
        normFxp = norm(Fxp)

        ρ = (0.5 * (normFx^2 - normFxp^2)) / (0.5 * (normFx^2 - norm(Jx * d + Fx)^2 - λ*norm(d)^2))
        
        if ρ < η₁
            λ = max(λ₀, σ₁ * λ)
            status = :increase_λ
        else
            x   = xp
            Fx  = residual(nlp, x)
            Jx  = jac_residual(nlp, x)
            normFx   = norm(Fx)
            normdual = norm(Jx' * Fx)
            status = :success
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        @info log_row(Any[iter, neval_residual(nlp), normFx, normdual, status, norm(d), λ])

        iter_time    = time() - start_time
        iter        += 1

        many_evals   = neval_residual(nlp) > max_eval
        iter_limit   = iter > max_iter
        tired        = many_evals || iter_time > max_time || iter_limit
        optimal      = min(normFx, normdual) ≤ ϵₐ + ϵᵣ*normdual₀
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
                    dual_feas = normdual,
                    iter = iter, 
                    elapsed_time = iter_time)
end
