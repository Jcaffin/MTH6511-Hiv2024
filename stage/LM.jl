using LinearAlgebra, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, SparseArrays

function maj_J(Jx, rows, cols, vals)
    for k = 1:length(rows)
        i = rows[k]
        j = cols[k]
        Jx[i,j] = vals[k]
    end
    return Jx
end


function argmin_q(Fx, Jx, λ, n, D; δ=0)
    if δ == 0
        A = [Jx; sqrt(λ * I(n))]
    else
        A = [Jx; sqrt(D + λ * I(n))]
    end
    b = [Fx; zeros(n)]
    b .*= -1
    QR = qr(A)
    d = QR\(b)
    return d
end

function LM_D(nlp  :: AbstractNLSModel;
    fctD           :: Function =  Andrei,
    x0             :: AbstractVector = nlp.meta.x0, 
    ϵₐ             :: AbstractFloat = 1e-8,
    ϵᵣ             :: AbstractFloat = 1e-8,
    η₁             :: AbstractFloat = 1e-3, 
    η₂             :: AbstractFloat = 0.66, 
    σ₁             :: AbstractFloat = 10.0, 
    σ₂             :: AbstractFloat = 0.5,
    ApproxD        :: Bool = true,
    Disp_grad_obj  :: Bool = false,
    max_eval       :: Int = 100000, 
    max_time       :: AbstractFloat = 720.,
    max_iter       :: Int = typemax(Int64)
    )

    ################ On évalue F(x₀) et J(x₀) ################
    x   = copy(x0)
    xᵖ  = similar(x)
    x₋₁ = similar(x)
    Fx  = residual(nlp, x)
    Fxᵖ = similar(Fx)
    rows, cols = jac_structure_residual(nlp)
    vals       = jac_coord_residual(nlp, x)
    Jx         = sparse(rows, cols, vals)
    Jx₋₁       = similar(Jx)
    Gx         = Jx' * Fx

    m,n = size(Jx)

    yk₋₁   = zeros(n)
    sk₋₁   = zeros(n)
    
    local D
    if ApproxD
        D = Diagonal(ones(n))
    else
        D = Diagonal(zeros(n))
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
        d = argmin_q(Fx, Jx, λ, n, D)

        ######################## Calcul du ratio ρ #########################
        xᵖ     .= x .+ d
        residual!(nlp, xᵖ,Fxᵖ)
        normFxᵖ = norm(Fxᵖ)

        ρ = (normFx^2 - normFxᵖ^2) / (normFx^2 - norm(Jx * d + Fx)^2 - d'*D*d)

        if ρ < η₁
            λ = max(λ₀, σ₁ * λ)
            status = :increase_λ
        else
            ############### Stockage des anciennes valeurs ###############
            x₋₁  .= x
            Jx₋₁ .= Jx

            ######################## Mise à jour #########################
            x    .= xᵖ
            Fx   .= Fxᵖ
            jac_coord_residual!(nlp, x, vals)
            Jx   .= maj_J(Jx, rows, cols, vals)
            mul!(Gx,Jx',Fx)
            normFx   = norm(Fx)
            normGx = norm(Gx)

            ######################## Calcul de D #########################
            mul!(yk₋₁,Jx',Fx)
            mul!(yk₋₁,Jx₋₁',Fx,-1,1)
            sk₋₁ .= x .- x₋₁

            if ApproxD
                D = fctD(D, sk₋₁, yk₋₁)
            end

            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        push!(objectif,normFx)
        push!(gradient, normGx)

        @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, status, norm(d), norm(D,Inf), λ])

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
    # @show status
    # @show time() - start_time
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



function LM_Dalternative(nlp        :: AbstractNLSModel;
    fctD          :: Function =  Andrei,
    x0             :: AbstractVector = nlp.meta.x0, 
    ϵₐ             :: AbstractFloat = 1e-8,
    ϵᵣ             :: AbstractFloat = 1e-8,
    η₁             :: AbstractFloat = 1e-3, 
    η₂             :: AbstractFloat = 0.66, 
    σ₁             :: AbstractFloat = 10.0, 
    σ₂             :: AbstractFloat = 0.5,
    ApproxD        :: Bool = true,
    Disp_grad_obj  :: Bool = false,
    max_eval       :: Int  = 1000, 
    max_time       :: AbstractFloat = 60.,
    max_iter       :: Int  = typemax(Int64)
    )

    ################ On évalue F(x₀) et J(x₀) ################

    x   = copy(x0)
    xᵖ  = similar(x)
    x₋₁ = similar(x)
    xᵃ  = similar(x)
    Fx  = residual(nlp, x)
    Fxᵖ = similar(Fx)
    Fxᵃ = similar(Fx)
    Jx  = jac_residual(nlp, x)
    Jx₋₁= similar(Jx)
    Gx  = Jx' * Fx

    normFx₀ = norm(Fx)
    normGx₀ = norm(Gx)
    normGx  = normGx₀
    normFx  = normFx₀

    fx = 0.5* normFx^2
    

    m,n = size(Jx)
    yk₋₁   = zeros(n)
    sk₋₁   = zeros(n)
    
    if ApproxD
        D = Diagonal(ones(n))
    else
        D = Diagonal(zeros(n))
    end

    iter = 0    
    λ    = 0.0
    λ₀   = 1e-6
    δ    = 0

    iter_time = 0.0
    tired   = neval_residual(nlp) > max_eval || iter_time > max_time
    status  = :unknown
    start_time = time()
    optimal    = normGx ≤ ϵₐ + ϵᵣ*normGx₀ || normFx ≤ ϵₐ + ϵᵣ*normFx₀

    #################### Tracé des graphes ###################
    objectif = [normFx]
    gradient = [normGx]

    @info log_header(
        [:iter, :nf, :obj, :grad, :status, :nd, :nD, :λ, :δ],
        [Int, Int, Float64, Float64, String, Float64, Float64, Float64, Int64],
        hdr_override=Dict(
        :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :nd => "‖d‖", :nD => "‖D‖∞", :λ => "λ", :δ => "δ")
        )

    while !(optimal || tired)

        ########################### Calcul de xᵖ ###########################
        d = argmin_q(Fx, Jx, λ, n, D; δ = δ)
        
        xᵖ     .= x .+ d
        residual!(nlp, xᵖ, Fxᵖ)
        fxᵖ     = 0.5* norm(Fxᵖ)^2

        # valeur absolue d'une composante de Jx * d + Fx pour le test composante par composante
        # Jx * d + Fx à précalculer
        # d'*D*d à précalculer
        qxᵖ  = 0.5 * (norm(Jx * d + Fx)^2 + δ * d'*D*d)
        qᵃxᵖ = 0.5 * (norm(Jx * d + Fx)^2 + (1-δ) * d'*D*d)

        if abs(qxᵖ - fxᵖ) > 1.5 * abs(qᵃxᵖ - fxᵖ) 
            xᵃ .= x .+ argmin_q(Fx, Jx, λ, n, D; δ = 1-δ)
            residual!(nlp, xᵃ, Fxᵃ)
            fxᵃ = (1/2)* norm(Fxᵃ)^2
            if fxᵃ < fxᵖ
                δ = 1-δ
                xᵖ  .= xᵃ
                Fxᵖ .= Fxᵃ
                fxᵖ  = fxᵃ
                qxᵖ  = qᵃxᵖ
            end
        end

        ######################## Calcul du ratio ρ #########################
        

        ρ = (fx - fxᵖ) / (fx - qxᵖ)

        if ρ < η₁
            λ = max(λ₀, σ₁ * λ)
            status = :increase_λ
        else
            ############### Stockage des anciennes valeurs ###############
            x₋₁  .= x
            Jx₋₁ .= Jx

            ######################## Mise à jour #########################
            x    .= xᵖ
            Fx   .= Fxᵖ
            Jx    = jac_residual(nlp, x)
            mul!(Gx, Jx', Fx)
            normFx  = norm(Fx)
            normGx  = norm(Gx)
            fx      = 0.5 * normFx^2

            ######################## Calcul de D #########################
            mul!(yk₋₁,Jx',Fx)
            mul!(yk₋₁,Jx₋₁',Fx,-1,1)
            sk₋₁ .= x .- x₋₁

            if ApproxD
                D = fctD(D, sk₋₁, yk₋₁)
            end

            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        push!(objectif,normFx)
        push!(gradient, normGx)

        @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, status, norm(d), norm(D,Inf), λ, δ])

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


LM            = (nlp ; bool=false) -> LM_D(nlp; ApproxD = false, Disp_grad_obj = bool)
LM_SPG        = (nlp ; bool=false) -> LM_D(nlp; fctD = SPG    , Disp_grad_obj = bool)
LM_Zhu        = (nlp ; bool=false) -> LM_D(nlp; fctD = Zhu    , Disp_grad_obj = bool)
LM_Andrei     = (nlp ; bool=false) -> LM_D(nlp; fctD = Andrei , Disp_grad_obj = bool)
LM_SPG_alt    = (nlp ; bool=false) -> LM_Dalternative(nlp; fctD = SPG    , Disp_grad_obj = bool)
LM_Zhu_alt    = (nlp ; bool=false) -> LM_Dalternative(nlp; fctD = Zhu    , Disp_grad_obj = bool)
LM_Andrei_alt = (nlp ; bool=false) -> LM_Dalternative(nlp; fctD = Andrei , Disp_grad_obj = bool)
