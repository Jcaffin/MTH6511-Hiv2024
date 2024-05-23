
function SPG_test(D, s, y; ϵ = 1/100)
    n   = size(D,1)
    sty = s' * y
    ss  = s' * s
    non_nul = sty > ϵ ? true : false
    if non_nul
        σ  = sty /ss
        for i = 1:n
            D[i,i]  = σ
        end
    end
    return D
end

function Zhu_test(D, s, y; ϵ = 1/100)
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

function Andrei_test(D, s, y; ϵ = 1/100)
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

function is_quasi_nul_test(Fxi, Fx₋₁i; τ₁ = 0.01, τ₂ = 0.01)
    quasi_nul = abs(Fxi) < τ₁ * abs(Fx₋₁i) + τ₂ ? true : false
    return quasi_nul
end

function is_quasi_lin_test(Fxi, Fx₋₁i, Jx₋₁i, d; τ₃ = 0.01)
    quasi_lin = abs(Fxi - (Fx₋₁i + Jx₋₁i'*d))/(1+abs(Fxi)) < τ₃ ? true : false
    return quasi_lin
end

function LM_tst(nlp     :: AbstractNLSModel;
    x0                :: AbstractVector = nlp.meta.x0, 
    ϵₐ                :: AbstractFloat = 1e-8,
    ϵᵣ                :: AbstractFloat = 1e-8,
    η₁                :: AbstractFloat = 1e-3, 
    η₂                :: AbstractFloat = 2/3, 
    σ₁                :: AbstractFloat = 10., 
    σ₂                :: AbstractFloat = 1/2,
    disp_grad_obj     :: Bool = false,
    max_eval          :: Int = 1000, 
    max_time          :: AbstractFloat = 60.,
    max_iter          :: Int = typemax(Int64)
    )

    ################ On évalue F(x₀) et J(x₀) ################
    x   = copy(x0)
    xᵖ  = similar(x)
    x₋₁ = similar(x)
    Fx  = residual(nlp, x)
    Fxᵖ = similar(Fx)
    Fx₋₁ = similar(Fx)

    Jx    = jac_residual(nlp, x)
    Jx₋₁  = similar(Jx)
    Gx    = Jx' * Fx

    normFx₀ = norm(Fx)
    normGx₀ = norm(Gx)
    normGx  = normGx₀
    normFx  = normFx₀

    fx = (1/2) * normFx^2
    m,n = size(Jx)

    iter = 0    
    λ = 0
    λ₀ = 1e-6

    iter_time = 0
    tired   = neval_residual(nlp) > max_eval || iter_time > max_time
    status  = :unknown
    start_time = time()
    optimal    = normGx ≤ ϵₐ + ϵᵣ*normGx₀ || normFx ≤ ϵₐ + ϵᵣ*normFx₀

    #################### Tracé des graphes ###################
    objectif = [normFx]
    gradient = [normGx]

    @info log_header(
        [:iter, :nf, :obj, :grad, :ρ, :status, :nd, :λ],
        [Int, Int, Float64, Float64, Float64, String, Float64, Float64],
        hdr_override=Dict(
        :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ")
        )

    while !(optimal || tired)
        ########## Calcul d (facto QR) ##########
        A = [Jx; sqrt(λ * I(n))]
        b = [Fx; zeros(n)]
        b .*= -1
        QR = qr(A)
        d = QR\(b)

        xᵖ     .= x .+ d
        residual!(nlp, xᵖ,Fxᵖ)
        fxᵖ  = (1/2) * norm(Fxᵖ)^2
        qxᵖ  = (1/2) * (norm(Jx * d + Fx)^2)


        ρ = (fx - fxᵖ) / (fx - qxᵖ)

        if ρ < η₁
            λ = max(λ₀, σ₁ * λ)
            status = :increase_λ
        else
            #### Stockage anciennes valeurs #####
            x₋₁  .= x
            Jx₋₁ .= Jx
            Fx₋₁ .= Fx

            ############ Mise à jour ############
            x    .= xᵖ
            Fx   .= Fxᵖ
            # jac_coord_residual!(nlp, x, vals)
            # Jx   .= maj_J(Jx, rows, cols, vals)
            Jx    = jac_residual(nlp, x)
            mul!(Gx,Jx',Fx)
            normFx = norm(Fx)
            normGx = norm(Gx)
            fx      = (1/2) * normFx^2
            
            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        push!(objectif,normFx)
        push!(gradient, normGx)

        @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, ρ, status, norm(d), λ])

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

    if disp_grad_obj
        return GenericExecutionStats(nlp; 
            status, 
            solution = x,
            objective = (1/2) * normFx^2,
            dual_feas = normGx,
            iter = iter, 
            elapsed_time = iter_time), objectif, gradient
    else
        return GenericExecutionStats(nlp; 
            status, 
            solution = x,
            objective = (1/2) * normFx^2,
            dual_feas = normGx,
            iter = iter, 
            elapsed_time = iter_time) 
    end
end


function LM_D_tst(nlp     :: AbstractNLSModel;
    x0                :: AbstractVector = nlp.meta.x0, 
    fctD              :: Function =  Andrei_test,
    is_NW             :: Bool = true,
    ϵₐ                :: AbstractFloat = 1e-8,
    ϵᵣ                :: AbstractFloat = 1e-8,
    η₁                :: AbstractFloat = 1e-3, 
    η₂                :: AbstractFloat = 2/3, 
    σ₁                :: AbstractFloat = 10., 
    σ₂                :: AbstractFloat = 1/2,
    γ₁                :: AbstractFloat = 3/2,
    τ₁                :: AbstractFloat = 1/100,
    τ₂                :: AbstractFloat = 1/100,
    τ₃                :: AbstractFloat = 1/100,
    alternative_model :: Bool = false,
    disp_grad_obj     :: Bool = false,
    max_eval          :: Int = 1000, 
    max_time          :: AbstractFloat = 60.,
    max_iter          :: Int = typemax(Int64)
    )

    ################ On évalue F(x₀) et J(x₀) ################
    x   = copy(x0)
    xᵖ  = similar(x)
    x₋₁ = similar(x)
    Fx  = residual(nlp, x)
    Fxᵖ = similar(Fx)
    Fx₋₁ = similar(Fx)
    JxdFx = similar(Fx)
    dDd   = 0
    if alternative_model
        xᵃ    = similar(x)
        Fxᵃ   = similar(Fx)
    end
    δ    = 0

    Jx    = jac_residual(nlp, x)
    Jx₋₁  = similar(Jx)
    Gx    = Jx' * Fx

    normFx₀ = norm(Fx)
    normGx₀ = norm(Gx)
    normGx  = normGx₀
    normFx  = normFx₀
    

    fx = (1/2) * normFx^2
    m,n = size(Jx)

    yk₋₁   = zeros(n)
    sk₋₁   = zeros(n)

    iter = 0    
    λ = 0
    λ₀ = 1e-6

    local D
    D = Diagonal(ones(n))

    iter_time = 0
    tired   = neval_residual(nlp) > max_eval || iter_time > max_time
    status  = :unknown
    start_time = time()
    optimal    = normGx ≤ ϵₐ + ϵᵣ*normGx₀ || normFx ≤ ϵₐ + ϵᵣ*normFx₀

    #################### Tracé des graphes ###################
    objectif = [normFx]
    gradient = [normGx]

    @info log_header(
        [:iter, :nf, :obj, :grad, :ρ, :status, :nd, :λ],
        [Int, Int, Float64, Float64, Float64, String, Float64, Float64],
        hdr_override=Dict(
        :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ")
        )

    while !(optimal || tired)
        ########## Calcul d (facto QR) ##########
        A = [Jx; sqrt(D + λ * I(n))]
        b = [Fx; zeros(n)]
        b .*= -1
        QR = qr(A)
        d = QR\(b)

        xᵖ     .= x .+ d
        residual!(nlp, xᵖ,Fxᵖ)
        fxᵖ  = (1/2) * norm(Fxᵖ)^2
        
        ##### sélection du modèle q adéquat #####
        JxdFx .= Jx * d + Fx
        dDd    = d'*D*d
        if alternative_model
            qxᵖ  = (1/2) * (norm(JxdFx)^2 + δ * dDd)
            qᵃxᵖ = (1/2) * (norm(JxdFx)^2 + (1-δ) * dDd)
            if abs(qxᵖ - fxᵖ) > γ₁ * abs(qᵃxᵖ - fxᵖ) 
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
        else
            qxᵖ  = (1/2) * (norm(JxdFx)^2 + dDd)
        end


        ρ = (fx - fxᵖ) / (fx - qxᵖ)

        if ρ < η₁
            λ = max(λ₀, σ₁ * λ)
            status = :increase_λ
        else
            #### Stockage anciennes valeurs #####
            x₋₁  .= x
            Jx₋₁ .= Jx
            Fx₋₁ .= Fx

            ############ Mise à jour ############
            x    .= xᵖ
            Fx   .= Fxᵖ
            # jac_coord_residual!(nlp, x, vals)
            # Jx   .= maj_J(Jx, rows, cols, vals)
            Jx    = jac_residual(nlp, x)
            mul!(Gx,Jx',Fx)
            normFx = norm(Fx)
            normGx = norm(Gx)
            fx      = (1/2) * normFx^2

            ##### Maj yk₋₁ pour calcul de D #####
            if is_NW
                mul!(yk₋₁,Jx',Fx)
                mul!(yk₋₁,Jx₋₁',Fx,-1,1)
            else
                mul!(yk₋₁,Jx',Fx₋₁)
                mul!(yk₋₁,Jx₋₁',Fx₋₁,-1,1)
            end
            sk₋₁ .= x .- x₋₁

            D = fctD(D, sk₋₁, yk₋₁)
            
            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        push!(objectif,normFx)
        push!(gradient, normGx)

        @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, ρ, status, norm(d), λ])

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
    @show status

    if disp_grad_obj
        return GenericExecutionStats(nlp; 
            status, 
            solution = x,
            objective = (1/2) * normFx^2,
            dual_feas = normGx,
            iter = iter, 
            elapsed_time = iter_time), objectif, gradient
    else
        return GenericExecutionStats(nlp; 
            status, 
            solution = x,
            objective = (1/2) * normFx^2,
            dual_feas = normGx,
            iter = iter, 
            elapsed_time = iter_time) 
    end
end


LM_test          = (nlp ; bool_grad_obj=false) -> LM_tst(nlp; disp_grad_obj = bool_grad_obj)
LM_D_y_diese     = (nlp ; bool_grad_obj=false) -> LM_D_tst(nlp; fctD = Andrei_test    , disp_grad_obj = bool_grad_obj)
LM_D_y_tilde     = (nlp ; bool_grad_obj=false) -> LM_D_tst(nlp; fctD = Andrei_test, is_NW = false, disp_grad_obj = bool_grad_obj)
LM_SPG           = (nlp ; bool_grad_obj=false) -> LM_D_tst(nlp; fctD = SPG_test    , disp_grad_obj = bool_grad_obj)
LM_Zhu           = (nlp ; bool_grad_obj=false) -> LM_D_tst(nlp; fctD = Zhu_test    , disp_grad_obj = bool_grad_obj)
LM_Andrei        = (nlp ; bool_grad_obj=false) -> LM_D_tst(nlp; fctD = Andrei_test , disp_grad_obj = bool_grad_obj)
LM_SPG_alt       = (nlp ; bool_grad_obj=false) -> LM_D_tst(nlp; fctD = SPG_test,    alternative_model = true, disp_grad_obj = bool_grad_obj)
LM_Zhu_alt       = (nlp ; bool_grad_obj=false) -> LM_D_tst(nlp; fctD = Zhu_test,    alternative_model = true, disp_grad_obj = bool_grad_obj)
LM_Andrei_alt    = (nlp ; bool_grad_obj=false) -> LM_D_tst(nlp; fctD = Andrei_test, alternative_model = true, disp_grad_obj = bool_grad_obj)