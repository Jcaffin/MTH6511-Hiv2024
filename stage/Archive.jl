function maj_J(Jx, rows, cols, vals)
    for k = 1:lastindex(rows)
        i = rows[k]
        j = cols[k]
        Jx[i,j] = vals[k]
    end
    return Jx
end

function is_quasi_lin(Fxi, Fx₋₁i, Jx₋₁i, d; τ₃ = 0.01)
    quasi_lin = abs(Fxi - (Fx₋₁i + Jx₋₁i'*d))/(1+abs(Fxi)) < τ₃ ? true : false
    return quasi_lin
end

function is_quasi_nul(Fxi, Fx₋₁i; τ₁ = 0.01, τ₂ = 0.01)
    quasi_nul = abs(Fxi) < τ₁ * abs(Fx₋₁i) + τ₂ ? true : false
    return quasi_nul
end

function argmin_q(Fx, Jx, λ, n, D; δ=0)
    if δ == 0
        A = [Jx; sqrt(λ * I(n))]
    else
        A = [Jx; sqrt(D + λ * I(n))]
    end
    b = [Fx; zeros(n)]
    b .*= -1
    # qrm_init()
    # spmat = qrm_spmat_init(A)
    # x = qrm_least_squares(spmat, b)
    QR = qr(A)
    x = QR\(b)
    return x
end


function LM_D(nlp     :: AbstractNLSModel;
    fctD              :: Function =  Andrei,
    x0                :: AbstractVector = nlp.meta.x0, 
    ϵₐ                :: AbstractFloat = 1e-8,
    ϵᵣ                :: AbstractFloat = 1e-8,
    η₁                :: AbstractFloat = 1e-3, 
    η₂                :: AbstractFloat = 2/3, 
    σ₁                :: AbstractFloat = 10., 
    σ₂                :: AbstractFloat = 1/2,
    approxD           :: Bool = true,
    approxD_quasilin  :: Bool = false,
    disp_grad_obj     :: Bool = false,
    alternative_model :: Bool = false,
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
    r   = similar(Fx)
    Fx₋₁ = similar(Fx)
    JxdFx = similar(Fx)
    dDd   = 0
    # rows, cols = jac_structure_residual(nlp)
    # vals       = jac_coord_residual(nlp, x)
    # Jx         = sparse(rows, cols, vals)
    Jx    = jac_residual(nlp, x)
    Jx₋₁  = similar(Jx)
    Gx    = Jx' * Fx

    normFx₀ = norm(Fx)
    normGx₀ = norm(Gx)
    normGx  = normGx₀
    normFx  = normFx₀

    fx = (1/2) * normFx^2

    if alternative_model
        xᵃ    = similar(x)
        Fxᵃ   = similar(Fx)
    end
    δ    = 0

    m,n = size(Jx)

    yk₋₁   = zeros(n)
    sk₋₁   = zeros(n)
    
    local D
    D = approxD ? Diagonal(ones(n)) : Diagonal(zeros(n))

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
        [:iter, :nf, :obj, :grad, :ρ, :status, :nd, :nD, :λ],
        [Int, Int, Float64, Float64, Float64, String, Float64, Float64, Float64],
        hdr_override=Dict(
        :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :nD => "‖D‖∞", :λ => "λ")
        )

    while !(optimal || tired)
        ########## Calcul d (facto QR) ##########
        d = argmin_q(Fx, Jx, λ, n, D; δ = δ)

        xᵖ     .= x .+ d
        residual!(nlp, xᵖ,Fxᵖ)
        fxᵖ     = (1/2) * norm(Fxᵖ)^2

        ##### sélection du modèle q adéquat #####
        JxdFx .= Jx * d + Fx
        dDd    = d'*D*d
        if alternative_model
            qxᵖ  = (1/2) * (norm(JxdFx)^2 + δ * dDd)
            qᵃxᵖ = (1/2) * (norm(JxdFx)^2 + (1-δ) * dDd)
            if abs(qxᵖ - fxᵖ) > (3/2) * abs(qᵃxᵖ - fxᵖ) 
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
            if approxD_quasilin
                for i = 1:lastindex(Fx)
                    quasi_lin = is_quasi_lin(Fx[i], Fx₋₁[i], Jx₋₁[i,:], d)
                    quasi_nul = is_quasi_nul(Fx[i], Fx₋₁[i])
                    if quasi_lin || quasi_nul
                        r[i] = 0
                    else
                        r[i] = Fx[i]
                    end
                end
                mul!(yk₋₁,Jx',r)
                mul!(yk₋₁,Jx₋₁',r,-1,1)
            else
                mul!(yk₋₁,Jx',Fx)
                mul!(yk₋₁,Jx₋₁',Fx,-1,1)
            end
            sk₋₁ .= x .- x₋₁

            ############ Calcul de D ###########
            if approxD
                D = fctD(D, sk₋₁, yk₋₁)
            end
            
            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        push!(objectif,normFx)
        push!(gradient, normGx)

        @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, ρ, status, norm(d), norm(D,Inf), λ])

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


function SPG(D, s, y; ϵ = 1/100)
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

function Zhu(D, s, y; ϵ = 1/100)
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

function Andrei(D, s, y; ϵ = 1/100)
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


LM_wo_D         = (nlp ; bool_grad_obj=false) -> LM_D(nlp; approxD = false, disp_grad_obj = bool_grad_obj)
# LM_SPG            = (nlp ; bool_grad_obj=false) -> LM_D(nlp; fctD = SPG    , disp_grad_obj = bool_grad_obj)
# LM_Zhu            = (nlp ; bool_grad_obj=false) -> LM_D(nlp; fctD = Zhu    , disp_grad_obj = bool_grad_obj)
# LM_Andrei         = (nlp ; bool_grad_obj=false) -> LM_D(nlp; fctD = Andrei , disp_grad_obj = bool_grad_obj)
# LM_SPG_alt        = (nlp ; bool_grad_obj=false) -> LM_D(nlp; fctD = SPG    , disp_grad_obj = bool_grad_obj, alternative_model = true)
# LM_Zhu_alt        = (nlp ; bool_grad_obj=false) -> LM_D(nlp; fctD = Zhu    , disp_grad_obj = bool_grad_obj, alternative_model = true)
# LM_Andrei_alt     = (nlp ; bool_grad_obj=false) -> LM_D(nlp; fctD = Andrei , disp_grad_obj = bool_grad_obj, alternative_model = true)

