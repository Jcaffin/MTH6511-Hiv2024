include("LM.jl")

function LM_D(nlp     :: AbstractNLSModel;
    x0                :: AbstractVector = nlp.meta.x0, 
    fctD              :: Function =  Andrei,
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
    alternative_model      :: Bool = false,
    approxD_quasi_nul_lin  :: Bool = false,
    disp_grad_obj     :: Bool = false,
    max_eval          :: Int = 100000, 
    max_time          :: AbstractFloat = 3600.,
    max_iter          :: Int = typemax(Int64)
    )

    ################ On évalue F(x₀) et J(x₀) ################
    x   = copy(x0)
    xᵖ  = similar(x)
    x₋₁ = similar(x)
    d = similar(x)
    Fx  = residual(nlp, x)
    Fxᵖ = similar(Fx)
    Fx₋₁ = similar(Fx)
    # rows, cols = jac_structure_residual(nlp)
    # vals       = jac_coord_residual(nlp, x)
    # Jx         = sparse(rows, cols, vals)
    Jx    = jac_residual(nlp, x)
    Jx₋₁  = similar(Jx)
    Gx    = Jx' * Fx
    #### ajout alternative_model ####
    JxdFx = similar(Fx)
    dDd   = 0
    if alternative_model
        xᵃ    = similar(x)
        Fxᵃ   = similar(Fx)
    end
    δ    = 0
    ###### ajout quasi_lin_nul ######
    r   = similar(Fx)

    

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

  
        ########## Calcul d (facto QR) ##########
        A = [Jx; sqrt(D + λ * I(n))]
        b = [Fx; zeros(n)]
        b .*= -1
        # qrm_init()
        # spmat = qrm_spmat_init(A)
        # x = qrm_least_squares(spmat, b)
        QR = qr(A)
        d .= QR\(b)


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
            if approxD_quasi_nul_lin
                for i = 1:lastindex(Fx)
                    quasi_nul = is_quasi_nul(Fx[i], Fx₋₁[i], τ₁, τ₂)
                    quasi_lin = is_quasi_lin(Fx[i], Fx₋₁[i], Jx₋₁[i,:], d, τ₃)
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

            D = fctD(D, sk₋₁, yk₋₁)
            
            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        push!(objectif, normFx)
        push!(gradient, normGx)

        @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, ρ, status, norm(d), λ])

        iter_time    = time() - start_time
        iter        += 1

        many_evals   = neval_residual(nlp) > max_eval
        iter_limit   = iter > max_iter
        tired        = many_evals || iter_time > max_time || iter_limit
        optimal      = normGx ≤ ϵₐ + ϵᵣ*normGx₀ || normFx ≤ ϵₐ + ϵᵣ*normFx₀
        

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