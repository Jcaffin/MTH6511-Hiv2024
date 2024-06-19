function maj_J(Jx, Jrows, Jcols, Jvals)
    for k = 1:lastindex(Jrows)
        i = Jrows[k]
        j = Jcols[k]
        Jx[i,j] = Jvals[k]
    end
    return Jx
end

function argmin_q2!(Arows, Acols, Avals, b, Fx, Jvals, sqrt_DλI, d, λ, D, m, n, nnzj, δ, is_λD)
    for i = 1:n
        if is_λD
            sqrt_DλI[i] = sqrt(δ * λ * D[i,i])
        else
            sqrt_DλI[i] = sqrt(δ * D[i,i] + λ)
        end
    end
    Avals[1:nnzj]       .= Jvals
    Avals[nnzj + 1:end] .= sqrt_DλI
    b[1:m] .= Fx
    b     .*= -1
    spmat = qrm_spmat_init(m+n, n, Arows, Acols, Avals)
    spfct = qrm_analyse(spmat)
    qrm_factorize!(spmat, spfct)
    z = qrm_apply(spfct, b, transp='t')
    d .= qrm_solve(spfct, z, transp='n')
end

function SPG(D, s, y, n, ϵₜ)
    sty = s' * y
    ss  = s' * s
    non_nul = sty > ϵₜ ? true : false
    if non_nul
        σ  = sty /ss
        for i = 1:n
            D[i,i]  = σ
        end
    end
    return D
end

function Zhu(D, s, y, n, ϵₜ)
    tr   = sum(si^4 for si ∈ s)
    sy   = s' * y
    sDs = sum(s[i]^2 * D[i, i] for i = 1 : n)
    frac  = (sy - sDs) / tr
    for i = 1:n
        Di = D[i,i] + frac * s[i]^2
        if Di > ϵₜ
            D[i,i] = Di
        else
            D[i,i] = 1
        end
    end
    return D
end 

function Andrei(D, s, y, n, ϵₜ)
    tr   = sum(si^4 for si ∈ s)
    sy   = s' * y 
    ss   = s' * s
    sDs  = s' * D * s
    frac = (sy + ss - sDs)/tr

    for i = 1:n
        Di = D[i,i] + frac * s[i]^2 - 1
        if Di > ϵₜ
            D[i,i] = Di
        else
            D[i,i] = 1
        end
    end
    return D
end 

function is_quasi_nul(Fxi, Fx₋₁i, τ₁, τ₂)
    return abs(Fxi) < τ₁ * abs(Fx₋₁i) + τ₂
end

function is_quasi_lin(Fxi, Fx₋₁i, Jx₋₁i, d, τ₃, τ₄)
    return abs(Fxi - (Fx₋₁i + Jx₋₁i'*d)) < τ₃ * abs(Fxi) + τ₄
end

function LM_tst(nlp   :: AbstractNLSModel;
    x0                :: AbstractVector = nlp.meta.x0, 
    ϵₐ                :: AbstractFloat = 1e-8,
    ϵᵣ                :: AbstractFloat = 1e-8,
    η₁                :: AbstractFloat = 1e-3, 
    η₂                :: AbstractFloat = 2/3, 
    σ₁                :: AbstractFloat = 10., 
    σ₂                :: AbstractFloat = 1/2,
    disp_grad_obj     :: Bool = false,
    verbose           :: Bool = false,
    max_eval          :: Int = 100000, 
    max_time          :: AbstractFloat = 3600.,
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

    fx = normFx^2 / 2
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

    verbose && @info log_header(
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
        fxᵖ  = norm(Fxᵖ)^2 / 2
        qxᵖ  = (norm(Jx * d + Fx)^2) / 2


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
            # jac_coord_residual!(nlp, x, Jvals)
            # Jx   .= maj_J(Jx, Jrows, Jcols, Jvals)
            Jx    = jac_residual(nlp, x)
            mul!(Gx,Jx',Fx)
            normFx = norm(Fx)
            normGx = norm(Gx)
            fx      = normFx^2 / 2
            
            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        push!(objectif,normFx)
        push!(gradient, normGx)

        verbose && @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, ρ, status, norm(d), λ])

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

function LM_D(nlp     :: AbstractNLSModel;
    x0                :: AbstractVector = nlp.meta.x0, 
    fctD              :: Function =  Andrei,
    ϵₐ                :: AbstractFloat = 1e-8,
    ϵᵣ                :: AbstractFloat = 1e-8,
    ϵₜ                 :: AbstractFloat = 1/100,
    η₁                :: AbstractFloat = 1e-3, 
    η₂                :: AbstractFloat = 2/3, 
    σ₁                :: AbstractFloat = 10., 
    σ₂                :: AbstractFloat = 1/2,
    γ₁                :: AbstractFloat = 3/2,
    τ₁                :: AbstractFloat = 1/100,
    τ₂                :: AbstractFloat = 1/100,
    τ₃                :: AbstractFloat = 1/100,
    τ₄                :: AbstractFloat = 1/100,
    alternative_model      :: Bool = false,
    approxD_quasi_nul_lin  :: Bool = false,
    disp_grad_obj          :: Bool = false,
    is_λD                  :: Bool = false,
    verbose                :: Bool = false,
    max_eval          :: Int = 1000, 
    max_time          :: AbstractFloat = 60.,
    max_iter          :: Int = typemax(Int64)
    )
    filename = "./stage/stockage_sparse"
    ################ On évalue F(x₀) et J(x₀) ################
    m,n  = nlp.nls_meta.nequ, nlp.meta.nvar

    x    = copy(x0)
    xᵖ   = similar(x)
    x₋₁  = similar(x)
    d    = similar(x)
    Fx   = residual(nlp, x)
    Fxᵖ  = similar(Fx)
    Fx₋₁ = similar(Fx)
    Jrows, Jcols = jac_structure_residual(nlp)
    Jvals        = jac_coord_residual(nlp, x)
    Jx           = sparse(Jrows, Jcols, Jvals, m, n)
    nnzj         = nnz(Jx)
    Jx₋₁  = similar(Jx)
    Gx    = Jx' * Fx
    #### ajout alternative_model ####
    JxdFx = similar(Fx)
    dDd   = 0
    if alternative_model
        xᵃ    = similar(x)
        Fxᵃ   = similar(Fx)
    end
    δ = 1
    ###### ajout quasi_lin_nul ######
    r = similar(Fx)

    normFx₀ = norm(Fx)
    normGx₀ = norm(Gx)
    normGx  = normGx₀
    normFx  = normFx₀

    fx  = normFx^2 / 2
    iter = 0 
    λ₀ = 1e-6   
    λ = is_λD ? 1 : λ₀

    #pré-allocations
    yk₋₁ = zeros(n)
    sk₋₁ = zeros(n)
    Arows              = zeros(Int64, nnzj + n)
    Arows[1:nnzj]     .= Jrows 
    Arows[nnzj+1:end] .= [k for k = m+1:m+n]
    Acols              = zeros(Int64, nnzj + n)
    Acols[1:nnzj]     .= Jcols 
    Acols[nnzj+1:end] .= [k for k = 1:n]
    Avals              = zeros(Float64, nnzj + n)

    sqrt_DλI = ones(n)
    b        = zeros(Float64, m+n)
    qrm_init()


    local D
    D = Diagonal(ones(n))

    iter_time  = 0
    tired      = neval_residual(nlp) > max_eval || iter_time > max_time
    status     = :unknown
    start_time = time()
    optimal    = normGx ≤ ϵₐ + ϵᵣ*normGx₀ || normFx ≤ ϵₐ + ϵᵣ*normFx₀

    #################### Tracé des graphes ###################
    disp_grad_obj && (objectif = [normFx])
    disp_grad_obj && (gradient = [normGx])

    verbose && @info log_header(
        [:iter, :nf, :obj, :grad, :ρ, :status, :nd, :λ, :δ],
        [Int, Int, Float64, Float64, Float64, String, String, Float64, Int],
        hdr_override=Dict(
        :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
        )

    while !(optimal || tired)
        ########## Calcul d (facto QR) ##########
        argmin_q2!(Arows, Acols, Avals, b, Fx, Jvals, sqrt_DλI, d, λ, D, m, n, nnzj, δ, is_λD)
        xᵖ     .= x .+ d
        residual!(nlp, xᵖ,Fxᵖ)
        fxᵖ  = norm(Fxᵖ)^2 / 2
        
        ##### sélection du modèle q adéquat #####
        JxdFx .= Jx * d + Fx
        dDd    = d'*D*d
        if alternative_model
            qxᵖ  = (norm(JxdFx)^2 + δ * dDd) / 2
            qᵃxᵖ = (norm(JxdFx)^2 + (1-δ) * dDd) / 2
            if abs(qxᵖ - fxᵖ) > γ₁ * abs(qᵃxᵖ - fxᵖ) 
                argmin_q2!(Arows, Acols, Avals, b, Fx, Jvals, sqrt_DλI, d, λ, D, m, n, nnzj, 1-δ, is_λD)
                xᵃ .= x .+ d
                residual!(nlp, xᵃ, Fxᵃ)
                fxᵃ = norm(Fxᵃ)^2 / 2
                if fxᵃ < fxᵖ
                    δ = 1-δ
                    xᵖ  .= xᵃ
                    Fxᵖ .= Fxᵃ
                    fxᵖ  = fxᵃ
                    qxᵖ  = qᵃxᵖ
                end
            end
        else
            qxᵖ  = (norm(JxdFx)^2 + dDd) / 2
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
            jac_coord_residual!(nlp, x, Jvals)
            Jx   .= maj_J(Jx, Jrows, Jcols, Jvals)     # Jx    = jac_residual(nlp, x)
            mul!(Gx,Jx',Fx)
            normFx = norm(Fx)
            normGx = norm(Gx)
            fx     = normFx^2 / 2

            ##### Maj yk₋₁ pour calcul de D #####
            if approxD_quasi_nul_lin
                for i = 1:lastindex(Fx)
                    quasi_nul = is_quasi_nul(Fx[i], Fx₋₁[i], τ₁, τ₂)
                    quasi_lin = is_quasi_lin(Fx[i], Fx₋₁[i], Jx₋₁[i,:], d, τ₃, τ₄)
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
            D = fctD(D, sk₋₁, yk₋₁, n, ϵₜ)
            
            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        disp_grad_obj && (normFx !=0) && push!(objectif, normFx)
        disp_grad_obj && (normGx !=0) && push!(gradient, normGx)
        norm_d_str = @sprintf("%.17f", norm(d))
        verbose && @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, ρ, status, norm_d_str, λ, δ])

        iter_time    = time() - start_time
        iter        += 1

        many_evals   = neval_residual(nlp) > max_eval
        iter_limit   = iter > max_iter
        tired        = many_evals || iter_time > max_time || iter_limit
        optimal      = normGx ≤ ϵₐ + ϵᵣ*normGx₀ || normFx ≤ ϵₐ + ϵᵣ*normFx₀
        
    end
    qrm_finalize()

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


LM_test                            = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_tst(nlp; disp_grad_obj = bool_grad_obj, verbose = bool_verbose)
LM_SPG                             = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = SPG   , disp_grad_obj = bool_grad_obj, verbose = bool_verbose)
LM_Zhu                             = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = Zhu   , disp_grad_obj = bool_grad_obj, verbose = bool_verbose)
LM_Andrei                          = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei, disp_grad_obj = bool_grad_obj, verbose = bool_verbose)
LM_Andrei_λD                       = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei, disp_grad_obj = bool_grad_obj, verbose = bool_verbose, is_λD = true)
LM_SPG_alt                         = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = SPG   , disp_grad_obj = bool_grad_obj, verbose = bool_verbose, alternative_model = true)
LM_Zhu_alt                         = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = Zhu   , disp_grad_obj = bool_grad_obj, verbose = bool_verbose, alternative_model = true)
LM_Andrei_alt                      = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei, disp_grad_obj = bool_grad_obj, verbose = bool_verbose, alternative_model = true)
LM_Andrei_alt_λD                   = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei, disp_grad_obj = bool_grad_obj, verbose = bool_verbose, alternative_model = true, is_λD = true)
LM_SPG_quasi_nul_lin               = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = SPG   , disp_grad_obj = bool_grad_obj, verbose = bool_verbose, alternative_model = true, approxD_quasi_nul_lin = true)
LM_Zhu_quasi_nul_lin               = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = Zhu   , disp_grad_obj = bool_grad_obj, verbose = bool_verbose, alternative_model = true, approxD_quasi_nul_lin = true)
LM_Andrei_quasi_nul_lin            = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei, disp_grad_obj = bool_grad_obj, verbose = bool_verbose, alternative_model = true, approxD_quasi_nul_lin = true)
LM_Andrei_quasi_nul_lin_λD         = (nlp ; bool_grad_obj=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei, disp_grad_obj = bool_grad_obj, verbose = bool_verbose, alternative_model = true, approxD_quasi_nul_lin = true, is_λD = true)