function argmin_q!(spmat, spfct, Avals, b, d, λ, D, n, nnzj, δ, is_λD)
    for i = 1:n
        Avals[nnzj + i] = is_λD ? sqrt(δ * λ * D[i]) : sqrt(δ * D[i] + λ)
    end
    qrm_factorize!(spmat, spfct)
    z = qrm_apply(spfct, b, transp='t')
    d .= qrm_solve(spfct, z, transp='n')
end

function SPG!(D, s, y, n, ϵₜ)
    sty = s' * y
    ss  = s' * s
    non_nul = sty > ϵₜ ? true : false
    if non_nul
        σ  = sty /ss
        for i = 1:n
            D[i]  = σ
        end
    end
end

function Zhu!(D, s, y, n, ϵₜ)
    tr   = sum(si^4 for si ∈ s)
    sy   = s' * y
    sDs  = sum(s[i]^2 * D[i] for i = 1 : n)
    frac  = (sy - sDs) / tr
    for i = 1:n
        Di = D[i] + frac * s[i]^2
        if Di > ϵₜ
            D[i] = Di
        else
            D[i] = 1
        end
    end
end 

function Andrei!(D, s, y, n, ϵₜ)
    tr   = sum(si^4 for si ∈ s)
    sy   = s' * y 
    ss   = s' * s
    sDs  = sum(s[i]^2 * D[i] for i = 1 : n)
    frac = (sy + ss - sDs)/tr

    for i = 1:n
        Di = D[i] + frac * s[i]^2 - 1
        if Di > ϵₜ
            D[i] = Di
        else
            D[i] = 1
        end
    end
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
    save_df           :: Bool = false,
    verbose           :: Bool = false,
    max_eval          :: Int = 1000, 
    max_time          :: AbstractFloat = 60.,
    max_iter          :: Int = typemax(Int64)
    )

    ################ On évalue F(x₀) et J(x₀) ################
    m, n, nnzj = nlp.nls_meta.nequ, nlp.meta.nvar, nlp.nls_meta.nnzj
    λ₀ = 1e-6
    λ = λ₀

    x   = copy(x0)
    xᵖ  = similar(x)
    d   = similar(x)
    Fx  = residual(nlp, x)
    Fxᵖ = similar(Fx)
    Jxd₊Fx = similar(Fx)

    Arows = Vector{Int}(undef, nnzj + n)
    Acols = Vector{Int}(undef, nnzj + n)
    Avals = Vector{Float64}(undef, nnzj + n)
    Arows[nnzj+1:end]   .= [k for k = m+1:m+n]
    Acols[nnzj+1:end]   .= [k for k = 1:n]
    Jrows   = view(Arows, 1:nnzj)
    Jcols   = view(Acols, 1:nnzj)
    Jvals   = view(Avals, 1:nnzj)

    jac_structure_residual!(nlp, Jrows, Jcols)
    jac_coord_residual!(nlp, x, Jvals)
    Jx = SparseMatrixCOO(m, n, Arows[1:nnzj], Acols[1:nnzj], Avals[1:nnzj])

    Gx    = Jx' * Fx

    normFx₀ = norm(Fx)
    normGx₀ = norm(Gx)
    normGx  = normGx₀
    normFx  = normFx₀
    fx = normFx^2 / 2

    qrm_init()
    spmat = qrm_spmat_init(m+n, n, Arows, Acols, Avals)
    spfct = qrm_analyse(spmat)
    b     = zeros(Float64, m+n)

    iter = 0  
    iter_time = 0
    tired   = neval_residual(nlp) > max_eval || iter_time > max_time
    status  = :unknown
    start_time = time()
    optimal    = normGx ≤ ϵₐ + ϵᵣ*normGx₀ || normFx ≤ ϵₐ + ϵᵣ*normFx₀

    save_df && (df = DataFrame(:iter => Int[], :nf => Int[], :F => Float64[], :G => Float64[], :ρ => Float64[],
               :status => :String, :nd => Float64[]))
    save_df && push!(df, Any[iter, neval_residual(nlp), normFx, normGx, 1, status, 0])

    verbose && @info log_header(
        [:iter, :nf, :obj, :grad, :ρ, :status, :nd],
        [Int, Int, Float64, Float64, Float64, String, Float64],
        hdr_override=Dict(
        :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖")
        )

    while !(optimal || tired)
        ########## Calcul d (facto QR) ##########
        b[1:m] .= Fx
        b     .*= -1
        Avals[nnzj + 1:nnzj+n] .= sqrt(λ)
        qrm_factorize!(spmat, spfct)
        z = qrm_apply(spfct, b, transp='t')
        d .= qrm_solve(spfct, z, transp='n')

        xᵖ     .= x .+ d
        residual!(nlp, xᵖ,Fxᵖ)
        fxᵖ  = norm(Fxᵖ)^2 / 2

        mul!(Jxd₊Fx, Jx, d) 
        Jxd₊Fx .+= Fx
        qxᵖ  = (norm(Jxd₊Fx)^2) / 2


        ρ = (fx - fxᵖ) / (fx - qxᵖ)

        if ρ < η₁ #|| qxᵖ > fx
            λ = max(λ₀, σ₁ * λ)
            status = :increase_λ
        else
            ############ Mise à jour ############
            x    .= xᵖ
            Fx   .= Fxᵖ
            jac_coord_residual!(nlp, x, Jvals)
            Jx.vals .= Jvals
            mul!(Gx,Jx',Fx)
            normFx = norm(Fx)
            normGx = norm(Gx)
            fx      = normFx^2 / 2
            
            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        verbose && @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, ρ, status, norm(d)])
        save_df && push!(df, Any[iter, neval_residual(nlp), normFx, normGx, ρ, status, norm(d)])

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

    if save_df
        return GenericExecutionStats(nlp; 
            status, 
            solution = x,
            objective = normFx^2 / 2,
            dual_feas = normGx,
            iter = iter, 
            elapsed_time = iter_time), df
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
    fctD              :: Function =  Andrei!,
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
    save_df                :: Bool = false,
    is_λD                  :: Bool = false,
    verbose                :: Bool = false,
    max_eval          :: Int = 1000, 
    max_time          :: AbstractFloat = 60.,
    max_iter          :: Int = typemax(Int64)
    )
    ################ On évalue F(x₀) et J(x₀) ################
    m, n, nnzj = nlp.nls_meta.nequ, nlp.meta.nvar, nlp.nls_meta.nnzj

    x    = copy(x0)
    xᵖ   = similar(x)
    x₋₁  = similar(x)
    d    = similar(x)
    Fx   = residual(nlp, x)
    Fxᵖ  = similar(Fx)
    
    Arows        = Vector{Int}(undef, nnzj + n)
    Acols        = Vector{Int}(undef, nnzj + n)
    Avals        = Vector{Float64}(undef, nnzj + n)
    Arows[nnzj+1:end]   .= [k for k = m+1:m+n]
    Acols[nnzj+1:end]   .= [k for k = 1:n]
    Avals[nnzj + 1:end] .= 1
    Jrows   = view(Arows, 1:nnzj)
    Jcols   = view(Acols, 1:nnzj)
    Jvals   = view(Avals, 1:nnzj)
    D       = ones(n)

    jac_structure_residual!(nlp, Jrows, Jcols)
    jac_coord_residual!(nlp, x, Jvals)
    Jx = SparseMatrixCOO(m, n, Arows[1:nnzj], Acols[1:nnzj], Avals[1:nnzj])
    
    Gx    = Jx' * Fx
    JᵀF   = similar(Gx)
    ############ ajout alternative_model ############
    Jxd₊Fx = similar(Fx)
    dDd   = 0
    alternative_model ? xᵃ = similar(x) : nothing
    alternative_model ? Fxᵃ = similar(Fx) : nothing
    δ = 1
    ############## ajout quasi_lin_nul ##############
    approxD_quasi_nul_lin ? r = similar(Fx) : nothing
    approxD_quasi_nul_lin ? Fx₋₁ = similar(Fx) : nothing
    approxD_quasi_nul_lin ? Jx₋₁ = similar(Jx) : nothing

    normFx₀ = norm(Fx)
    normGx₀ = norm(Gx)
    normGx  = normGx₀
    normFx  = normFx₀

    fx  = normFx^2 / 2
   
    λ₀ = 1e-6   
    λ = is_λD ? 1 : λ₀

    ############## pré-allocations ##################
    yk₋₁ = zeros(n)
    sk₋₁ = zeros(n)
    qrm_init()
    spmat = qrm_spmat_init(m+n, n, Arows, Acols, Avals)
    spfct = qrm_analyse(spmat)
    b     = zeros(Float64, m+n)

    iter = 0 
    iter_time  = 0
    tired      = neval_residual(nlp) > max_eval || iter_time > max_time
    status     = :unknown
    start_time = time()
    optimal    = normGx ≤ ϵₐ + ϵᵣ*normGx₀ || normFx ≤ ϵₐ + ϵᵣ*normFx₀

    ################## Gestion de l'affichage #################
    save_df && (df = DataFrame(:iter => Int[], :nf => Int[], :F => Float64[], :G => Float64[], :ρ => Float64[],
               :status => :String, :nd => Float64[], :λ => Float64[], :δ => Int[]))
    # save_df && rename!(df, :iter => "itérations", :nf => "évaluations", :F => "‖F(x)‖", :G => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
    save_df && push!(df, Any[iter, neval_residual(nlp), normFx, normGx, 1, status, 0, λ, δ])

    verbose && @info log_header(
        [:iter, :nf, :obj, :grad, :ρ, :status, :nd, :λ, :δ],
        [Int, Int, Float64, Float64, Float64, String, Float64, Float64, Int],
        hdr_override=Dict(
        :nf => "#F", :obj => "‖F(x)‖", :grad => "‖J'.F‖", :ρ => "ρ", :nd => "‖d‖", :λ => "λ", :δ => "δ")
        )
    while !(optimal || tired)
        ########## Calcul d (facto QR) ##########
        b[1:m] .= Fx
        b     .*= -1
        argmin_q!(spmat, spfct, Avals, b, d, λ, D, n, nnzj, δ, is_λD)
        xᵖ     .= x .+ d
        residual!(nlp, xᵖ,Fxᵖ)
        fxᵖ  = norm(Fxᵖ)^2 / 2
        
        ##### sélection du modèle q adéquat #####
        mul!(Jxd₊Fx, Jx, d)
        Jxd₊Fx .+= Fx
        dDd     = sum((d[i]^2) * D[i] for i = 1 : n)
        if alternative_model
            qxᵖ  = (norm(Jxd₊Fx)^2 + δ * dDd) / 2
            qᵃxᵖ = (norm(Jxd₊Fx)^2 + (1-δ) * dDd) / 2
            if abs(qxᵖ - fxᵖ) > γ₁ * abs(qᵃxᵖ - fxᵖ)
                argmin_q!(spmat, spfct, Avals, b, d, λ, D, n, nnzj, 1-δ, is_λD)
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
            qxᵖ  = (norm(Jxd₊Fx)^2 + dDd) / 2
        end
        ρ = (fx - fxᵖ) / (fx - qxᵖ)

        if ρ < η₁
            λ = max(λ₀, σ₁ * λ)
            status = :increase_λ
        else
            #### Stockage anciennes valeurs #####
            x₋₁  .= x
            approxD_quasi_nul_lin ? Jx₋₁ .= Jx : nothing
            approxD_quasi_nul_lin ? Fx₋₁ .= Fx : nothing

            ############ Mise à jour ############

            x    .= xᵖ
            Fx   .= Fxᵖ
            mul!(JᵀF, Jx',Fx)
            jac_coord_residual!(nlp, x, Jvals)
            Jx.vals .= Jvals
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
                yk₋₁ .-= JᵀF
            end
            sk₋₁ .= x .- x₋₁
            fctD(D, sk₋₁, yk₋₁, n, ϵₜ)
            
            status = :success    
            if ρ ≥ η₂
                λ = σ₂ * λ
            end
        end

        verbose && @info log_row(Any[iter, neval_residual(nlp), normFx, normGx, ρ, status, norm(d), λ, δ])
        save_df && push!(df, Any[iter, neval_residual(nlp), normFx, normGx, ρ, status, norm(d), λ, δ])

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

    if save_df
        return GenericExecutionStats(nlp; 
            status, 
            solution = x,
            objective = normFx^2 / 2,
            dual_feas = normGx,
            iter = iter, 
            elapsed_time = iter_time), df
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


LM_GN                              = (nlp ; bool_df=false, bool_verbose = false) -> LM_tst(nlp; save_df = bool_df, verbose = bool_verbose)
LM_SPG                             = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = SPG!   , save_df = bool_df, verbose = bool_verbose)
LM_Zhu                             = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Zhu!   , save_df = bool_df, verbose = bool_verbose)
LM_Andrei                          = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei!, save_df = bool_df, verbose = bool_verbose)
LM_SPG_λD                          = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = SPG!   , save_df = bool_df, verbose = bool_verbose, is_λD = true)
LM_Zhu_λD                          = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Zhu!   , save_df = bool_df, verbose = bool_verbose, is_λD = true)
LM_Andrei_λD                       = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei!, save_df = bool_df, verbose = bool_verbose, is_λD = true)
LM_SPG_alt                         = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = SPG!   , save_df = bool_df, verbose = bool_verbose, alternative_model = true)
LM_Zhu_alt                         = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Zhu!   , save_df = bool_df, verbose = bool_verbose, alternative_model = true)
LM_Andrei_alt                      = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei!, save_df = bool_df, verbose = bool_verbose, alternative_model = true)
LM_SPG_alt_λD                      = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = SPG!   , save_df = bool_df, verbose = bool_verbose, alternative_model = true, is_λD = true)
LM_Zhu_alt_λD                      = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Zhu!   , save_df = bool_df, verbose = bool_verbose, alternative_model = true, is_λD = true)
LM_Andrei_alt_λD                   = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei!, save_df = bool_df, verbose = bool_verbose, alternative_model = true, is_λD = true)
LM_SPG_quasi_nul_lin               = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = SPG!   , save_df = bool_df, verbose = bool_verbose, alternative_model = true, approxD_quasi_nul_lin = true)
LM_Zhu_quasi_nul_lin               = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Zhu!   , save_df = bool_df, verbose = bool_verbose, alternative_model = true, approxD_quasi_nul_lin = true)
LM_Andrei_quasi_nul_lin            = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei!, save_df = bool_df, verbose = bool_verbose, alternative_model = true, approxD_quasi_nul_lin = true)
LM_Andrei_quasi_nul_lin_λD         = (nlp ; bool_df=false, bool_verbose = false) -> LM_D(nlp; fctD = Andrei!, save_df = bool_df, verbose = bool_verbose, alternative_model = true, approxD_quasi_nul_lin = true, is_λD = true)