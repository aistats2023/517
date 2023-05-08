using LinearAlgebra, JuMP, Hypatia, SparseArrays
import LambertW: lambertw
import Roots: find_zero

function partial_trace_A(X, dims)
    n_A = dims[1]
    n_B = prod(dims[2:end])
    return [tr(X[n_B*(0:n_A-1) .+ i, n_B*(0:n_A-1) .+ j]) for i=1:n_B, j=1:n_B]
end

function D(X, Y)
    return real(tr(X*(log(X) - log(Y)) + Y - X))
end

function D_B(X, Y, dims)
    X_B = partial_trace_A(X, dims)
    Y_B = partial_trace_A(Y, dims)
    return real(tr(X_B*(log(X_B) - log(Y_B)) + Y_B - X_B))
end

function lambertw_mat(X::AbstractMatrix)
    λ, U = eigen(X)
    return U * diagm(lambertw.(λ)) * U'
end
function lambertw_mat_normalized(X::AbstractMatrix, target)
    λ, U = eigen(X)
    f(z) = sum(lambertw.(z * λ)) - target
    u = (target / length(λ)) * exp(target / length(λ))
    bracket = (u / maximum(λ), u / minimum(λ))
    z = find_zero(f, bracket)
    return U * diagm(lambertw.(z * λ)) * U'
end

abstract type Problem end  
abstract type ProblemNormalized <: Problem end  
struct ProblemRelentNormalized <: ProblemNormalized
    n_A::Int64
    n_B::Int64
    H::Hermitian
    X0::Hermitian
    logX0::Hermitian
    γ::Float64
end
ProblemRelentNormalized(n_A, n_B, H, X0, γ) = ProblemRelentNormalized(n_A, n_B, H, X0, log(X0), γ)
struct ProblemFrobNormalized <: ProblemNormalized
    n_A::Int64
    n_B::Int64
    H::Hermitian
    X0::Hermitian
    γ::Float64
end
function F(X, p::ProblemRelentNormalized)
    @assert isapprox(tr(X), 1)
    X_B = partial_trace_A(X, [p.n_A, p.n_B])
    return real(tr(X'*p.H) + tr(X*log(X)) - tr(X_B*log(X_B))) + p.γ * D(X, p.X0)
end
function F(X, p::ProblemFrobNormalized)
    @assert isapprox(tr(X), 1)
    X_B = partial_trace_A(X, [p.n_A, p.n_B])
    return real(tr(X'*p.H) + tr(X*log(X)) - tr(X_B*log(X_B))) + p.γ * sum(abs.(X-p.X0).^2) / 2
end

traceless(X) = X - tr(X) / size(X, 1) * I

function gradF(X, p::ProblemRelentNormalized)
    @assert isapprox(tr(X), 1)
    X_B = partial_trace_A(X, [p.n_A, p.n_B])
    grad = p.H + (1+p.γ)*log(X) - kron(Matrix(I, p.n_A, p.n_A), log(X_B)) - p.γ * log(p.X0)
    return Hermitian(grad)
end
function gradF(X, p::ProblemFrobNormalized)
    @assert isapprox(tr(X), 1)
    X_B = partial_trace_A(X, [p.n_A, p.n_B])
    grad = p.H + log(X) - kron(Matrix(I, p.n_A, p.n_A), log(X_B)) + p.γ * (X - p.X0)
    return Hermitian(grad)
end


function solve_hypatia(p::ProblemNormalized; verbose=false) # real problems only
    model = Model(()->Hypatia.Optimizer(verbose=verbose))
    @variable(model, X[1:(p.n_A * p.n_B),1:(p.n_A * p.n_B)], Symmetric)
    X_B = partial_trace_A(X, [p.n_A, p.n_B])
    reduced = kron(Matrix(I, p.n_A, p.n_A), X_B)
    @variable(model, neg_cond_ent)
    full_vec = Vector{AffExpr}(undef, binomial((p.n_A * p.n_B + 1), 2))
    red_vec = Vector{AffExpr}(undef, binomial((p.n_A * p.n_B + 1), 2))
    Hypatia.Cones.smat_to_svec!(full_vec, 1.0 * X, sqrt(2))
    Hypatia.Cones.smat_to_svec!(red_vec, reduced, sqrt(2))
    cone_elem = [neg_cond_ent; red_vec; full_vec]
    @constraint(model, cone_elem in Hypatia.EpiTrRelEntropyTriCone{Float64}(length(cone_elem)))
    obj = tr(X*p.H') + neg_cond_ent
    if p.γ > 0
        if p isa ProblemFrobNormalized
            obj += 0.5 * p.γ * sum((X - p.X0).^2)
        else
            @variable(model, div_from_X0)
            X0_vec = Vector{Float64}(undef, binomial((p.n_A * p.n_B + 1), 2))
            Hypatia.Cones.smat_to_svec!(X0_vec, p.X0, sqrt(2))
            cone_elem = [div_from_X0; X0_vec; full_vec]
            @constraint(model, cone_elem in Hypatia.EpiTrRelEntropyTriCone{Float64}(length(cone_elem)))
            obj += p.γ *(div_from_X0 + tr(p.X0 - X))
        end
    end
    @constraint(model, tr(X) == 1)
    @objective(model, Min, obj)
    optimize!(model)
    println(termination_status(model))
    return value.(X)
end


function sym_to_tri(X)
    return [X[i, j] for j=1:size(X,2) for i=1:j]
end
function tri_to_sym(v)
    n = Int(floor(sqrt(length(v)*2)))
    X = zeros(n,n)
    idx = 1
    for j=1:n
        for i=1:j
            X[i,j] = v[idx]
            X[j,i] = v[idx]'
            idx += 1
        end
    end
    return X
end

function update(X, problem::ProblemRelentNormalized)
    logX_B = log(partial_trace_A(X, [problem.n_A, problem.n_B]))
    Z = exp((kron(Matrix(I,problem.n_A,problem.n_A), logX_B) + problem.γ*problem.logX0 - problem.H) / (1+problem.γ))
    return Hermitian(Z / tr(Z))
end
function update(X, problem::ProblemFrobNormalized)
    logX_B = log(partial_trace_A(X, [problem.n_A, problem.n_B]))
    Z = exp(kron(Matrix(I,problem.n_A,problem.n_A), logX_B) + problem.γ*problem.X0 - problem.H)
    return Hermitian(lambertw_mat_normalized(Hermitian(problem.γ*Z), problem.γ) / problem.γ)
end