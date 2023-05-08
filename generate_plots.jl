include("utils.jl")
import Random: MersenneTwister
import PGFPlots
using BenchmarkTools


function setup(n_B, γ=1; seed=0)
    n_A = 2
    H = randn(MersenneTwister(seed), Float64, n_A*n_B, n_A*n_B) |> Hermitian
    H /= maximum(abs.(eigvals(H)))
    X0 = Matrix(I, n_A*n_B, n_A*n_B)/ (n_A*n_B) |> Hermitian
    return ProblemRelentNormalized(n_A, n_B, H, X0, γ) 
end

function run_dca(prob; max_iter=Inf, tol=1e-7)
    X = copy(prob.X0);
    grad_norms = [2*maximum(abs.(eigvals(traceless(gradF(X, prob)))))]
    fvals = [F(X, prob)]
    total = 0.
    while length(grad_norms) <= max_iter
        total += @elapsed X = update(X, prob)
        grad_norm = 2*maximum(abs.(eigvals(traceless(gradF(X, prob)))))
        grad_norm > tol || break
        push!(grad_norms, grad_norm)
        push!(fvals, F(X, prob))
    end
    return total, grad_norms, fvals
end

geom_mean(X) = X .|> log |> mean |> exp

println("Warming up...")
p = setup(4);
run_dca(p);
solve_hypatia(p);
run_dca(p);
solve_hypatia(p);
println("Done")

num_samples = 5;
dBmax = 9;
times_hypatia= [];
times_dca = [];
for dB = 1:dBmax
    println("Starting nB = ", 2^dB)
    problems = [setup(2^dB; seed=i) for i=1:num_samples]
    dB <= 5 && push!(times_hypatia, [@elapsed solve_hypatia(p) for p in problems])
    push!(times_dca, [run_dca(p)[1] for p in problems])
end
mean_hyp = geom_mean.(times_hypatia);
mean_dca = geom_mean.(times_dca);
plot_hyp = PGFPlots.Plots.Linear(1:length(mean_hyp), mean_hyp, onlyMarks=true, legendentry="IPM");
plot_dca = PGFPlots.Plots.Linear(1:length(mean_dca), mean_dca, onlyMarks=true, legendentry="DCA");
ax0 = PGFPlots.Axis(xlabel="\$\\log_2(d_B)\$", ylabel="Time \$\\mathrm{[s]}\$", ymode="log", width="7cm");
push!(ax0, plot_hyp);
push!(ax0, plot_dca);
ax0.legendPos="south east"
ax0.legendStyle = "nodes={scale=0.7, transform shape}"
PGFPlots.save("paper/prox_timings.tex", ax0, include_preamble=false)

nB = 2^9
γ = 1.
p = setup(nB, γ);
_, grad_norms, fvals = run_dca(p, max_iter=1000);
ferrs = fvals[1:end-1] .- fvals[end];
ax1 = PGFPlots.Axis(xlabel="\$k\$ (iterations)", ymode="log", ymin=1e-10, width="7cm");
push!(ax1, PGFPlots.Plots.Linear(0:length(grad_norms)-1, grad_norms, legendentry="\$2\\|\\nabla F(X^k)\\|_\\sigma\$", markSize=1))
push!(ax1, PGFPlots.Plots.Linear(0:length(ferrs)-1, ferrs, legendentry="\$F(X^k) - F_*\$", markSize=1))
push!(ax1, PGFPlots.Plots.Linear(1:length(grad_norms), [log(nB)/(1+p.γ)^(k-1) for k in 1:length(grad_norms)], legendentry="\$\\frac{\\log d_B}{(1+\\gamma)^{k-1}}\$", markSize=1))
ax1.legendStyle = "nodes={scale=0.7, transform shape}"
ax1
PGFPlots.save("paper/prox_convergence_db=$nB.tex", ax1, include_preamble=false)

###########################################

nB = 2^6
p = setup(nB, 0.);
total, grad_norms, fvals = run_dca(p, max_iter=5000);
println(total)
ferrs = fvals[1:end-1] .- fvals[end];
ax2 = PGFPlots.Axis(xlabel="\$k\$ (iterations)", ymode="log", ymin=1e-10, width="7cm");
push!(ax2, PGFPlots.Plots.Linear(1:length(grad_norms), grad_norms, legendentry="\$2\\|\\nabla F(X^k)\\|_\\sigma\$", markSize=0.2))
push!(ax2, PGFPlots.Plots.Linear(1:length(ferrs), ferrs, legendentry="\$F(X^k) - F_*\$", markSize=0.2))
push!(ax2, PGFPlots.Plots.Linear(1:length(ferrs), log(nB) ./ (1:length(ferrs)), legendentry="\$\\frac{1}{k}\\log(d_B)\$", markSize=0.2))
ax2.legendStyle = "nodes={scale=0.7, transform shape}"
ax2
PGFPlots.save("paper/dual_convergence_db=$nB.tex", ax2, include_preamble=false)

num_samples = 5;
dBmax = 6;
times_hypatia= [];
times_dca = [];
for dB = 1:dBmax
    println("Starting nB = ", 2^dB)
    problems = [setup(2^dB, 0.; seed=i) for i=1:num_samples]
    dB <= 5 && push!(times_hypatia, [@elapsed solve_hypatia(p) for p in problems])
    push!(times_dca, [run_dca(p)[1] for p in problems])
end
mean_hyp = geom_mean.(times_hypatia);
mean_dca = geom_mean.(times_dca);
plot_hyp = PGFPlots.Plots.Linear(1:length(mean_hyp), mean_hyp, onlyMarks=true, legendentry="IPM");
plot_dca = PGFPlots.Plots.Linear(1:length(mean_dca), mean_dca, onlyMarks=true, legendentry="DCA");
ax3 = PGFPlots.Axis(xlabel="\$\\log_2(d_B)\$", ylabel="Time \$\\mathrm{[s]}\$", ymode="log", width="7cm");
push!(ax3, plot_hyp);
push!(ax3, plot_dca);
ax3.legendPos="south east"
ax3.legendStyle = "nodes={scale=0.7, transform shape}"
PGFPlots.save("paper/dual_timings.tex", ax3, include_preamble=false)


###############################
function setup_frob(n_B, γ=1; seed=0)
    n_A = 2
    H = randn(MersenneTwister(seed), Float64, n_A*n_B, n_A*n_B) |> Hermitian
    H /= maximum(abs.(eigvals(H)))
    X0 = Matrix(I, n_A*n_B, n_A*n_B)/ (n_A*n_B) |> Hermitian
    return ProblemFrobNormalized(n_A, n_B, H, X0, γ) 
end

println("Warming up...")
p = setup_frob(4);
run_dca(p);
solve_hypatia(p);
run_dca(p);
solve_hypatia(p);
println("Done")


nB = 2^8
p = setup_frob(nB);
total, grad_norms, fvals = run_dca(p, max_iter=5000);
println(total)
ferrs = fvals[1:end-1] .- fvals[end];
ax4 = PGFPlots.Axis(xlabel="\$k\$ (iterations)", ymode="log", ymin=1e-10, width="7cm", height="5cm");
push!(ax4, PGFPlots.Plots.Linear(1:length(grad_norms), grad_norms, legendentry="\$2\\|\\nabla F(X^k)\\|_\\sigma\$", mark="none"))
push!(ax4, PGFPlots.Plots.Linear(1:length(ferrs), ferrs, legendentry="\$F(X^k) - F_*\$", mark="none"))
push!(ax4, PGFPlots.Plots.Linear(1:length(ferrs), log(nB) ./ (1:length(ferrs)), legendentry="\$\\frac{1}{k}\\log(d_B)\$", mark="none"))
ax4.legendStyle = "nodes={scale=0.7, transform shape}"
ax4
PGFPlots.save("paper/frob_convergence_db=$nB.tex", ax4, include_preamble=false)

num_samples = 5;
dBmax = 8;
times_hypatia= [];
times_dca = [];
for dB = 1:dBmax
    println("Starting nB = ", 2^dB)
    problems = [setup_frob(2^dB; seed=i) for i=1:num_samples]
    dB <= 5 && push!(times_hypatia, [@elapsed solve_hypatia(p) for p in problems])
    push!(times_dca, [run_dca(p)[1] for p in problems])
end
mean_hyp = geom_mean.(times_hypatia);
mean_dca = geom_mean.(times_dca);
plot_hyp = PGFPlots.Plots.Linear(1:length(mean_hyp), mean_hyp, onlyMarks=true, legendentry="IPM");
plot_dca = PGFPlots.Plots.Linear(1:length(mean_dca), mean_dca, onlyMarks=true, legendentry="DCA");
ax5 = PGFPlots.Axis(xlabel="\$\\log_2(d_B)\$", ylabel="Time \$\\mathrm{[s]}\$", ymode="log", width="7cm", height="5cm");
push!(ax5, plot_hyp);
push!(ax5, plot_dca);
ax5.legendPos="south east"
ax5.legendStyle = "nodes={scale=0.7, transform shape}"
PGFPlots.save("paper/frob_timings.tex", ax5, include_preamble=false)