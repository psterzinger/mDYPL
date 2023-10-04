# mDYPL
This repo contains all materials to the paper: *Maximum Diaconis & Ylvisaker penalized likelihood for high dimensional logistic regression: p/n → κ, arbitrary signal, arbitrary covariate covariance*.

- **Scripts**: All scripts to recreate the numerical results, simulation experiments and plots in the paper. 
	- `AMP_DY.jl`: A julia module to compute the mDYPL estimator and to numerically find the state evolution parameters (μ,b,σ).
 	- `betady_scaled_plots.jl` Recreate Figures 1 & 2 of the main text.  
- **Results**: Data from simulation experiments and numerical studies.
- **mDYPL_supplementary.pdf**: Document with proofs of the results in the main paper, along with additional numerical results. 
