# mDYPL
This repo contains all materials to the paper: *Maximum Diaconis & Ylvisaker penalized likelihood for high dimensional logistic regression: p/n → κ, arbitrary signal, arbitrary covariate covariance*.

- **Scripts**: All scripts to recreate the numerical results, simulation experiments and plots in the paper. 
	- `adaptive_alpha_surface_plots.jl`: Generate Figure 5 of the main text 
	- `adaptive_shrinkage_simul.jl`: Generate Table 1 of the main text 
	- `AMP_DY_unbiased.jl`: A julia mini-module to find shrinkage parameter α for state evolution parameters (1,b,σ)	
	- `AMP_DY.jl`: A julia mini-module to compute the mDYPL estimator and to numerically find the state evolution parameters (μ,b,σ)
	- `AMP_DY_helpers.jl`: Helper functions for `AMP_DY_unbiased.jl`, `AMP_DY.jl`, `AMP_Ridge.jl`
	- `beta_norm_simul.jl`: Generate data for Figure 3 of the main text and Figure 1 of the supplementary material document 
 	- `betady_scaled_plots.jl`: Recreate Figures 1 & 2 of the main text
	- `llr_simul.jl`: Recreate Figure 4 of the main text 
	- `logistic_loglikl.jl`: Logistic loglikl and variants thereof for `AMP_DY.jl`, `AMP_Ridge.jl`
	- `min_sigma_mu_plot.jl`: Recreate Figure 6b of the main text and Figure 2 of the supplementary material document 
	- `ridge.jl`: Recreate Figure 7 of the main text and Figure 3 of the supplementary material document 
	- `unbiased_alpha_plot.jl`: Recreate Figure 6a of the main text
- **Results**: Data from simulation experiments and numerical studies.
- **Figures**: All figures froom the paper and the supplementary material document, from running scripts in `Scripts` 
- **mDYPL_supplementary.pdf**: Document with proofs of the results in the main paper, along with additional numerical results. 
