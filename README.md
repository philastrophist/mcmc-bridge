# mcmc_bridge
A way to link pymc3's models with the emcee sampler


Measure correlation using a Gaussian:

    from mcmc_bridge import EmceeTrace, export_to_emcee, get_start_point
    import pymc3 as pm
  
    values = read_data()
    nmeas, ndim = values.shape
    with pm.Model() as model:
        packed_L = pm.LKJCholeskyCov('packed_L', n=ndim, eta=1, sd_dist=pm.HalfCauchy.dist(2.5))
        L = pm.expand_packed_triangular(ndim, packed_L)
        cov = pm.Deterministic('cov', L.dot(L.T))

        std = pm.Deterministic('std', tt.sqrt(tt.diag(cov)))
        diag = tt.diag(std)
        diaginv = tt.nlinalg.matrix_inverse(diag)
        corr = tt.dot(diaginv, tt.dot(cov, diaginv))

        indices = np.triu_indices(ndim, k=1)
        pm.Deterministic('corr_coeffs', corr[indices])

        mu = pm.Normal('mu', mu=values.mean(axis=0), sd=values.std(axis=0), shape=ndim)  # centre of gaussian
        pm.MvNormal('like', mu=mu, chol=L, observed=values, shape=ndim)

        sampler = export_to_emcee(nwalker_multiple=nwalker_multiple, threads=threads, pool=pool)
        start = get_start_point(sampler)
        sampler.run_mcmc(start, steps)
        trace = EmceeTrace(sampler)  # pymc3 trace object!
