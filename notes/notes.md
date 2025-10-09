Probability 
https://bjlkeng.io/posts/probability-the-logic-of-science/
Om E.T. Jaynes bog



Why does accept-reject work, se bog ved seng derhjemme
https://academic.uprm.edu/wrolke/esma5015/fund.html
https://www.stats.ox.ac.uk/~winkel/ASim11.pdf


Rejection sampling dimensionality curse mackay 29.3
Billingslei reference i bog ved seng



Importance sampling - Diagnostics 
ESS
https://www.tuananhle.co.uk/notes/ess.html
https://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html
Art owens noter

Opdater kode til ESS og gns af vægte og evt Art Owen som bruger integrand 
PSIS
Og relation til Arviz

Se også Mackey bog for IS 


## Markov chains

* https://mpaldridge.github.io/math2750/S10-stationary-distributions.html
  
## Thinning

* https://besjournals.onlinelibrary.wiley.com/doi/pdf/10.1111/j.2041-210X.2011.00131.x#:~:text=3.,MCMC%20appeared%20in%20ecological%20publications.
* https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size.section
* https://discourse.julialang.org/t/thinning-mcmc-posteriors-to-reduce-autocorrelation/85254/3

## Pseudo-marginal MCMC

* https://lips.cs.princeton.edu/pseudo-marginal-mcmc/
* https://darrenjw.wordpress.com/2010/09/20/the-pseudo-marginal-approach-to-exact-approximate-mcmc-algorithms/

## Rejection sampling

https://cswr.nrhstat.org/reject-samp#adaptive
Accept-reject - piecewise log-concave densities

https://amyanchen.github.io/files/Adaptive_Rejection_Sampling
Adaptive Squeezed Rejection Sampling

https://longhaisk.github.io/software/ars/ars.html
Adaptive rejection sampling - ARS

Importance sampling - confidence band

https://joss.theoj.org/papers/10.21105/joss.06906
Alternative sampling strategy

https://emcee.readthedocs.io/en/stable/
Jonathan Goodman

## Gibbs sampling

* https://arxiv.org/pdf/2403.18054
* https://xuwd11.github.io/am207/wiki/introgibbs.html
* https://xuwd11.github.io/am207/wiki/tetchygibbs.htm
* https://xuwd11.github.io/am207/wiki/gibbsfromMH.html

## Metropolis-Hastings

https://ermongroup.github.io/cs323-notes/probabilistic/mh/
Metropolis-Hastings og noget om hvorfor den konvergeres til den ønskede fordeling
og Gibbs som special tilfælde
https://ermongroup.github.io/cs323-notes/probabilistic/gibbs/

https://blog.djnavarro.net/posts/2023-04-12_metropolis-hastings/
Metropolis-Hastings - effekt af lag, burn-in, etc
Jeg har allerede dette som en notebook

https://prappleizer.github.io/Tutorials/MetropolisHastings/MetropolisHastings_Tutorial.html
Som anbefaler
https://arxiv.org/pdf/1710.06068#page3

https://bjlkeng.io/posts/markov-chain-monte-carlo-mcmc-and-the-metropolis-hastings-algorithm/

## HMC

* https://bjlkeng.io/posts/hamiltonian-monte-carlo/ - Og se implementation her
* https://github.com/bjlkeng/sandbox/blob/master/hmc/hmc.ipynb
* https://blogs.rstudio.com/ai/posts/2019-10-03-intro-to-hmc/
* https://github.com/ColCarroll/minimc?tab=readme-ov-file
* https://www.youtube.com/watch?v=a-wydhEuAm0&list=PL4SE6Ciqnz113-Fx8LJexTe7hlV1t4CrY&index=4

## Langevin Monte Carlo

* https://bjlkeng.io/posts/bayesian-learning-via-stochastic-gradient-langevin-dynamics-and-bayes-by-backprop/#langevin-monte-carlo
* https://abdulfatir.com/blog/2020/Langevin-Monte-Carlo/
* https://danmackinlay.name/notebook/mcmc_langevin
* https://friedmanroy.github.io/blog/2022/Langevin/

## Gaussians

* https://medium.com/mti-technology/how-to-generate-gaussian-samples-347c391b7959
* https://cs229.stanford.edu/section/gaussians.pdf
* https://online.stat.psu.edu/stat505/book/export/html/636#:~:text=Every%20single%20variable%20has%20a has%20a%20univariate%20normal%20distribution.

### Various links

https://github.com/pints-team/pints/blob/d4440e38cb0608190b90f4a5f6a5426d7f98bfcc/examples/sampling/first-example.ipynb
https://friedmanroy.github.io/blog/2023/AIS/

## Software

* https://pints.readthedocs.io/en/stable/mcmc_samplers/nuts_mcmc.html
* https://code.ornl.gov/2kv/slicesampling
* https://github.com/willvousden/ptemcee/tree/main
* https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py#L123
* https://colab.research.google.com/github/lyndond/lyndond.github.io/blob/master/code/2021-02-09-elliptical-slice-sampling.ipynb#scrollTo=0OZbf2dL-T-W
