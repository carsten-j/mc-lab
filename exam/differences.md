# Differences

## Rejection sampling

minor:
M always larger than 1.
mention squeeze rejection from exercises

## Importance sampling

I would mentioned that distribution of weight can have severe impact on estimate
and some rules of thumb for how to choose proposal distribution

minor:
q(x) > 0 whenever π(x)ϕ(x) = 0

## Gibbs

nothing to note

## Metropolis-Hastings

I would add a bit more about the acceptance rate and the intuition for why is neccesary 
and looks as it does

minor:

MALA: Langevin-based proposal. I would say gradient-based proposal

## Pseudo-marginal MCMC

I would say a bit more about the random variable that is added in the extended space

and under practical consideration I would add
that it is important to store π^hat

## Slice sampling

maybe add drawning showing how it works.

maybe mention that sampling for S = {x : π(x) ≥ y }  is difficult and one way
to do it is stepping out

maybe mention that Elliptical slice sampling work perfectly
with regression for GP as prior is gaussian

## MALA

nothing to note

## Gelman

nothing to note

## Parallel tempering

here i would definitely mention why swapping is a good idea and the
intuition for swapping
and why acceptance rate for swapping looks as it does

maybe mention that it works for unnormalized distribtutions

## Unbiased MCMC

Here i would add the sketch proof for the Glynn-Rhee estimator

## HMC

maybe add what HMC adds over MALA

mention why flipping momentum does not matter since K(p) is symmetric in
vanilla HMC

maybe say that for L=1 HMC vanilla is equvivalent to MALA

maybe mention why Symplecticity of leapfrog is important
