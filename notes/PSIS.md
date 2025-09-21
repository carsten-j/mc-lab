# All I need is time, a moment that is mine, while I’m in between

Posted on [June 28, 2019 10:14 PM](https://statmodeling.stat.columbia.edu/2019/06/28/all-i-need-is-time-a-moment-that-is-mine-while-im-in-between/ "10:14 pm") by [Dan Simpson](https://statmodeling.stat.columbia.edu/author/simpson/ "View all posts by Dan Simpson")

_You’re an ordinary boy and that’s the way I like it – [Magic Dirt](https://www.youtube.com/watch?v=eYqKlwSyhPw)_

Look. I’ll say something now, so it’s off my chest. I _hate_ order statisics. I loathe them. I detest them. I wish them nothing but ill and strife. They are just awful. And I’ve spent the last god only knows how long buried up to my neck in them, like Jennifer Connelly forced into the fetid pool at the end of Phenomena.

It would be reasonable to ask why I suddenly have opinions about order statistics. And the answer is weird. It’s because of Pareto Smoothing Importance Sampling (aka PSIS aka the technical layer that makes the loo package work).

The original PSIS paper was written by Aki, Andrew, and Jonah. However [there is a brand new sparkly version by Aki, Me, Andrew, Yuling, and Jonah](https://arxiv.org/abs/1507.02646) that has added a pile of theory and restructured everything (edit by Aki: link changed to the updated arXiv version). Feel free to read it. The rest of the blog post will walk you through some of the details.

**What is importance sampling?**

Just a quick reminder for those who don’t spend their life thinking about algorithms. The problem at hand is estimating the expectation $I_h=\mathbb{E}(h(\theta))$ for some function $h$ when $\theta\sim p(\theta)$.  If we could sample directly from $p(\theta)$ then the Monte Carlo estimate of the expectation would be

$\frac{1}{S}\sum_{s=1}^Sh(\theta_s)$,  where $\theta_s\stackrel{\text{iid}}{\sim}p(\theta)$.

But in a lot of real life situations we have two problems with doing this directly: firstly it is usually very hard to sample from $p(\theta)$. If there is a different distribution that we _can_ sample from, say $g$, then we can use the following modification of the Monte Carlo estimator

$$I_h^S= \frac{1}{S}\sum_{s=1}^S\frac{p(\theta_s)}{g(\theta_s)}h(\theta_s),$$

 where $\theta_s$ are iid draws from $g(\theta)$. This is called an _importance sampling_ estimator. The good news is that it always converges in probability to the true expectation. The bad news is that it is a _random variable_ and it can have infinite variance.

The second problem is that often enough we only know the density $p(\theta)$ up to a normalizing constant, so if $f(\theta)\propto p(\theta)$, then the following _self-normalized_ _importance sampler_ is useful

$$I_h^S= \frac{\sum_{s=1}^Sr(\theta_s)h(\theta_s)}{\sum_{s=1}^Sr(\theta_s)},$$

where the _importance ratios_ are defined as

$$r_s=r(\theta_s) = \frac{f(\theta_s)}{g(\theta_s)},$$

where again $\theta_s\sim g$. This will converge to the correct answer as long as $\mathbb{E}(r_s)<\infty$.  For the rest of this post I am going to completely ignore self-normalized importance samplers, but everything I’m talking about still holds for them.

**So does importance sampling actually work?**

Well god I do hope so because it is used _a lot_. But there’s a lot of stuff to unpack before you can declare something “works”. (That is a lie, of course, all kinds of people are willing to pick a single criterion and, based on that occurring, declaring that it works. And eventually that is what we will do.)

First things first, an importance sampling estimator is a sum of independent random variables. We may well be tempted to say that, by the central limit theorem, it will be asymptotically normal. And sometimes that is true, _but_ _only if_ the importance weights have finite variance. This will happen, for example, if the proposal distribution _g_ has heavier tails than the target distribution _p_.

And there is a temptation to stop there. To declare that if the importance ratios have finite variance then importance sampling works. _That. Is. A. Mistake._

Firstly, this is demonstrably untrue in moderate-to-high dimensions. It is pretty easy to construct examples where the importance ratios are bounded (and hence have finite variance) but there is no feasible number of samples that would give small variance. This is a problem as old as time: just because the central limit theorem says the error will be around $\sigma/\sqrt{S}$, that doesn’t mean that $\sigma$ won’t be an _enormous_ number.

And here’s the thing: we do not know $\sigma$ and our only way to estimate it is _to use the importance sampler_. So when the importance sampler doesn’t work well, we may not be able to get a decent estimate of the error. So even if we can guarantee that the importance ratios have finite variance (which is really hard to do in most situations), we may end up being far too optimistic about the error.

[Chatterjee and Diaconis](https://arxiv.org/pdf/1511.01437.pdf) recently took a quite different route to asking whether an importance sampler converges. They asked what the minimum sample size required to ensure, with high probability, that $|I_h^S – I_h|$ is small (with high probability). They showed that you need approximately $\exp(\mathbb{E}[r_s \log(r_s)])$ samples and this number can be _large_.  This quantity is also quite hard to compute (and they proposed another heuristic, but that’s not relevant here), but it is going to be important later.

**Modifying importance ratios**

So how do we make importance sampling more robust. A good solution is to somehow modify the importance ratios to ensure they have finite variance. Ionides proposed a method called [Truncated Importance Sampling (TIS)](http://dept.stat.lsa.umich.edu/~ionides/pubs/tech424-revised.pdf) where the importance ratios are replaced with truncated weights $w_s=\max\{r_s,\tau_S\}$, for some sequence of thresholds $\tau_S\rightarrow\infty$ as $S\rightarrow\infty$.  The resulting TIS estimator is

$$I_h^S= \frac{1}{S}\sum_{s=1}^Sw_s h(\theta_s).$$

A lot of real estate in Ionides’ paper is devoted to choosing a good sequence of truncations. There’s theory to suggest that it depends on the tail of the importance ratio distribution. But the suggested choice of truncation sequence is $\tau_S=C\sqrt{S}$, where $C$ is the normalizing constant of $f$ which is one when using ordinary rather than self-normalized importance sampling. (For the self normalized version, Appendix B suggests taking $C$ as the sample mean of the importance ratios, but the theory only works for deterministic truncations.)

This simple truncation _guarantees_ that TIS is asymptotically unbiased, has finite variance that asymptotically goes to zero, and (with some caveats) is asymptotically normal.

But, as we discussed above, none of this actually guarantees that TIS will work for a certain problem. (It does work asymptotically for a vast array of problems and does a lot better that ordinary importance sampler, but no simple truncation scheme can overcome a poorly chosen proposal distribution. And most proposal distributions in high dimensions are poorly chosen.)

**Enter Pareto-Smoothed Importance Sampling**

So a few years ago Aki and Andrew worked on an alternative to TIS that would make things even better. (They originally called it the “Very Good Importance Sampling”, but then Jonah joined the project and ruined the acronym.) The algorithm they came up with was called _[Pareto-Smoothed Importance Sampling](https://arxiv.org/abs/1507.02646v5)_ (henceforth PSIS, the link is to the three author version of the paper).

They noticed that TIS basically replaces all of the large importance ratios with a single value $\tau_S$. Consistent with both Aki and Andrew’s penchant for statistical modelling, they thought [they could do better than that](https://www.youtube.com/watch?v=XucZfgxxeps) (Yes. It’s the Anna Kendrick version. Deal.)

PSIS is based on the principle the idea that, while using the same value for each extreme importance ratio _works_, it would be even better to _model the distribution of extreme importance ratios!_ The study of distributions of extremes of independent random variables has been an extremely important (and mostly complete) part of statistical theory. This means that we _know things_.

One of the key facts of extreme value theory is that the distribution of ratios larger than some sufficiently large threshold _u_  approximately has a [generalized Pareto distribution](https://en.m.wikipedia.org/wiki/Generalized_Pareto_distribution) (gPd). Aki, Andrew, and Jonah’s idea was to fit a generalized Pareto distribution to the _M_ largest importance ratios and replace the upper weights with appropriately chosen quantiles of the fitted distribution. (Some time later, I was very annoyed they didn’t just pick a deterministic threshold, but this works better even if it makes proving things much harder.)

They learnt a few things after extensive simulations. Firstly, this almost always does better than TIS (the one example where it doesn’t is example 1 in the revised paper). Secondly, the gPd has two parameters that need to be estimated (the third parameter is an order statistic of the sample. ewwwww) And one of those parameters is _extremely_ useful!

The shape parameter (or tail parameter) of the gPd, which we call _k,_ controls how many moments the distribution has. In particular, a distribution who’s upper tail limits to a gPd with shape parameter _k_ has at most $k^{-1}$ finite moments. This means that if $k<1/2$ then an importance sampler will have finite variance.

But we do not have access to the true shape parameter. We can only estimate it from a finite sample, which gives us $\hat{k}$, or, as we constantly write, “k-hat”. The k-hat value has proven to be an extremely useful diagnostic in a wide range of situations. (I mean, sometimes it feels that every other paper I write is about k-hat. I love k-hat. If I was willing to deal with voluntary pain, I would have a k-hat tattoo. I once met a guy with a nabla tattooed on his lower back, but that’s not relevant to this story.)

Aki, Andrew, and Jonah’s extensive simulations showed something that may well have been unexpected: the value of k-hat is a good proxy for the quality of PSIS. (Also TIS, but that’s not the topic). In particular, if k-hat was bigger than around 0.7 it became massively expensive to get an accurate estimate. So we can use k-hat to work out if we can trust our PSIS estimate.

PSIS ended up as the engine driving the loo package in R, which last time I checked had around 350k downloads from the RStudio CRAN mirror. It works for high-dimensional problems and can automatically assess the quality of an importance sampler proposal for a given realization of the importance weights.

So PSIS is robust, reliable, useful, has R and Python packages, and the paper was full of detailed computational experiments that showed that it was robust, reliable, and useful even for high dimensional problems. What could possibly go wrong?

**What possibly went wrong**

Reviewers.

**It works, but where is the theory?**

I wasn’t an author so it would be a bit weird for me to do a postmortem on the reviews of someone else’s paper. But one of the big complaints was that Aki, Andrew, and Jonah had not shown that PSIS was asymptotically unbiased, had finite vanishing variance, or that it was asymptotically normal.

(Various other changes of emphasis or focus in the revised version are possibly also related to reviewer comments from the previous round, but also to just having more time.)

These things turn out to be tricky to show. So Aki, Andrew, and Jonah invited me and Yuling along for the ride.

The aim was to restructure the paper, add theory, and generally take a paper that was very good and complete and add some sparkly bullshit. So sparkly bullshit was added. Very slowly (because theory is hard and I am not good at it).

**Justifying k-hat < 0.7**

Probably my favourite addition to the paper is due to Yuling, who read the [Chatterjee and Diaconis](https://arxiv.org/pdf/1511.01437.pdf)  paper and noticed that we could use their lower bound on sample size to justify k-hat. The idea is that it is the tail of $r_s$ that breaks the importance sampler. So if we make the assumption that the entire distribution of $r_s$ is generalized Pareto with shape parameter _k_, we can actually compute the minimum sample size for a particular accuracy from ordinary importance sampling. This is not an accurate sample size calculation, but should be ok for an order-of-magnitude calculation.

The first thing we noticed is, consistent with the already existing experiments, the error in importance sampling (and TIS and PSIS) increases smoothly as _k_ passes 0.5 (in particular the finite-sample behaviour does not fall off a cliff the moment the variance isn’t finite). But the minimum sample size starts to increase _very_ rapidly as soon as _k_ got bigger than about 0.7. This is consistent with the experiments that originally motivated the 0.7 threshold and suggests (at least to me) that there may be something fundamental going on here.

We can also use this to justify the threshold on k-hat as follows. The method Aki came up with for estimating k-hat is (approximately) Bayesian, so we can interpret the k-hat at a value selected so that the data is _consistent_ with _M_ independent samples from a gPd with shape parameter k-hat. So a k-hat value that is bigger than 0.7 can be interpreted loosely as saying that the extreme importance ratios could have come from a distribution that has a tail that is too heavy for PSIS to work reliably.

This is what actually happens in high dimensions (for an example we have that has bounded ratios and hence finite variance). With a reasonable sample size, the estimator for k-hat simply cannot tell that the distribution of extreme ratios has a large but finite variance rather than an infinite variance. And this is exactly what we want to happen! I have no idea how to formalized this intuition, but nevertheless it works.

**So order statistics**

It turned out that–even though it is quite possible that other people would not have found proving unbiasedness and finite variance hard–I found it very hard. Which is quite annoying because the proof for TIS was literally 5 lines.

What was the trouble? Aki, Andrew, and Jonah’s decision to choose the threshold as the _M_th largest importance ratio. This means that the threshold is an order statistic and hence is _not independent_ of the rest of the sample. So I had to deal with that.

This meant I had to read an absolute tonne of papers about order statistics. These papers are dry and technical and were all written between about 1959 and 1995 and at some later point poorly scanned and uploaded to JSTOR. And they rarely answered the question I wanted them to. So basically I am quite annoyed with order statistics.

But the end point is that, under some conditions, PSIS is asymptotically unbiased and has finite, vanishing variance.

The conditions are a bit weird, but are usually going to be satisfied.  Why are they weird? Well…

**PSIS is TIS with an adaptive threshold and bias correction**

In order to prove asymptotic properties of PSIS, I used the following representation of the PSIS estimator

$$\frac{1}{S}\sum_{s=1}^{S}\min\left(r(\theta_s),r(\theta_{(S-M+1):S})\right)h(\theta_s)+\frac{1}{S}\sum_{m=1}^M\tilde{w}_mh(\theta_{S-M+m}),$$

where the samples $\theta_s$ have been ordered so that $r(\theta_1)\leq r(\theta_2)\leq\ldots\leq r(\theta_S)$ and the weights $\tilde{w}_m$ are deterministic (and given in the paper). They are related to the quantile function for the gPd.

The first term is just TIS with random threshold $\tau_S=r(\theta_{(S-M+1):S})$, while the second term is an approximation to the bias. So PSIS has higher variance than TIS (because of the random truncation), but lower bias (because of the second term) and this empirically usually leads to lower mean-square error than TIS.

But that random truncation is automatically adapted to the tail behaviour of the importance ratios, which is an extremely useful feature!

This representation also gives hints as to where the ugly conditions come from. Firstly, anything that is adaptive is much harder to prove things about than a non-adaptive method, and the technical conditions that we need to be able to adapt our non-adaptive proof techniques are often quite esoteric. The idea of the proof is to show that, conditional on $r(\theta_{(S-M+1):S})=U$, all of the relevant quantities go to zero (or are finite) with some explicit dependence on _U_. The proof of this is very similar to the TIS proof (and would be exactly the same if the second term wasn’t there).

Then we need to let _U_ vary and hope it doesn’t break anything. The technical conditions can be split into the ones needed to ensure $r(\theta_{(S-M+1):S})=U$ behaves itself as _S_ gets big; the ones needed to ensure that $h(\theta)$ doesn’t get too big when the importance ratios are large; and the ones that control the last term.

Going in reverse order, to ensure the last term is well behaved we need that _h_ is square-integrable with respect to the proposal _g_ in addition to the standard assumption that its square integrable with respect to the target _p_.

We need to put growth conditions on _h_ because we are only modifying the ratios, which does not help if _h_ is also enormous out in the tails. These conditions are actually very easy to satisfy for most problems I can think of, but almost certainly there’s some one out there with a _h_ that grows super-exponentially just waiting to break PSIS.

The final conditions are just annoying. They are impossible to verify in practice, but there is a 70 year long literature that coos reassuring phrases like “this almost always holds” into our ears. These conditions are strongly related to the conditions needed to estimate _k_ correctly (using something like the Hill estimator). My guess is that these conditions are not vacuous, but are relatively unimportant for finite samples, where the value of k-hat should weed out the catastrophic cases.

**What’s the headline**

With some caveats, PSIS is asymptotically unbiased; has finite, vanishing variance; and a variant of it is asymptotically normal as long as the importance ratios have more than $(1+\delta)$-finite moments. But it probably won’t be useful unless it has at least 1/0.7 = 1.43 moments.

**And now we send it back off into the world and see what happens**
