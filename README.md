# PP-Seq

This repo implements the point process model of neural sequences (PP-Seq) described in:

> **[Alex H. Williams](alexhwilliams.info) :coffee:, Anthony Degleris, Yixin Wang, [Scott W. Linderman](https://web.stanford.edu/~swl1/).**  Point process models for sequence detection in high-dimensional neural spike trains. *NeurIPS 2020*, Vancouver, CA.

This model aims to identify sequential firing patterns in neural spike trains in an unsupervised manner.
For example, consider the spike train below<sup>(1)</sup>:

![image](https://user-images.githubusercontent.com/636625/94763493-36bc0400-035f-11eb-95b7-40f583ed599b.png)

By eye, we see no obvious structure in these data. However, by re-ordering the neurons according to PP-Seq's inferred sequences, we obtain:

![image](https://user-images.githubusercontent.com/636625/94767663-ee521580-0361-11eb-8729-de9eee1ab7d9.png)

Further, the model provides (probabilistic) assignment labels to each spike. In this case, we fit a model with two types of sequences. Below we use the model to color each spike as red (sequence 1), blue (sequence 2), or black (non-sequence background spike):

![image](https://user-images.githubusercontent.com/636625/94767637-db3f4580-0361-11eb-89dd-28fd8a25c468.png)

The model is fully probabilistic and Bayesian, so there are many other nice summary statistics and plots that we can make. See our paper for full details.

**Footnote.** (1) These data are deconvolved spikes from a calcium imaging recording from zebra finch HVC. These data were published in [Mackevicius*, Bahle*, et al. (2019)](https://elifesciences.org/articles/38471) and are freely available online at https://github.com/FeeLab/seqNMF.
