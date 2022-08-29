# Recurrent Policies for Handling Partially Observable Environments with [ReLAx](https://github.com/nslyubaykin/relax)

This repository contains an [implementation](https://github.com/nslyubaykin/rnns_for_pomdp/blob/master/lags_for_pomdp.ipynb) of PPO-GAE algorithm with lagged LSTM policy (and critic) and its comparison with 0-lag MLP PPO-GAE.

To simulate partial observability in a controlled manner a gym.Wrapper which masks observation's array elements with zeros with eps probability was created. 
In our experiments, the degree of partial observability was controlled through altering eps value.

Experiments results are shown below:

![pomdp_comparison](https://github.com/nslyubaykin/rnns_for_pomdp/blob/master/pomdp_comparison.png)

As we can see, for the fully observable case (eps=0) MLP and LSTM policies show roughly the same performance. 
For a moderate degree of partial observability (eps=0.25) LSTM policy shows slightly faster learning at the early stages.
For a considerable degree of partial observability (eps=0.5) LSTM policy shows significantly better performance comparing to MLP policy. 
However, both actors struggled to converge to fully observable case asymptotic performance.
For a staggering degree of partial observability (eps=0.75) both policies failed to learn.
