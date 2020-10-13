<p align="center">
  <a href="https://sites.research.unimelb.edu.au/bmmlab">
    <img width="250px" src="./bmm_logo.png">
  </a>
</p>
</br>
<h1 align="center">
  Exploiting Distributional Temporal Difference Learning </br> To Deal With Tail Risk
</h1>


This repository contains both the environment and the agents of the leptokurtosis project.

For a detailed description of the experiment, please refer to our paper.

#### Environment (envs):
* Leptokurtosis

#### Implemented algorithms (agents):
* Tabular SARSA
* Categorical Temporal Difference Learning (a tabular version of Categorical distributional reinforcement learning)
* (Efficient) Distributional Temporal Difference Learning: 
    - Integration over reward distribution (sample average) method.
    - Maximum Likelihood Estimator (EM-MLE) method.

## Prerequisites 
see [```requirements.txt```](./requirements.txt)
* [Python 3.6](https://www.python.org/)
* [Numpy](http://www.numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [OpenAI-gym](https://github.com/openai/gym)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

</br>

## Results replication
- To replicate the simulation results, run [```stats_and_plots/run_game_100_times.py```](./outlier_project/stats_and_plots/run_game_100_times.py) (warning: takes time)
- Archived simulation results (in json format) are available in a zip file.
- The performance statistics can be calculated by [```stats_and_plot/analysis.py```](./outlier_project/stats_and_plots/analysis.py)
- To generate the data for plots in the paper, run [```stats_and_plots/figures.py```](./outlier_project/stats_and_plots/figures.py)
- To see all the essential meta-parameters used in the paper, please refer to [```utils/config.py```](./outlier_project/utils/config.py)

</br>

## To reference this repository
```
@misc{distributionalRLTailRisk,
  author = {Peter Bossaerts, Shijie Huang and Nitin Yadav},
  title = {Exploiting Distributional Temporal Difference Learning To Deal With Tail Risk},
  year = {2020}
}
```

</br>

## Key references
### Outlier and Leptokurtosis:
* [Neural Mechanisms Behind Identification of Leptokurtic Noise and Adaptive Behavioral Response](https://academic.oup.com/cercor/article/26/4/1818/2367610)
* [Supplementary Material: behavioural results and reinforcement learning](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/cercor/26/4/10.1093_cercor_bhw013/1/bhw013_Supplementary_Data.zip?Expires=1554856720&Signature=i9Et2W4Zz2MTisN2U61Vt6O1fq5KZT-CBXDfg2ovegF79hUSscQjNQC1SRPqUl9dB9KCb-bYIR~1GtzsyuvgUzh3W8yufSsMJGNKVqGJivvotr-c2Bv-7kdhXgP6aIjM~Wnvq~--10sCEHowuBgIOGnhe580G8xljhTrrZtlv-XJ8LMfOk-MNwBQsoZjsma4rS6Q8tagL-5vXyAPdd-bFD7C8EtZivJos34nRucLHZzyupEnn2d4znP8kzmzghMh29~vH1VwvG~UmOdAeRS78FM1K85zGPsP7iWXleKHcu5NiTf3GAoo8KOqtm37C0ekDWmJlSFpdaoIdTdU9occ6Q__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

### Statistics:
* Casella, G., & Berger, R. L. (2002). Statistical inference (Vol. 2). Pacific Grove, CA: Duxbury.
* Schervish, M. J. (2012). Theory of statistics. Springer Science & Business Media.

### (Distributional) Reinforcement Learning:

#### Primary:
* [Reinforcement Learning](http://incompleteideas.net/book/RLbook2018.pdf)
* [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* [Statistics and Samples in Distributional Reinforcement Learning](https://arxiv.org/abs/1902.08102)

#### Others:
* [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
* [An Analysis of Categorical Distributional Reinforcement Learning](https://arxiv.org/abs/1802.08163)
* [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
* [A Comparative Analysis of Expected and Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)