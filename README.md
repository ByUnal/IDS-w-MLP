<h2 align="center">Intrusion Detection on DARPA by using Multi-layer Perceptron</h2>
<p align="center">
  Developed by <a href="https://github.com/ByUnal"> M.Cihat Unal </a> 
</p>

## Overview

This repository presents neural network based approach for intrusion detection while classifying the attacks.
One of the most important reasons why this study is valuable that the model not only detects whether the attack exist, 
but also it detects the type of the attacks. In other words, the presented approach aims to solve multi-class problem by using
Multi Layer Perceptron (MLP). Various neural architectures have been tried to find optimal model. The results show that 
even simpler neural networks are capable of detecting the attacks with high accuracy.

**This repository imitates the work in this study [[1]](#1)**

*I'm sharing the paper, thus further reading and investigation is eligible.*

## Run the Code
The repository containes pure python. Therefore, no additional library is needed but python3.x

## Dataset
The experiments in this study have been concluded on 1999 Defense Advanced Research Projects Agency (DARPA) intrusion 
detection evaluation dataset. The raw version of that contains more than 450,000 connection records, but 20,055 records 
are included in [[1]](#1). The authors stated that they created this subset manually by choosing a reasonable 
number of normal events and attack types arbitrarily. We've followed the same strategy in this study, but the authors didn't 
give the details about how they chose the data. Therefore, we've taken the 20,055 records randomly from the raw dataset. 

There are at least four different known categories of computer attacks including denial of service attacks, user to root 
attacks, remote to user attacks and probing attacks [[1]](#1) [[2]](#2). Sixty different 
attack types exist in 1999 DARPA dataset, however only two of them, which belong to different attack categories, 
used in this experiment: SYN Flood (Neptune) and Satan. These attack types are under Denial Of Service and Probing attack
categories respectively. The authors preferred to use these attack types due to the availability of enough data and 
the possibility to compare with previous works that use the same types. Table \ref{table:1} shows detailed information 
about the amount of attack types in train, validation and test datasets. In the following subsections, a description of 
the attack types is provided.

## Results
As aimed, we could imitate some of the results obtained in the original experiment, but not all of them. In the first
step of the experiment, there was a huge difference between experiments even though we used the exact parameters used
in the original experiment. This is reasonable due to several reasons. One reason is random initialization in weights.
Since we have no seed value to obtain similar weight initialization, train-test results differ. Another reason is the lack
of hyperparameter knowledge in the original experiment. Such as the activation function between hidden layers and
the learning rate are not given. Last, even though train-validation-test sample sizes are aligned with the original
experiment, data are not the same because they choose their data manually and don’t explain how to do this. Although
this is the case and the shape of the loss functions are similar, the generalization was better in our experiment.

They added early stopping as a second step for improvement, but it did not affect our experiments.
Again, it can be due to random weight initialization and different data distribution. In fact, I tried the training with 
several seed values to make results alike with original experiments, which I couldn’t achieve. 

In the third step of the experiment, the hidden-layer size was reduced to one and neuron size was increased to 45. We obtained 
nearly the same train-test accuracy. Besides, execution time decreased and no severe generalization loss occurred as 
intended in the original experiment.

Lastly, there is a huge difference between the execution times of experiments. This most probably occurs due to
improvements in the computational powers of computers during the past years. We’ve concluded the experiment after
20 years from the original experiment. Hence, excessive difference in execution time is expected and comprehensible.


## References
<a id="1">[1]</a> 
Moradi, M., & Zulkernine, M. (2004, November). 
A neural network based system for intrusion detection and classification of attacks. 
In Proceedings of the IEEE international conference on advances in intelligent systems-theory and applications (pp. 15-18). 
IEEE Lux-embourg-Kirchberg, Luxembourg.

<a id="2">[2]</a> 
Kendall, K. K. R. (1999). A database of computer attacks for the evaluation of intrusion detection systems (Doctoral dissertation, 
Massachusetts Institute of Technology).
