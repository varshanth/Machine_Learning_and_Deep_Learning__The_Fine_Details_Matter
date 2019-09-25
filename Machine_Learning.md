# Machine Learning
These resources are not mine and I do not claim that I have written/composed/presented them. They are an aggregation of publicly available resources. If any of the authors do not want their resources cited here, please report it as an issue and I will remove it from the repo at the earliest.

## Winnow & Perceptron:
* J. Kivinen, M.K. Warmuth, P. Auer, The perceptron algorithm versus winnow: linear versus logarithmic mistake bounds when few input variables are relevant  
http://www.sciencedirect.com/science/article/pii/S0004370297000398

* (Other topics also covered here) SIMS 290-2: Applied Natural Language Processing: Marti Hearst & Barbara Rosario:  
courses.ischool.berkeley.edu/i256/f06/lectures/lecture17.ppt

* 8803 Machine Learning Theory. Maria-Florina Balcan: The Winnow Algorithm - www.cs.cmu.edu/~ninamf/ML11/lect0906.pdf

## Linear, Lasso and Ridge Regression
* Linear vs Lasso vs Ridge:  
https://discuss.analyticsvidhya.com/t/comparison-between-ridge-linear-and-lasso-regression/8213
* Machine Learning Thoughts: When does sparsity occur:  
www.ml.typepad.com/machine_learning_thoughts/2005/11/when_does_spars.html
* Excellent explanation of L1 vs L2 Regularization by Prof. Alexander Ihler:  
https://www.youtube.com/watch?v=sO4ZirJh9ds&list=PLkWzaBlA7utJMRi89i9FAKMopL0h0LBMk&index=16
* Visualizing Norms as a unit circle: https://www.youtube.com/watch?v=SXEYIGqXSxk
* Proximal Operator as the Shrinkage Operator in Soft Thresholding Algorithms:  
http://www.onmyphd.com/?p=proximal.operator
* Derivation of the soft thresholding operator:  
https://math.stackexchange.com/questions/471339/derivation-of-soft-thresholding-operator
* Differentiable criteria: When can I say a function is differentiable? Useful for the understanding the soft thresholding operator:  
https://www.mathsisfun.com/calculus/differentiable.html

## Logistic Regression
* Bernoulli Distribution: https://en.wikipedia.org/wiki/Bernoulli_distribution
* Logits: https://stats.stackexchange.com/questions/52825/what-does-the-logit-value-actually-mean
* Logits & Log-odds:  
https://stats.idre.ucla.edu/stata/faq/how-do-i-interpret-odds-ratios-in-logistic-regression/
* Likelihood Function for Logistic Regression: Prof. Cosma Shalizi's 2012 Lecture:  
http://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf
* Simple Derivation of Logistic Regression:  
http://www.win-vector.com/blog/2011/09/the-simpler-derivation-of-logistic-regression/
* y = {0,1} vs y = {-1,1} Loss function derivation:  
http://www.robots.ox.ac.uk/~az/lectures/ml/2011/lect4.pdf

## Nearest Neighbors
* The only resource you will ever need to understand: Prof. Victor Lavrenko's:  
https://www.youtube.com/playlist?list=PLBv09BD7ez_68OwSB97WXyIOvvI5nqi-3
* Mahalanobis Distance:
1. Explanation by similarity: Gopal Malakar: https://www.youtube.com/watch?v=3IdvoI8O9hU
2. Explanation through example: Matthew E. Clapham: https://www.youtube.com/watch?v=spNpfmWZBmg

## Hard & Soft Margin Support Vector Machines & KKT Conditions
Warning: These links might contain kernel concepts which are covered in the next section. So you might want to ignore kernels until you read that.
* Prof. Alexander Ihler
1. Part 1: https://www.youtube.com/watch?v=IOetFPgsMUc
2. Part 2: https://www.youtube.com/watch?v=1aQLEzeGJC8
* Prof. Patrick Winston: https://www.youtube.com/watch?v=_PwhiWxHK8o

* Dual SVM & Kernels by Bert Huang: https://www.youtube.com/watch?v=XkpsruJC5Mk
* De Facto PDF for SVM Lecture by Prof. Andrew Zisserman; Simple, maths explained well with examples:  
http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf
* A well organized ppt to summarize the above: Support Vector Machines & Kernels by Prof. Bryan Pardo:  
http://www.cs.northwestern.edu/~pardo/courses/eecs349/lectures/eecs349_support_vector_machines.pdf
* MLPR: SVM by Prof. Coryn Bailer-Jones:  
http://www.mpia.de/homes/calj/ss2007_mlpr/course/support_vector_machines.odp.pdf
* A bit complicated but if you have seen the above, you will understand this:  
A Tutorial on SVM for Pattern Recognition by Christopher J.C Burges: www.cmap.polytechnique.fr/~mallat/papiers/svmtutorial.pdf

* Support Vector Regression: Hard Margin & Soft Margin  
http://www.saedsayad.com/support_vector_machine_reg.htm  

* Advanced: v-SVM:
1. A Tutorial on v-SVM by Chen et. al:  
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.407.9879&rep=rep1&type=pdf
2. A Geometric Interpretation of v-SVM Classifiers by Crisp & Burges:  
https://papers.nips.cc/paper/1687-a-geometric-interpretation-of-v-svm-classifiers.pdf


## Kernels
* Prof. Alexander Ihler: https://www.youtube.com/watch?v=OmTu0fqUsQk
* Visualizing a polynomial kernel by Udi Aharoni: https://www.youtube.com/watch?v=3liCbRZPrZA
* The Kernel Trick - Udacity : https://www.youtube.com/watch?v=N_r9oJxSuRs
* Representer Theorem, Kernel Examples and Proofs by Prof. Peter Bartlett:  
https://people.eecs.berkeley.edu/~bartlett/courses/281b-sp08/8.pdf
*  Positive Definite Kernels, RKHS, Representer Theorem: NPTEL - Prof. P.S Sastry  
https://www.youtube.com/watch?v=_dyUl_luJl4

## Gaussian Processes
* From Scratch by Prof. Nando de Freitas
1. https://www.youtube.com/watch?v=4vGiHC35j9s
2. https://www.youtube.com/watch?v=MfHKW5z-OOA

## Mixture of Gaussians
* Gaussian Basics to Expectation Maximization Algorithm by Prof. Victor Lavrenko:  
https://www.youtube.com/watch?v=3JYcCbO5s6M&list=PLBv09BD7ez_7beI0_fuE96lSbsr_8K8YD
* In case you understand the above, this might help mathematically cement your understanding:  
MathematicalMonk Videos 16.3 to 16.9: https://www.youtube.com/watch?v=AnbiNaVp3eQ

* Application of EM Algorithm: Probabilistic Latent Semantic Analysis by Prof. ChengXiang Zhai
1. https://www.youtube.com/watch?v=vtadpVDr1hM
2. https://www.youtube.com/watch?v=hrSjJo1Z-UE

## Basic Feedforward Neural Networks
* Verbal Explanation: http://www.explainthatstuff.com/introduction-to-neural-networks.html
* Inspiration: A visual proof that neural nets can compute any function by Michael Nielsen:  
http://neuralnetworksanddeeplearning.com/chap4.html
* But what *is* a neural network by 3Blue1Brown: https://www.youtube.com/watch?v=aircAruvnKk 
* Explanation with a bit of math: https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/

### Backpropagation
* What is backprop and what is it actually doing? by 3Blue1Brown:  
https://www.youtube.com/watch?v=Ilg3gGewQ5U
* Backpropagation Calculus by 3Blue1Brown: https://www.youtube.com/watch?v=tIeHLnjs5U8
* Intuition converting to maths: https://www.youtube.com/watch?v=An5z8lR8asY
* Backprop maths: https://www.youtube.com/watch?v=gl3lfL-g5mA
* More Backprop maths: https://www.youtube.com/watch?v=aVId8KMsdUU

## Density Estimation

* What is PDF?  
https://www.quora.com/How-does-one-interpret-probability-density-greater-than-one-What-is-the-physical-significance-of-probability-density-Is-it-just-a-mathematical-tool  
* Histograms vs KDE. Visualizations and Explanations:  
https://mglerner.github.io/posts/histograms-and-kernel-density-estimation-kde-2.html  
* KDE Brief Explanation:  
http://www.mvstat.net/tduong/research/seminars/seminar-2001-05/  
* KDE Explanation with Visualization:  
https://www.quora.com/What-is-the-intuitive-explanation-of-the-formula-for-kernel-density-estimator  
https://www.quora.com/What-is-kernel-density-estimation  
* KDE Master Explanation with Maths:  
http://faculty.washington.edu/yenchic/18W_425/Lec6_hist_KDE.pdf
* KDE in Python (Influence and Diversity of Parameters):  
https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

## Miscellaneous
* What is the difference between a generative and discriminative algorithm?  
https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm
* What is the difference between a hyperparameter and a parameter?  
https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/
