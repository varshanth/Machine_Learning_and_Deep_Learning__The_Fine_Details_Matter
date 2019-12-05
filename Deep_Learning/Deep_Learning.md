# Deep Learning
These resources are not mine and I do not claim that I have written/composed/presented them. They are an aggregation of publicly available resources. If any of the authors do not want their resources cited here, please report it as an issue and I will remove it from the repo at the earliest.

## '101' Videos of Deep Learning
* CS231n Spring 2017 Stanford University  
https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk
* Deep Learning.ai - Andrew NG  
https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w

## Useful Blogs
* This blog is amazing. Many resources listed below are from this:  
https://www.analyticsvidhya.com/blog/category/deep-learning/
* https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html

## Data Preprocessing
* Difference between normalization and standardization and when to use them:  
https://machinelearningmastery.com/normalize-standardize-machine-learning-data-weka/  
* Useful thread on image normalization:  
http://forums.fast.ai/t/images-normalization/4058  
* ZCA Whitening: Amazing explanation (with simple maths):  
https://www.youtube.com/watch?v=eUiwhV1QcQ4
* Difference between PCA whitening and ZCA whitening:  
https://www.youtube.com/watch?v=eUiwhV1QcQ4

## Activation Functions
* Why do we need activation functions? Different Types of Activation Functions
* https://www.youtube.com/watch?v=-7scQpJT7uo
* CS231n Lecture 6
* Activation functions and when to use them:  
https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/
* ReLU isn't differentiable? Do we have to care?  
https://stackoverflow.com/questions/30236856/how-does-the-back-propagation-algorithm-deal-with-non-differentiable-activation
  
## Evaluation Metrics  
* Metrics borrowed from IR:  
https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf  

## Optimization Algorithms
* CS231n Lecture 7 -  Momentum, Nesterov momentum, AdaGrad, RMSProp, Adam
* https://www.analyticsvidhya.com/blog/2017/03/introduction-to-gradient-descent-algorithm-along-its-variants/
* Overview of SGD Algorithms by Sebastian Ruder: https://arxiv.org/pdf/1609.04747.pdf
* Excellent reasoning behind design of optimization techniques with code:  
https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
* Visualizing Optimizing Algorithms: https://imgur.com/a/Hqolp
* Excellent post on LR Scheduling/Annealing (literature as of end of 2017):  
https://www.jeremyjordan.me/nn-learning-rate/  

## Convolutional Neural Networks (CNN)
### Let's diverge for a second: What is convolution? (and why should CNN be actually called Cross Correlation-al Neural Network)
This is optional, I spent some time on understanding the motivation behind a "convolutional" approach and these were the resources I referred. This isn't mathemetically important to CNN, but intuitively gives a good idea as to what the filters in the CNN are trying to achieve. 
* Obviously: https://en.wikipedia.org/wiki/Convolution
* Notice the animation: Remember the convolution operation is the integration, so it is the area UNDER the curve.  
http://mathworld.wolfram.com/Convolution.html
* Graphical Convolution Example: https://www.youtube.com/watch?v=zoRJZDiPGds
* Convolution vs Cross Correlation - Udacity : https://www.youtube.com/watch?v=C3EEy8adxvc

### Coming Back: CNN
* CS231n Lec 5 - Internals of CNN, receptive field, calculating new dimensions after convolution etc.
* Calculating the effective field of a CNN:  
http://shawnleezx.github.io/blog/2017/02/11/calculating-receptive-field-of-cnn/
* How CNN Work by Brandon Rohrer: https://www.youtube.com/watch?v=FmpDIaiMIeA
* What are 1x1 convolutions used for:  
https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network  
* Convolution Arithmetic for Deep Learning:  
https://arxiv.org/pdf/1603.07285.pdf  

## Recurrent Neural Networks (RNN), Long Short Term Memory (LSTM) and Gated Recurrent Units (GRU)
* CS231n Lec 10 - RNN architecture, examples, drawbacks, requirement for LSTM, LSTM architecture etc.
* **Very important** to understand the difference between number of hidden units and RNN Cells because the definition in literature is different from that of implementation:   
https://stats.stackexchange.com/questions/241985/understanding-lstm-units-vs-cells  
In short, number of hidden units = size of the h vector, RNN cell in literature = A single RNN with a single hidden unit  
* How RNN & LSTMs work by Brandon Rohrer: https://www.youtube.com/watch?v=WCUNPb-5EYI
* Understanding LSTM - Colah's blog - Simple explanations with motivations:  
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
* RNN & LSTMs from intuition to backpropagation with examples by Rohan and Lenny:  
https://ayearofai.com/rohan-lenny-3-recurrent-neural-networks-10300100899b
* GRU detailed: https://arxiv.org/pdf/1412.3555v1.pdf
* Relationship between LSTM, GRU & Highway Network:  
https://www.researchgate.net/profile/Tamer_Alkhouli/publication/307889449_LSTM_GRU_Highway_and_a_Bit_of_Attention_An_Empirical_Overview_for_Language_Modeling_in_Speech_Recognition/links/57d3433908ae6399a38da357/LSTM-GRU-Highway-and-a-Bit-of-Attention-An-Empirical-Overview-for-Language-Modeling-in-Speech-Recognition.pdf
* Brief summary of RNN, LSTM and GRU:  
https://www.slideshare.net/gakhov/recurrent-neural-networks-part-1-theory
* Using different batch sizes with LSTM:  
https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
* Implementing mini batch algo in RNNs:  
https://www.quora.com/How-do-I-implement-mini-batch-algorithm-in-normal-RNN-and-Bidirectional-LSTM-RNN-for-NLP-task
* How are inputs fed to LSTM-RNN in mini batch method?  
https://www.quora.com/How-are-inputs-fed-into-the-LSTM-RNN-network-in-mini-batch-method  
* What is temperature in LSTM?  
https://www.quora.com/What-is-Temperature-in-LSTM

## Generative Adversarial Networks (GAN)
* Intro to GAN: Motivation to Algo:  
https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/

## Learning Caffe
* http://rodriguezandres.github.io/2016/04/28/caffe/  
* http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html  

## Learning PyTorch
* http://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/  
* https://jhui.github.io/2018/02/09/PyTorch-Variables-functionals-and-Autograd/  
* https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/  
* https://cs230-stanford.github.io/pytorch-getting-started.html  
* https://github.com/yunjey/pytorch-tutorial  
* https://github.com/bfortuner/pytorch-cheatsheet/blob/master/pytorch-cheatsheet.ipynb

### PyTorch Debugging
* https://stackoverflow.com/questions/48915810/pytorch-contiguous  
* https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference  

## Keras Stuff
* Grid Search With Keras:  
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/  
Grid Search Doc:  
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html  
* Early Stopping with Grid Search in Keras:  
https://stackoverflow.com/questions/48127550/early-stopping-with-keras-and-sklearn-gridsearchcv-cross-validation  

## Miscellaneous
* 25 Must Know Concepts for Beginners:  
https://www.analyticsvidhya.com/blog/2017/05/25-must-know-terms-concepts-for-beginners-in-deep-learning/  
* Different Types of Loss Functions:  
https://isaacchanghau.github.io/post/loss_functions/  
* Why do we need GPUs?  
https://www.analyticsvidhya.com/blog/2017/05/gpus-necessary-for-deep-learning/
* Understanding Internal Covariate Shift and BatchNorm:  
https://www.quora.com/Why-does-an-internal-covariate-shift-slow-down-the-training-procedure  
https://www.youtube.com/embed/Xogn6veSyxA?start=325&end=664&version=3 
* Weight Normalization & Layer Normalization:  
http://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/  
https://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2/  
* Multi-Label Classification using Neural Nets:  
https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/  
* Bilinear Interpolation: Image transformation using 2 degree neighborhood  
https://en.wikipedia.org/wiki/Bilinear_interpolation  
* Tradeoff between batch size and number of iterations:  
https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
* Why are NN becoming deeper not wider?  
https://stats.stackexchange.com/questions/222883/why-are-neural-networks-becoming-deeper-but-not-wider  
* Why Convolutions use odd numbers as filter size?  
https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size 

### Handling imbalanced data:  
* (Online) Hard Example Mining:  
http://www.erogol.com/online-hard-example-mining-pytorch/  
* General overview of the techniques:  
https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/  

## In Depth: [DL: Computer Vision](Computer_Vision/Computer_Vision_DL.md)
## In Depth: [DL: Natural Language Processing](Natural_Language_Processing/NLP_DL.md)

