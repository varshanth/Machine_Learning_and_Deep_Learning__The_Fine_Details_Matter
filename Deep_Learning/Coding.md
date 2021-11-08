# Coding for Deep Learning

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
* https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/16
* PyTorch view, transpose, reshape and permute explained:  
https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/
* Plotting gradient flow to make sure all layers are learning. See Roshan Rane's answer:    
https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/5
* Performing a Conv operation with a custom kernel:  
https://discuss.pytorch.org/t/setting-custom-kernel-for-cnn-in-pytorch/27176

## Keras Stuff
* Grid Search With Keras:  
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/  
Grid Search Doc:  
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html  
* Early Stopping with Grid Search in Keras:  
https://stackoverflow.com/questions/48127550/early-stopping-with-keras-and-sklearn-gridsearchcv-cross-validation  

## H5PY: An efficient data format for deep learning datasets
* Learn to create groups and datasets from here: https://blade6570.github.io/soumyatripathy/hdf5_blog.html  
* Resizing H5 files makes it grow exponentially. Create dataset once with fixed shape:  https://discuss.pytorch.org/t/save-torch-tensors-as-hdf5/39556/2
* Make H5 loading faster with multi-processing: https://github.com/pytorch/pytorch/issues/11929
* Concurrent data manipulations using Single Write Multiple Read: https://docs.h5py.org/en/stable/swmr.html
