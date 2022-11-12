# Dog-Identification-App

# Overview
At the end of this project, the code accepts a user-supplied image as input. When a dog is recognized in an image, it provides an estimate of its breed. When a human is found, it provides an estimate of the dog breed that most closely resembles that human.

# Motivation
I have completed several guided projects and this was an opportunity to showcase my skills and creativity. In this final project, you'll use what you learned during the program to build a data science project.


# Instructions

We have to complete all of the given task below:
Steps
 0: Import Datasets
 1: Detect Humans
 2: Detect Dogs
 3: Create a CNN to Classify Dog Breeds (from Scratch)
 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
 6: Write your Algorithm
 7: Test Your Algorithm


# Requirements
  
  Scipy
  OpenCV
  Pandas
  scikit-learn
  keras
  Tensorflow
  Matplotlib
  matplotlib
  Tqdm
  Pillow
  Skimage
  IPython Kernel
  NumPy
  
  We recommend installing Anaconda, a pre-built Python distribution that includes all the libraries and software needed for this project.
  

# Files

  dog_app.ipynb - File where code and possible solutions can be found.

  bottleneck_features - If you chose one of the architectures above, download the appropriate bottleneck feature and save the downloaded file in your     repository's bottleneck_features/ folder.

  Images Folder - Find images here to test the algorithm. Use at least two photos of the person and her two dogs.

  saved_models - the location of the models you worked with is saved_models.
  
  
 # Improve Results:
 
 Create a CNN for Dog Breed Classification (From Scratch)

 I created a CNN from scratch using transfer learning and trained the CNN with a test accuracy of 1.4%.
The main steps required to create a convolutional neural network to classify images are:

 1. Creating convolutional layers by applying kernels or feature maps. 
 2. 2. Apply Max Pool to the translation invariance. 
 3. 3. Input flattening. 
 4. 4. Creating a fully connected neural network. Five. Train your model. 
 5. 6. Output prediction. A convolutional layer treats an input image viewed as a two-dimensional matrix. Apply a convolution to the input data using a convolution filter to create a feature map. The size of Kernel is the size of the convolutional layer's filter matrix. So a kernel size of 2 means a 2x2 filter matrix or feature detector. The filter is pushed into the inlet.
You can control the behavior of the convolutional layers by controlling the number of filters and the size of each filter. To increase the number of nodes in the convolutional layer, we can increase the number of filters. You can also increase the size of the filter to increase the size of the recognized pattern.
Padding is also performed so that the height and width of the output feature map match the input. ReLU (Rectifier Linear Unit) is the activation function we use here to deal with the nonlinearity of neural networks. The first layer also receives an input image in the format 224,224,3.
Higher dimensionality uses more parameters, which can lead to overfitting. Pooling is used for dimensionality reduction. Here Max Pooling Layer was used to provide transformation invariance. Translational invariance means that the output does not change with small changes in the input. Max pooling reduces the number of cells. Pooling helps to recognize features such as colors, edges, etc. To maximize pooling, we use a 2 x 2 pool_size matrix for all 32 feature maps.
Then I used the "flat" layer to act as a connection between the folds and the dense layer. This allows us to smooth inputs that serve as inputs to a fully connected neural network.
Dense is the layer type to use for the output layer, as is often used in neural networks.
Also, use a 20% dropout rate to prevent overfitting. In the ch dog category, equipped with softmax.
The activation function used near the output layer is 'softmax', which formats the output from 0 to 1, so the output can be interpreted as probabilities. These probabilities then serve as output prediction probabilities.
Classify Dog Breeds Using CNN (with Transfer Learning)

 We used CNN to classify dog ​​breeds using a pre-trained VGG-16 model with a test accuracy of 38.1579%.
This model uses a pretrained VGG-16 model as a fixed feature extractor and the final convolution output from VGG-16 is fed as input to the model. Add only layers that are fully connected with the global average pooling layer. The latter contains a node for each dog category and is equipped with softmax. Building a CNN for Dog Breed Classification (Using Transfer Learning)

 We then used transfer learning to create a CNN that could identify dog ​​breeds from images with 80.1435% accuracy on the test set.
My final CNN architecture is built on the Resnet50 bottleneck. Additionally, GlobalAverage used Pooling2D to reduce vector features. These vectors were injected into the fully connected layer towards the end of the ResNet50 model. The fully connected layer contains a node for each dog category and is supported by the softmax function.

# Results

The Learning Transfer - Resnet50 model is used here to demonstrate how to implement an algorithm for a dog identification application. When a user provides an image, the algorithm first detects whether the image is of a human or a dog. If it's a dog, the breed predicts it. For humans, it returns similar dog breeds. This model yields a test accuracy of approximately 80%. This work also suggests a range of further improvements.
<img width="1098" alt="image" src="https://user-images.githubusercontent.com/31254252/201472278-b720497a-9c85-49df-b3d7-fecc3c8a8092.png">

<img width="873" alt="image" src="https://user-images.githubusercontent.com/31254252/201472296-a114a110-2296-43d0-a663-cf2ac03a007a.png">


This model uses a pretrained VGG-16 model as a fixed feature extractor and the final convolution output from VGG-16 is fed as input to the model. Add only layers that are fully connected with the global average pooling layer. The latter contains a node for each dog category and is equipped with softmax. Building a CNN for Dog Breed Classification (Using Transfer Learning)
