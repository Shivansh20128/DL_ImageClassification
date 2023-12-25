# GPU Accelerated Tensorflow 
In this project, I have created a deep learning project based on image classification of lions vs tigers in tensorflow using GPU

# About this project
If you have done deep learning before, then you must already know how a model for a binary classification is made. This project also aims to perform a binary classification of lions and tiger images. But in this project, we are going to use our GPUs to accelerate the process of training the model. By using GPUs in your system, you can speed up the process of training more than 300%. 

We know that training a deep learning neural network takes a lot of time when it comes to images. For example, once I created a model for classifying images of cats and dogs. The dataset had around 12000 images in the training set. When I performed the training with just my CPU, which is an intel 5 core 11th gen CPU, it took me around 90 minutes, which was after prioritizing the process to high priority. 

But when I used my GPUs to acceralte the process, it took me only 20 minutes, with no priorities changed. Yes! That is the scale of impact we are talking about. 

Now, before you decide if you want to use GPUs for your model training, you need to ensure that you have a GPU in your system. And here, we are only talking about NVIDIA GPUs, because although there are other GPUs available in the market like AMD, tensorflow support has not been added for them as of now. So, you need to have an NVIDIA GPU in your system. You can check that by going into the Task Manager>Performance. At the bottom of this window, if you see GPUs, then you are good to go.

If you are new to deep learning, no worries. I have got you covered. I will be building a setup from complete scratch so that you can follow.

# Steps to make a Deep learning Environment

