# GPU Accelerated Tensorflow 
In this project, I have created a deep learning project based on image classification of lions vs tigers in tensorflow using GPU

# About this project
If you have done deep learning before, then you must already know how a model for a binary classification is made. This project also aims to perform a binary classification of lions and tiger images. But here, we are going to use our GPUs to accelerate the process of training the model. By using GPUs in your system, you can speed up the process of training by more than 300%. 

We know that training a deep learning neural network takes a lot of time when it comes to images. For example, once I created a model for classifying images of cats and dogs. The dataset had around 12000 images in the training set. When I performed the training with just my CPU, which is an intel 5 core 11th gen CPU, it took me around 90 minutes, which was after prioritizing the process to high priority. 

But when I used my GPUs to acceralte the process, it took me only 20 minutes, with no priorities changed. Yes! That is the scale of impact we are talking about. 

Now, before you decide if you want to use GPUs for your model training, you need to ensure that you have a GPU in your system. And here, we are only talking about NVIDIA GPUs, because although there are other GPUs available in the market like AMD, tensorflow support has not been added for them as of now. So, you need to have an NVIDIA GPU in your system. You can check that by going into the Task Manager>Performance. At the bottom of this window, if you see GPUs, then you are good to go.

If you are new to deep learning, no worries. I have got you covered. I will be building a setup from complete scratch so that you can follow.

# Steps to make a Deep learning Environment

1. The first thing you need to do is install Anaconda on your system. If you already have Anaconda installed on your system, you need to check if it is the one that supports python version <=3.10. This is because tensorflow-gpu support has only been added till python 3.10 now. For smooth operation, install Anaconda with python version 3.9 from here- https://anaconda.org/anaconda/python/files?version=3.9.12
1. In the installation, it asks if you want to add Anaconda3 to the path variable, and if you want to register Anaconda3 as default Python3.9. Check both the boxes (Even if it becomes red).
1. Leave the other settings as they are install Anaconda on your system.


Now that you have Anaconda installed on you system, we are going to use it. But before you need to make a folder in which you want to make your project, and navigate to the folder in command prompt/terminal.

At this point, we are going to use some terminal commands to configure some things.

1. First, we are going to make a virtual environment so that your system environment does not get affected with whatever we are going to do. This will make it easy for you try out new things without worrying about your system's settings. Because in case you mess up, you can always delete the virtual environment and create a new one. To create a new vurtaul environment, enter the following command in command prompt:
```
python -m venv myenv
```
This will create a new virtual environment name "myenv". you can give it any other name.

2. After that, we will activate the virtual environment. To do that, enter the command:
```
.\myenv\scripts\activate
```
This will activate the virtual environment. You should be able to see the name your virtual environment at the start of the line now.

3. Now that you virtual environment is active, we will install "ipykernel" int it.
```
pip install ipykernel
```

4. Then attach the ipykernel to you virtaul environment.
```
python -m ipykernel install --name=myenv
```

If you made any mistakes, you can easily delete the virtual environment using the command:
```
jupyter kernelspec uninstall myenv
```

5. Now, you can open Jupyter lab with command ```jupyter lab```and explore it a bit if you are new. Try to make a jupyter notebook in the new enviroment and run it (You can do that by changing the kernel).

# Installing Tensorflow
Now we are going to install Tensorflow for Deep learning in our new environment. For that, go back to the terminal, and follow these steps.

1. Since we want to use GPUs in our developemnt, we need to install an older version of tensorflow because for the newer versions, support has not been added yet. I am going to install tensorflow 2.8 in my system. 
```
pip install tensorflow==2.8 tensorflow-gpu==2.8 matplotlib opencv-python
```

This installation may take some time because tensorflow is a big library. After the installation, you check the installation with the command ```pip list```. You will be able to see tensorflow2.8 and tensorflow-gpu 2.8 in the list.

2. 