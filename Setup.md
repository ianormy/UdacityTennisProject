# Setup project
Here are notes about how to setup the project. 

Unfortunately, the Unity ML environment used by Udacity for this project is a very early version that is a few years old - [v0.4](https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0). This makes it extremely difficult to set things up, particularly in a Windows environment. 


## Getting Started
To setup the project follow those steps:

1. Provide an environment with `python 3.6.x` installed - ideally by creating a new virtual environment. An example would be:

```
virtualenv drlnd_tennis -py 3.6
```

2. Clone and install the requirements of the project: 
```
git clone git@github.com:ianormy/UnityTennisProject.git
cd UnityTennisProject
pip install -r requirements.txt
```

3. Install a version of pytorch compatible with your architecture. The version used by this project is 1.5.0. This is an old version, but it's the newest that works with this old project :-) To use the correct version that is compatible with your CUDA (if you want to use GPU) then you will need the correct install string. Please see [this document](https://pytorch.org/get-started/previous-versions/) for help with this. My version of CUDA is 10.1 on Windows so I used this command:

```
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

4. Install the unity environment provided by Udacity - **unityagents**. To help with that I have created a wheel that has everything in it. It's in the wheels folder of this repository. To install it simply do this:

```
pip install wheels\unityagents-0.4.0-py3-none-any.whl
```

5. Download and extract in the root of the project the environment compatible with your architecture:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
