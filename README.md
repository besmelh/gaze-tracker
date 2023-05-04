# Notes about project

## About the project

### Summary

The goal of this project is to detect when a user is looking at their screen or not to help users reduce eye strain by implementing the 20:20:20 rule.

When the user looks at their screen the screen frame will be red, when they aren't it will be green. If their looking around the borders of the screen it may be yellow.

### Future implications

- At the moment the values of the screen border are hard coded to match my personal laptop. This will be updated to be more dynamic and be set based on the users screen and webcam.
- Create a fully functioning app that has a timer and turns on the camera to check if the user is looking at the screen or not for 20 seconds.
- Test the app across 

### Existing products

An existing[ gaze estimation trained model](https://github.com/david-wb/gaze-estimation) is used to detect where the user's eye gaze direction is.

This project was made to implement the 20:20:20 rule. The rule states that every 20 minutes, people should look away from their screens for at least 20 seconds, and look 20 feet away. Many applications that help apply this already exist, in which they lock or blackout the user's screen every 20 minutes for 20 seconds. Personally I have tried using these apps, but uncontiosuly I tend to continye looking at my screen even though it's blackened during those 20 seconds. Or I would look down at another device such as my phone. Therefore, I decided to develop this project which would ensure that the user is looking upwards for the duration of 20 seconds.

## Running the project on a local Mac machine

1. Clone the repo `git clone https://github.com/david-wb/gaze-estimation`
2. Make sure `conda` and `wget` are installed.
3. Initiate environment
   `conda env create -f env-mac.yml`
4. Activate environment
   `conda activate ge-mac`
5. Install torchvision manually because I deleted it from the original `env-mac.yml` file due to conflicting version dependencies
   `conda install torchvision`
6. Make sure environment is running and run
   `./scripts/fetch_models.sh`
7. Finally run main file
   `python run_with_webcam.py`

# Info from original repo: Gaze Estimation with Deep Learning

This project implements a deep learning model to predict eye region landmarks and gaze direction.
The model is trained on a set of computer generated eye images synthesized with UnityEyes [1]. This work is heavily based on [2] but with some key modifications.
This model achieves ~14% mean angular error on the MPIIGaze evaluation set after training on UnityEyes alone.

### Setup

NOTE: This repo has been tested only on Ubuntu 16.04 and MacOS.

First, create a conda env for your system and activate it:

```bash
conda env create -f env-linux.yml
conda activate ge-linux
```

Then download the pretrained model files. One is for detecting face landmarks. The other is the main pytorch model.

```bash
./scripts/fetch_models.sh
```

Finally, run the webcam demo. You will likely need a GPU and have cuda 10.1 installed in order to get acceptable performance.

```bash
python run_with_webcam.py
```

If you'd like to train the model yourself, please see the readme under `datasets/UnityEyes`.

### Materials and Methods

Over 100k training images were generated using UnityEyes [1]. These images are each labeled
with a json metadata file. The labels provide eye region landmark positions in screenspace,
the direction the eye is looking in camera space, and other pieces of information. A rectangular region around the eye was extracted from each raw traing image and normalized to have a width equal to the eye width (1.5 times the distance between eye corners).
For each preprocessed image, a set of heatmaps corresponding
to 34 eye region landmarks was created. The model was trained to regress directly on the landmark locations and gaze direction in (pitch, yaw) form. The model was implemented in pytorch. The overall method is summarized in the following figure.
![alt text](static/fig1.png 'Logo Title Text 1')

The model architecture is based on the stacked hourglass model [3]. The main modification was to add a separate pre-hourglass layer for predicting the gaze direction. The output of the additional layer is concatenated with the predicted eye-region landmarks before being passed to two fully connected layers. This way, the model can make use of the high-level landmark features for predicting the gaze direction.

### Demo Video

[![Watch the video](static/ge_screenshot.png)](https://drive.google.com/open?id=1WUUmd4quXq_YA5ANWDoUxqFGgguE_QJi)

### References

1. https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/
2. https://github.com/swook/GazeML
3. https://github.com/princeton-vl/pytorch_stacked_hourglass
