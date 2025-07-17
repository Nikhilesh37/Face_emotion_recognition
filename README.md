# Face_emotion_recognition

# Tech Stack

Python 3.10

Flask

Tensorflow

Numpy

Datetime

# Key Features:

Real-time Emotion Detection: Classifies emotions from a live webcam feed.

Multi-face Detection: Capable of detecting and classifying emotions for multiple faces in the frame.

High Accuracy: Achieves competitive accuracy on the FER-2013 dataset.

Modular Code: Well-structured and commented code, making it easy to understand and modify.

# Dataset Overview:

The scope of this project includes the design, training, and deployment of an FER
system, centered on the RAF-DB dataset. This dataset consists of approximately
29,672 grayscale facial images, each resized to 48x48 pixels, annotated by 40
labelers to reflect a diverse range of age, gender, ethnicity, and environmental
conditions like lighting and occlusions. The images are divided into training and
test sets, labeled with the seven basic emotions mentioned earlier. This rich and
varied collection serves as the foundation for training and evaluating our CNN
models, with the ultimate goal of deploying the best-performing model through a
web interface for real-time emotion prediction, demonstrating the project’s
practical potential.

# Web Application implimentation:

The website lets you drag and drop a 48x48 grayscale face pic, and boom—it tells you if the person’s happy, sad, or maybe even a little disgusted I threw in a sleek design with Bootstrap to make it
look good, adding a card layout and some stylish buttons. It’s not just functional;
it feels user-friendly, like something you’d actually want to play around with.
The process wasn’t all smooth sailing—figuring out how to connect the model to
the website took some trial and error, especially with those pesky input shape
errors we fixed. But now, running it locally at http://127.0.0.1:5000/ feels like a
big win. This step makes our project stand out, showing we can not only build
smart models but also share them with the world in a practical way 

# Architecture

CNN Architectures for Facial Expression Recognition
Our Facial Expression Recognition (FER) system uses three distinct Convolutional Neural Network (CNN) architectures—AlexNet, LeNet, and SimpleCNN—each tailored for the 48x48 grayscale images of the RAF-DB dataset.

AlexNet: Deep Feature Extraction
We adapted the renowned AlexNet architecture for our smaller image size. By adjusting its deep convolutional and fully connected layers, we created a model that effectively extracts robust facial features, balancing network depth with computational efficiency to handle variations in lighting and occlusions.

LeNet: Foundational Efficiency
LeNet was selected for its simplicity and efficiency, making it ideal for the RAF-DB dataset. As a foundational CNN with fewer layers, it allows for rapid experimentation and baseline performance assessment without significant computational overhead.

SimpleCNN: A Balanced Custom Approach
To bridge the gap between AlexNet's complexity and LeNet's simplicity, we developed SimpleCNN. This custom architecture includes additional convolutional layers to better capture subtle emotional cues. It was iteratively refined to enhance its learning capacity and ensure it generalizes well across the diverse faces in the dataset.

All three models were designed to be trained using the Adam optimizer with a categorical crossentropy loss function, using accuracy as the primary evaluation metric.
