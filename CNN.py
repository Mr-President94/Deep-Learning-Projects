# part 1 - pre processing

# importing libraries
from keras.models import Sequential  # initialize NN as a sequnce of layers
from keras.layers import Convolution2D  # 2D coz it is an image not a video(3D) which has time stamp as well
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense  # to add fully connected layers

# intializing the CNN
cnn_classifier = Sequential()

# step 1 - convolution
cnn_classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
# arg1 = no. of feature detectors or filters arg2 = no. of rows of each feature detector arg3 = no. of cols of each
# feature detector arg4 = input_shape, i.e the shape of the image the goal is to force the shape of all images into
# the same format as the input images may be of different shape input_shape=(64, 64, 3) says that the dim. will be
# 64x64 and 3 represents the 3 channel of RGB as it is a colored image
# and also as classifying the image is a non-linear problem, we use rectifier to have non-linearity in our model

# step 2 - pooling
cnn_classifier.add(MaxPooling2D(pool_size=(2, 2)))
# pool_size represents the size of pooling image, usually kept 2x2 to not lose info and be precise as well

# adding 2d layer for better accuracy
# cnn_classifier.add(Convolution2D(32, 3, 3, activation='relu'))
# cnn_classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step 3 - flatten
cnn_classifier.add(Flatten())  # to create a vector of pooling image having a unique feature

# step 4 - full connection
cnn_classifier.add(Dense(output_dim=128, activation='relu'))  # input layer
cnn_classifier.add(Dense(output_dim=1, activation='sigmoid'))  # output layer (1 coz we need to predict a dog or a cat)

# compile(fit)
cnn_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# part 2 - fitting cnn to images
from keras.preprocessing.image import ImageDataGenerator

# template for Image preprocessing and fitting the model by taking the data from a directory
# image augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,                       # rescale all pixel values between 0 and 1
    shear_range=0.2,                        # random transvection (0.2) suggested by keras
    # A kind of linear mapping which leaves all points on one axis fixed, while other points are shifted parallel
    # to the axis by a distance proportional to their perpendicular distance from the axis
    zoom_range=0.2,                         # random zoom (0.2) suggested by keras
    horizontal_flip=True)                   # images will be flipped horizontally
# this generates many transformations so that we don't find same image in different batches


test_datagen = ImageDataGenerator(rescale=1. / 255)         # rescaling the test data


# training set (applying image aug on training set)
training_set = train_datagen.flow_from_directory('C:/Users/Rohit/Desktop/Data Science/Deep Learning/CNN 1/Part 2 - '
                                              'Convolutional Neural Networks/dataset/training_set',
                                              target_size=(64, 64), batch_size=32, class_mode='binary')
# target_size = sizeof images expected in the cnn model, same as the input shape
# batch_size = size of the batches where random samples of image will be included
#              after how many inputs the weights will be updated
# class_mode = 'binary' as this is a classification problem (2 classes)



# test set (applying image aug on testing set)
test_set = test_datagen.flow_from_directory('C:/Users/Rohit/Desktop/Data Science/Deep Learning/CNN 1/Part 2 - '
                                              'Convolutional Neural Networks/dataset/test_set',
                                            target_size=(64, 64), batch_size=32, class_mode='binary')

# fitting the model
cnn_classifier.fit(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=380)
# steps_per_epoch - no. of images in training set
# validation_steps - no. of images in testing set

# to improve accuracy add a 2nd convolutional layer and keep the target images size greater

# final output
# 8000/8000 [==============================] - 1707s 213ms/step - loss: 0.0129 - accuracy: 0.9960 -
# val_loss: 4.1958 - val_accuracy: 0.8160

# making predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('path', target_size=(64, 64))
test_set = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn_classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
