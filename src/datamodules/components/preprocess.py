# Run this file to preprocess the data
data_dir = '/home/SharedData/Vinit/pix3d'

# Importing the required libraries
import os
import random

# If the preprocessed data dir is present, delete it
if os.path.exists(data_dir + '_preprocessed'):
    os.system('rm -rf ' + data_dir + '_preprocessed')

# Make a new directory to store the preprocessed data
os.system('mkdir ' + data_dir + '_preprocessed')

new_data_dir = data_dir + '_preprocessed'

# Image folder
img_dir = data_dir + '/img'

shape_dir = data_dir + '/model'

# Make train and test folders
os.system('mkdir ' + new_data_dir + '/train')
os.system('mkdir ' + new_data_dir + '/test')

train_dir = new_data_dir + '/train'
test_dir = new_data_dir + '/test'

#Make img and model folders in train and test folders
os.system('mkdir ' + train_dir + '/img')
os.system('mkdir ' + train_dir + '/model')
os.system('mkdir ' + test_dir + '/img')
os.system('mkdir ' + test_dir + '/model')

train_img_dir = train_dir + '/img'
train_model_dir = train_dir + '/model'
test_img_dir = test_dir + '/img'
test_model_dir = test_dir + '/model'

# For each category copy 15% of the images to test and the rest to train
for category in os.listdir(img_dir):
    if category == 'misc':
        continue
    # Get the list of images in the category
    img_list = os.listdir(img_dir + '/' + category)
    # Get the number of images in the category
    num_imgs = len(img_list)
    # Get the number of images to be copied to test
    num_test_imgs = int(num_imgs * 0.1)
    # Get the number of images to be copied to train
    num_train_imgs = num_imgs - num_test_imgs
    # Get the list of images to be copied to test
    test_img_list = random.sample(img_list, num_test_imgs)
    train_img_list = [img for img in img_list if img not in test_img_list]

    # Make the directories for the category in train and test
    os.system('mkdir ' + train_img_dir + '/' + category)
    os.system('mkdir ' + test_img_dir + '/' + category)

    # Copy the images to test and train folders
    for img in test_img_list:
        os.system('cp ' + img_dir + '/' + category + '/' + img + ' ' + test_img_dir + '/' + category + '/' + img)
    for img in train_img_list:
        os.system('cp ' + img_dir + '/' + category + '/' + img + ' ' + train_img_dir + '/' + category + '/' + img)

# Copy the shapes to test and train folders
os.system('cp -r ' + shape_dir + ' ' + train_dir)
os.system('cp -r ' + shape_dir + ' ' + test_dir)

# Remove the misc folder from the train and test folders
os.system('rm -rf ' + train_model_dir + '/misc')
os.system('rm -rf ' + test_model_dir + '/misc')

