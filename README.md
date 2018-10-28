# Springboard-Capstone-2-yelp
___

## Project Reports

**01_Capstone 2 Project Proposal.pdf** - Project proposal outlining goal and utility of the project, overview of the data available, and potential approaches.

**02_Capstone 2 Milestone Report.pdf** - Progress report containing introduction, data cleaning and image inspection, and hyperparameter tuning.

**03_Capstone 2 Final Report.pdf** - Final report containing introduction, data cleaning and image inspection, hyperparameter tuning, solutions to class imbalance, final test set performance, and possible future directions.

**04_Capstone 2 Final Slides.pptx** - Final report in slide format, focusing on major points.

## Information for Image Filename, Label, and Set

**yelp_academic_dataset_photo.json** - Original json file for the image portion of the Yelp dataset

**photo_labels_all.csv** - Photo filenames and labels for the full image set, divided into training, validation, and test sets.

**photo_labels_downsampled.csv** - Photo filenames and labels for the training and validation sets after downsampling to balance class representation 

## Code

*Note: The code was split into multiple notebooks in order to reduce the memory load on my system*

**cap2tools.py** - Custom python module for storing various functions written for this project.

**01_dataset_sampling.ipynb** - Samples the full image set into train, validation, and test sets, with copying of files into relevant directories to enable feeding Keras CNN with flow_from_directory. Also, downsamples the full train and validation image sets to create class-balanced image sets for model hyperparameter tuning.

**02_VGG16_pretraining_comparison.ipynb** - Comparison among using pre-trained ImageNet convolutional weights, with and without further training allowed, and freshly initialized weights. Also add image augmentation.

**03_VGG16_width_comparison.ipynb** - Tunes number of nodes in the fully-connected layers following the convolutional layers.

**04_VGG16_learnrate_comparison.ipynb** - Tunes learning rate of the Adam optimizer

**05_VGG16_dropout1_comparison.ipynb** - Tunes dropout rate for the first fully-connected layer

**06_VGG16_dropout2_comparison.ipynb** - Tunes dropout rate for the second fully-connected layer

**07_downsample_model_training.ipynb** - Trains 10 models for 10 epochs on the downsampled training set. Trains the best model to convergence on the downsampled training set, then evaluates performance on the full validation set. Produces 'Downsampled' model

**08_full_set_model_training.ipynb** - Trains models on the full training set, with and without balanced class weights, and evaluates on the full validation set. Produces 'Full-Unbalanced' and 'Full-Balanced' models.

**09_combined_model_training.ipynb** - Combines 'Downsampled' and 'Full-Unbalanced' models with new output layer and trains on downsampled training set. Evaluates on full validation set. Produces 'Combined' model.

**10_hybrid_training.ipynb** - Performs two-phase training by training 'Downsampled' model for 5 epochs on full training set, with both balanced and root-balanced class weights. Evaluates models on the full validation set. Produces 'Hybrid-Balanced' and 'Hybrid-Root' models.

**11_figure_generation.ipynb** - Produces tables and figures for final report.

## Training histories

VGG16_dropout1_comparison_history.json

VGG16_dropout2_comparison_history.json

VGG16_learnrate_comparison_history.json

VGG16_pretraining_comparison_history.json

VGG16_width_comparison_history.json
