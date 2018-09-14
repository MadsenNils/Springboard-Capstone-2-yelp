
def build_datagens(train_path, valid_path, target_size=(224, 224), batch_size=8, 
                    shuffle=True, augment=False):
    '''
    Creates ImageDataGenerator objects to feed CNN model
    
    Parameters:
    train_path(str) - path to training set directory
    valid_path(str) - path to validation set directory
    target_size(tuple of ints) - target pixel dimensions of images
    batch_size(int) - number of images to feed to CNN per batch
    shuffle(bool) - whether generators should shuffle the image order
    augment(bool) - whether the training set generator should include real-
        time image augmentation
    
    Returns:
    Tuple of ImageDataGenerator objects
    '''
    
    from keras.preprocessing.image import ImageDataGenerator
    
    # training set datagen
    if augment == True:
        shift = 0.2
        train_datagen = ImageDataGenerator(horizontal_flip=True, 
                                           width_shift_range=shift, 
                                           height_shift_range=shift, 
                                           zoom_range=0.2, 
                                           fill_mode='reflect')
    else:
        train_datagen = ImageDataGenerator()

    train_batches = train_datagen.flow_from_directory(train_path, 
                                                      target_size=target_size, 
                                                      batch_size=batch_size, 
                                                      shuffle=shuffle)

    # validation set datagen
    valid_datagen = ImageDataGenerator()

    valid_batches = valid_datagen.flow_from_directory(valid_path, 
                                                      target_size=target_size, 
                                                      batch_size=batch_size, 
                                                      shuffle=shuffle)
    
    return tuple([train_batches, valid_batches])


def build_VGG16(width, new_weights=False, trainable=False, dropout1=0, 
                dropout2=0):
    '''
    Builds a modified version of the VGG16 model for transfer learning
    
    Parameters:
    width(int) - number of nodes present in each of the two new FC layers 
        after the convolutional layers
    new_weights(bool) - whether to reinitialize the weights in the VGG16
        convolutional layers
    trainable(bool) - whether to allow updating of convolutional weights
    dropout1(float) - proportion of nodes to drop from first FC layer
    dropout2(float) - proportion of nodes to drop from second FC layer
    
    Returns:
    Uncompiled keras functional API model object
    '''
    
    from keras.applications import VGG16
    from keras.layers import Dense, Flatten, Dropout
    from keras.models import Model
    
    if new_weights == True:
        weights = None
    else:
        weights = 'imagenet'
    
    # import only the convolutional layers of VGG16
    base_model = VGG16(include_top=False, 
                       weights=weights, 
                       input_shape=(224, 224, 3))
    
    if trainable == False:
        for layer in base_model.layers:
            layer.trainable = False
    
    # add two FC layers to end of convolutional layers
    inputs = base_model.output
    x = Flatten()(inputs)
    x = Dense(width, activation='relu')(x)
    x = Dropout(dropout1)(x)
    x = Dense(width, activation='relu')(x)
    x = Dropout(dropout2)(x)
    preds = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.inputs, outputs=preds)
    
    return model


def run_in_replicate(width, name, train_batches, valid_batches, replicates=2, 
                     n_epochs=10, new_weights=False, trainable=False, 
                     dropout1=0, dropout2=0, learning_rate=0.0001):
    '''
    Runs multiple replicates for each model training
    
    Parameters:
    width(int) - number of nodes in each of the fully connected layers
    name(str) - string to include in the filename
    train_batches - generator for feeding training images into the model
    valid_batches - generator for feeding validation images into the model
    replicates(int) - number of times to train the model variation
    n_epochs(int) - how many times the training will pass through the entire
        image set
    new_weights(bool) - whether to reinitialize the weights in the VGG16
        convolutional layers
    trainable(bool) - whether to allow updating of convolutional weights
    dropout1(float) - proportion of nodes to drop from first FC layer
    dropout2(float) - proportion of nodes to drop from second FC layer
    learning_rate(float) - learning rate for the Adam optimizer
    
    Returns:
    Training history averaged across the replicates as a dictionary of numpy
        arrays
    '''
    from datetime import datetime
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam
    from keras.backend import clear_session
    import numpy as np
    import pandas as pd
    
    filename = 'models/vgg16_{}_'.format(str(name))
    
    histories = []
    
    for idx in range(1, replicates+1):
        datetime_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} - Started training {}'.format(datetime_now, filename+str(idx)))

        # define callback for model saving
        saver = ModelCheckpoint(filename + str(idx) + '.h5', 
                                       monitor='val_loss', 
                                       verbose=0, 
                                       save_best_only=True)

        # build and train model
        model = build_VGG16(width, new_weights=new_weights, trainable=trainable,
                            dropout1=dropout1, dropout2=dropout2)
        
        model.compile(optimizer=Adam(lr=learning_rate, decay=0.1), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        hist = model.fit_generator(train_batches,
                                   validation_data=valid_batches,
                                   epochs=n_epochs, 
                                   callbacks=[saver], 
                                   verbose=0)
        histories.append(hist)
        
        # remove clutter from memory
        del model
        clear_session()
        
    def avg_metric(metric):
        average_metric = np.array(histories[0].history[metric])
        
        for idx in range(1, len(histories)):
            average_metric += np.array(histories[idx].history[metric])
            
        return average_metric/len(histories)
    
    # determine average metrics of the two runs
    acc = avg_metric('acc')
    loss = avg_metric('loss')
    val_acc = avg_metric('val_acc')
    val_loss = avg_metric('val_loss')
    
    return {'acc': acc, 'loss':loss, 'val_acc':val_acc, 'val_loss':val_loss}


def plot_metric(metric, histories):
    '''
    Plots one metric from dataframe of histories, for comparison of different
    models
    
    Parameters:
    metric(string) - the metric to be plotted
    histories(DataFrame) - CNN training history across epochs for different 
        models
        
    '''
    
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    
    # extract metric of interest
    metric_df = DataFrame()
    for i, history in histories.iterrows():
        metric_df[i] = history[metric]

    # plot history
    fig, ax = plt.subplots(figsize=(10, 8))
    metric_df.plot(ax=ax)
    plt.title('Performance over epochs: {}'.format(metric))
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.xticks(range(0, len(metric_df.iloc[:, 1])))
    plt.show()

def plot_history(history):
    '''
    Plots val loss and val accuracy history
    
    Parameters:
    history - keras history object for the model of interest
    
    Returns:
    None
    '''
    import matplotlib.pyplot as plt

    history = history.history
    n_epochs = len(history['loss'])
    
    # plot loss history
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.plot(history['loss'], label='training')
    plt.plot(history['val_loss'], label='validation')
    plt.title('Loss history')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(range(n_epochs))
    plt.legend(loc='upper right')
    plt.show()

    # plot acc history
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.plot(history['acc'], label='training')
    plt.plot(history['val_acc'], label='validation')
    plt.title('Accuracy history')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks(range(n_epochs))
    plt.legend(loc='lower right')
    plt.show()

    return None
    

def eval_models(model_paths, data_path):
    '''
    Evaluates performance of a model in terms of loss, accuracy, confusion
    matrix, and mean per-class recall
    
    Parameters:
    model_paths(dict) - dictionary with model names as keys and paths pointing 
        to the .h5 files of the trained models as values
    data_path(string) - path to the image directory of the target dataset
    
    Returns:
    Dictionary of dictionaries each containing loss, accuracy, confusion matrix, 
    and mean per-class recall for a given model
    '''
    
    from keras.models import load_model
    from keras.backend import clear_session
    from sklearn.metrics import confusion_matrix
    from keras.preprocessing.image import ImageDataGenerator
    
    # build generator to feed the model
    print('Building image generator...')
    generator = ImageDataGenerator().flow_from_directory(data_path, 
                                                       target_size=(224, 224), 
                                                       batch_size=8, 
                                                       shuffle=False)
    y_true = generator.classes

    # evaluate all models
    model_results = dict()
    for name, path in model_paths.items():
        # load model
        print('Loading {}'.format(path))
        model = load_model(path)

        # run basic evaluation
        print('Evaluating {}'.format(path))
        metrics = dict()
        metrics['loss'], metrics['acc'] = model.evaluate_generator(generator)
        
        # predict labels
        y_pred = model.predict_generator(generator)
        y_pred = y_pred.argmax(axis=1)

        # calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['cm'] = cm.tolist()
        
        # mean per-class recall
        pcr = cm.diagonal()/cm.sum(axis=1)
        metrics['pcr'] = pcr
        metrics['mpcr'] = pcr.mean()
        
        model_results[name] = metrics
        
        # remove clutter from memory
        del model
        clear_session()
    
    print('Evaluation complete.\n')
    
    return model_results


def print_eval(model_metrics):
    '''
    Prints out model metrics for a single model in more easily-readable format
    
    Parameters:
    model_metrics(dict) - single dictionary element of eval_models output,
        corresponding to the performance metrics of a single model
        
    Returns:
    Confusion matrix
    '''
    
    print('accuracy: ', model_metrics['acc'])
    print('loss: ', model_metrics['loss'])
    print('pcr: ', model_metrics['pcr'])
    print('mean pcr: ', model_metrics['mpcr'])
    print('confusion matrix: ')
    
    return model_metrics['cm']
        

def eval_table(model_metrics, index_name, columns=['acc', 'loss', 'mpcr'],  
               decimals=3):
    '''
    Creates a formatted table summarizing model performance
    
    Parameters:
    model_metrics(dict) - an output dictionary of eval_models
    index_name(string) - name of the index of the output table
    columns(list) - which columns to include in the table
    decimals(int) - number of decimals for rounding the table values
    
    Returns:
    pandas DataFrame with fields of accuracy, loss, and mean per-class recall,
    and rows of models evaluated
    '''
    
    from pandas import DataFrame
    
    metrics = DataFrame(model_metrics).transpose()
    metrics.index.name = index_name
    metrics = metrics[columns]
    metrics = metrics.astype('float64').round(decimals)
    
    return metrics
    
