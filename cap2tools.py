
def build_data_gens(path_list, target_size=(224, 224), batch_size=8, 
                    shuffle=True):
    '''
    Creates ImageDataGenerator objects to feed CNN model
    
    Parameters:
    path_list(list or tuple of strings) - list of paths to image directories
    target_size(tuple of ints) - target pixel dimensions of images
    batch_size(int) - number of images to feed to CNN per batch
    shuffle(bool) - whether generators should shuffle the image order
    
    Returns:
    Tuple of ImageDataGenerator objects
    '''
    
    from keras.preprocessing.image import ImageDataGenerator
    
    generators = []
    for path in path_list:
        gen = ImageDataGenerator().flow_from_directory(path, 
                                                       target_size=target_size, 
                                                       batch_size=batch_size, 
                                                       shuffle=shuffle)
        generators.append(gen)
    
    return tuple(generators)


def build_VGG16(width, new_weights=False, trainable=False, 
                learning_rate=0.0001, dropout1=0, dropout2=0):
    '''
    Builds a modified version of the VGG16 model for transfer learning
    
    Parameters:
    width(int) - number of nodes present in each of the two new FC layers 
        after the convolutional layers
    new_weights(bool) - whether to reinitialize the weights in the VGG16
        convolutional layers
    trainable(bool) - whether to allow updating of convolutional weights
    learning_rate(float) - learning rate for Adam optimizer
    dropout1(float) - proportion of nodes to drop from first FC layer
    dropout2(float) - proportion of nodes to drop from second FC layer
    
    Returns:
    Compiled keras functional API model object
    '''
    
    from keras.applications import VGG16
    from keras.layers import Dense, Flatten, Dropout
    from keras.optimizers import Adam
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

    # compile model
    model = Model(inputs=base_model.inputs, outputs=preds)
    model.compile(optimizer=Adam(lr=learning_rate, decay=0.1), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


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
    fig, ax = plt.subplots(figsize=(15, 10))
    metric_df.plot(ax=ax)
    plt.title('Performance over epochs: {}'.format(metric))
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.xticks(range(0, len(metric_df.iloc[:, 1])))
    plt.legend(loc='upper right')
    plt.show()


def eval_models(model_paths, data_path):
    '''
    Evaluates performance of a model in terms of loss, accuracy, confusion
    matrix, and mean per-class recall and precision
    
    Parameters:
    model_paths(dict) - dictionary with model names as keys and paths pointing 
        to the .h5 files of the trained models as values
    data_path(string) - path to the image directory of the target dataset
    
    Returns:
    Dictionary of dictionaries each containing loss, accuracy, confusion matrix, 
    and mean per-class recall and precision for a given model
    '''
    
    from keras.models import load_model
    from keras.backend import clear_session
    from sklearn.metrics import confusion_matrix
    
    # build generator to feed the model
    print('Building image generator...')
    generator = build_data_gens([data_path], shuffle=False)[0]
    y_true = generator.classes

    # evaluate all models
    model_results = dict()
    for name, path in model_paths.items():
        # load model
        print('Loading model {}'.format(path))
        model = load_model(path)

        # run basic evaluation
        print('Evaluating model {}'.format(path))
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


def eval_table(model_metrics, index_name, decimals=3):
    '''
    Creates a formatted table summarizing model performance
    
    Parameters:
    model_metrics(dict) - an output dictionary of eval_models
    index_name(string) - name of the index of the output table
    decimals(int) - number of decimals for rounding the table values
    
    Returns:
    pandas DataFrame with fields of accuracy, loss, and mean per-class recall,
    and rows of models evaluated
    '''
    
    from pandas import DataFrame
    
    metrics = DataFrame(model_metrics).transpose()
    metrics.index.name = index_name
    metrics.drop('cm', axis=1, inplace=True)
    metrics = metrics.astype('float64').round(decimals)
    
    return metrics
    
