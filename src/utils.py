import os
import random
import csv
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def read_data(filename):
    #Read dataset
    df = pd.read_csv(filename)
    df = df.loc[df.postText.isna() == False]
    df = df.drop_duplicates()
    df.loc[df.truthClass =='clickbait', 'label'] = 1
    df.loc[df.truthClass !='clickbait', 'label'] = 0
    return df


def compute_metrics(y_test, predictions):
    ##Computes evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print('Accuracy Score: ', accuracy)
    print('F1 Score: ',f1)
    print('Precision Score: ',precision)
    print('Recall Score: ',recall)

    return accuracy, f1, precision, recall


def plot_result(value, history):
    #Plots value over training epochs
    plt.plot(history.history[value], label=value)
    plt.plot(history.history["val_" + value], label="val_" + value)
    plt.xlabel("Epochs")
    plt.ylabel(value)
    plt.title("Train and Validation {} Over Epochs".format(value), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

def train_model(model, X_train, y_train, learning_rate, epochs, batch_size, class_weights = False):
    #Set Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)
    # Callbacks
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    group_callbacks = [earlystopper, reducer]
    #Compile the model
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]  )
    start_time = time.time()
    # Fit the model
    if class_weights == False:
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, callbacks = group_callbacks, batch_size = batch_size)
    else:
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs,class_weight=class_weights, callbacks = group_callbacks, batch_size = batch_size)
    runtime = time.time() - start_time
    print("--- %s seconds ---" % (time.time() - start_time))
    plot_result("loss", history)
    plot_result("accuracy", history)
    return model, history, runtime

def evaluate_model(model, X_test, y_test, threshold = 0.5):
    # Evaluate model over test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy', accuracy)
    raw_predictions = model.predict(X_test)
    predictions =np.where(raw_predictions > threshold, 1, 0)
    return predictions

def write_metrics(name, accuracy, f1, precision, recall, runtime, cols = ['model', 'accuracy', 'f1_score', 'precision', 'recall', 'runtime'], results_path = '../trained_models', results_filename='results.csv'):
    results_file_path = results_path +'/'+results_filename
    # Create File and/or write results (if file already exists)
    if not os.path.exists(results_file_path):
        print("File does not exist. Creating results file.")
        header = cols
        results = [name, accuracy, f1, precision, recall, runtime]
        with open(results_file_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(results)   
    else:
        results = [name, accuracy, f1, precision, recall, runtime]
        with open(results_file_path, 'a+') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer = csv.writer(f)
            writer.writerow(results)   

def set_seed(seed):
    #Set random seed for all libraries
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
