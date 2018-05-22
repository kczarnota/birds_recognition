import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys

def save_plot(history, name):
    plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
    plt.plot(history['top_k_categorical_accuracy'])
    plt.plot(history['val_top_k_categorical_accuracy'])
    plt.title('Dokładność ({})'.format(name))
    plt.ylabel('Dokładność')
    plt.xlabel('Epoka')
    plt.legend(['top 1 train', 'top 1 val', 'top 5 train', 'top 5 val'], loc='upper left')
    plt.savefig('acc_{}.pdf'.format(name))
    plt.close()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss ({})'.format(name))
    plt.ylabel('Loss')
    plt.xlabel('Epoka')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss_{}.pdf'.format(name))
    plt.close()


with open(sys.argv[1], 'rb') as f:
    history = pickle.load(f)
    save_plot(history, sys.argv[2])

