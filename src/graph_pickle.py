import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys

def save_plot(history, name, cut):
    plt.plot(history['categorical_accuracy'][0:cut])
    plt.plot(history['val_categorical_accuracy'][0:cut])
    plt.plot(history['top_k_categorical_accuracy'][0:cut])
    plt.plot(history['val_top_k_categorical_accuracy'][0:cut])
    print("top1: {}".format(history['categorical_accuracy'][-1]))
    print("top5: {}".format(history['top_k_categorical_accuracy'][-1]))
    print("vtop1: {}".format(history['val_categorical_accuracy'][-1]))
    print("vtop5: {}".format(history['val_top_k_categorical_accuracy'][-1]))
    plt.title('Dokładność ({})'.format(name))
    plt.ylabel('Dokładność')
    plt.xlabel('Epoka')
    plt.legend(['top 1 train', 'top 1 val', 'top 5 train', 'top 5 val'], loc='upper left')
    plt.savefig('acc_{}.pdf'.format(name.replace(" ", "_").lower()))
    plt.close()

    plt.plot(history['loss'][0:cut])
    plt.plot(history['val_loss'][0:cut])
    plt.title('Loss ({})'.format(name))
    plt.ylabel('Loss')
    plt.xlabel('Epoka')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss_{}.pdf'.format(name.replace(" ", "_").lower()))
    plt.close()


with open(sys.argv[1], 'rb') as f:
    history = pickle.load(f)

    if len(sys.argv) > 3:
        limit = int(sys.argv[3])
    else:
        limit = -1

    save_plot(history, sys.argv[2], limit)

