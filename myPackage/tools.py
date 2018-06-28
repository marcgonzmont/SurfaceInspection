from os import listdir, makedirs, errno, rename
from os.path import isfile, join, altsep
from natsort import natsorted, ns
from matplotlib import pyplot as plt
import numpy as np
import itertools

def getSamples(path):
    '''
    This function returns a list of sample names
    :param path: source path
    :return:  list of full path of the samples
    '''
    samples = [altsep.join((path, f)) for f in listdir(path)
              if isfile(join(path, f))]

    return samples

# def getSamplesTess(path):
#     samples = [altsep.join((path, f)) for f in listdir(path)
#               if isfile(join(path, f)) and f.endswith('.jpg')]
#     return samples

def makeDir(path):
    '''
    To create output path if doesn't exist
    see: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
    :param path: path to be created
    :return: none
    '''
    try:
        makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def natSort(list):
    '''
    Sort frames with human method
    see: https://pypi.python.org/pypi/natsort
    :param list: list that will be sorted
    :return: sorted list
    '''
    return natsorted(list, alg=ns.IGNORECASE)

def plotImages(titles, images, title, row, col):
    fig = plt.figure()
    for i in range(len(images)):
        plt.subplot(row, col, i + 1), plt.imshow(images[i])     #, 'gray'
        if len(titles) != 0:
            plt.title(titles[i])
        plt.gray()
        plt.axis('off')

    # plt.tight_layout()
    fig.suptitle(title, fontsize=14)
    plt.show()


def renameFiles(samples_path):
    files = listdir(samples_path)
    i = 0
    for file in files:
        rename(join(samples_path, file), join(samples_path,'img'+str(i)+'.png'))
        i = i+1

def parseSample(path):
    import cv2
    '''
    Read image and parse .reg file (GT bboxes)
    :param image:
    :param reg:
    :return:
    '''
    image = path + '.png'
    im = cv2.imread(image)

    # Read .reg file
    fs_read = cv2.FileStorage(path + ".reg", cv2.FILE_STORAGE_READ)
    boxes = fs_read.getNode("rectangles").mat()
    fs_read.release()

    # From [[x1,x2..],[y1..],[w1..],[h1..] to [[x1,y1,w1..][x2..]..]
    if boxes is not None:
        boxes = boxes.T

    return im, boxes


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cm: confusion matrix
    :param classes: array of classes' names
    :param normalize: boolean
    :param title: plot title
    :param cmap: colour of matrix background
    :return: plot confusion matrix
    '''

    # plt_name = altsep.join((plot_path,"".join((title,".png"))))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    print('\nSum of main diagonal')
    print(np.trace(cm))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label', labelpad=0)

    plt.show()

def computeMetrics(cnf_matrix):
    precision = cnf_matrix[0][0]/np.sum(cnf_matrix, axis=1)[0]
    recall = cnf_matrix[0][0] / np.sum(cnf_matrix, axis=0)[0]
    accuracy = np.trace(cnf_matrix)/np.sum(cnf_matrix)

    print("Precision: {:.3f}\n"
          "Recall: {:.3f}\n"
          "Accuracy: {:.3f}\n".format(precision, recall, accuracy))