from os import listdir, makedirs, errno, rename
from os.path import isfile, join, altsep
from natsort import natsorted, ns
from matplotlib import pyplot as plt
import csv
import yaml

def getSamples(path, mode):
    '''
    This function returns a list of sample names
    :param path: source path
    :param mode: 0 - images; 1 - yaml files
    :return:  list of full path of the samples
    '''
    samples = []
    if mode == 0:
        samples = [altsep.join((path, f)) for f in listdir(path)
                  if isfile(join(path, f)) and f.endswith('.png')]
    elif mode == 1:
        samples = [altsep.join((path, f)) for f in listdir(path)
                   if isfile(join(path, f)) and f.endswith('.reg')]
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

def getGTyaml(file):
    with open(file, 'r') as stream:
        try:
            next(stream)
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
