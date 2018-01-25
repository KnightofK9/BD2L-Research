import cv2
import numpy as np
import os
import sys

SZ = 20
CLASS_N = 10


def shuffle_data_and_labels_in_place(arr1, arr2):
    seed = np.random.randint(0, sys.maxint)
    prng = np.random.RandomState(seed)
    prng.shuffle(arr1)
    prng = np.random.RandomState(seed)
    prng.shuffle(arr2)


def load_digits(path):
    digits = []
    labels = []
    for path, dirs, files in os.walk(path):
        if len(files) == 0:
            continue
        label = path.split("\\")[-1]
        for file in files:
            file_path = path + "/" + file
            digits.append(load_digit_img(file_path))
            labels.append(int(label))

    return (digits, labels)


def load_digit_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (25, 50))
    return img


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
        # Calculate skew based on central momemts.
    skew = m['mu11'] / m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    # resp = list(map(lambda x: str(int(x)),resp))
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err) * 100))
    for i in range(0, len(digits) - 1):
        if labels[i] ==  resp[i]:
            continue
        print "Number {} predicted {}".format(labels[i], resp[i])



def get_hog():
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)
    return hog

class BaseFeature:
    def compute(self,img):
        return None

class HogFeature(BaseFeature):
    def __init__(self):
        winSize = (20, 20)
        blockSize = (10, 10)
        blockStride = (5, 5)
        cellSize = (10, 10)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradients = True

        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)
        self.hog = hog

    def compute(self,img):
        return self.hog.compute(img)


class NumberPixelInCellFeature(BaseFeature):
    def __init__(self):
        pass

    def compute(self,img):
        return None


class StatModel(object):
    def load(self, fn):
        self.model = cv2.ml.SVM_load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=12.5, gamma=0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)


    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


if __name__ == '__main__':
    #  Load digits
    digits, labels = load_digits("./data/plate/train")
    # digits, labels = load_digits("./out")
    shuffle_data_and_labels_in_place(digits, labels)
    feature = HogFeature()
    print 'Calculating HoG descriptor for every image ... '
    hog_descriptors = []
    for img in digits:
        hog_descriptors.append(feature.compute(img))

    train_n = int(0 * len(hog_descriptors))
    digits_train, digits_test = np.split(digits, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    print 'Training SVM model ...'
    model = SVM()
    # model.train(hog_descriptors_train, labels_train)

    # print 'Saving SVM model ...'
    # model.save('digits_svm.dat')

    print "Reload SVM model from file"
    model.load('digits_svm.dat')

    print 'Evaluating model ... '
    evaluate_model(model, digits_test, hog_descriptors_test, labels_test)
    exit(0)