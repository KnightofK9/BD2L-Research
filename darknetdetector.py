from ctypes import *
import math
import random
import cv2
import constant
import os
from sys import platform


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


if platform == "linux" or platform == "linux2":
    lib_path = constant.DARK_NET_LIB_LINUX + "libdarknet.so"
elif platform == "win32":
    lib_path = constant.DARK_NET_LIB_WIN + "yolo_cpp_dll.dll"
else:
    lib_path = constant.DARK_NET_LIB_MAC + "libdarknet.so"

lib = CDLL(lib_path, RTLD_GLOBAL)
# lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num = num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res


def write_img(image_path, res, path):
    img = cv2.imread(image_path)
    for e in res:
        img = cv2.rectangle(img, (int(e[2][0]), int(e[2][1])), (int(e[2][0] + e[2][2]), int(e[2][1] + e[2][3])),
                            (0, 255, 0), 5)
    cv2.imwrite(path + "temp_predict.jpg", img)


if __name__ == "__main__":
    # net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    # im = load_image("data/wolf.jpg", 0, 0)
    # meta = load_meta("cfg/imagenet1k.data")
    # r = classify(net, meta, im)
    # print r[:10]
    net = load_net("lib/darknet/cfg/yolo.cfg", "yolo.weights", 0)
    meta = load_meta("lib/darknet/cfg/coco.data")
    r = detect(net, meta, "out/frame_00000.jpg")
    print r

class Detector:
    def __init__(self, input_cfg, intput_weight, input_data):
        self.net = load_net(input_cfg, intput_weight, 0)
        self.meta = load_meta(input_data)

    def detect_image(self, image_url):
        return detect(self.net, self.meta, image_url)

    def detect_all_image_in_folder(self, folder_url, output_folder_url):
        import time
        now = time.time()
        for filename in os.listdir(folder_url):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = folder_url + filename
                output_path = output_folder_url + filename
                res_list = self.detect_image(img_path)
                img = self.get_image_from_path(img_path)
                img = self.write_rect_to_image(img, res_list)
                self.write_image_to_path(img, output_path)
                print "{} :".format(filename)
                print "{}".format(res_list)
        later = time.time()
        difference = int(later - now)
        print "Task finished in {} s".format(difference)

    def get_image_from_path(self, img_path):
        return cv2.imread(img_path)

    def write_image_to_path(self, img, img_path):
        cv2.imwrite(img_path, img)

    def write_rect_to_image(self, img, res_list):
        meta_list = self.res_to_info(res_list, img.shape[0], img.shape[1])
        for meta in meta_list:
            img = cv2.rectangle(img, (meta.left, meta.top), (meta.right, meta.bottom), (0, 255, 0), 5)
            self.write_text_to_image(img, meta.name + self.format_prob(meta.prob), meta.left, meta.top - 20)
        return img

    def format_prob(self, prob):
        return " %.2f %%" % (prob * 100)

    def write_text_to_image(self, img, text, x, y):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (x, y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    def res_to_info(self, res_list, img_w, img_h):
        return list(map(lambda x: Meta(x, img_w, img_h), res_list))


class Meta:
    def __init__(self, res, img_h, img_w):
        self.name = (res[0])
        self.prob = (res[1])
        self.w = res[2][2]
        self.h = res[2][3]
        x = res[2][0]
        y = res[2][1]
        self.left = x
        self.right = x + self.w
        self.top = y
        self.bottom = y + self.h

        self.left -= self.w/2.0
        self.right -= self.w/2.0
        self.top -= self.h/2.0
        self.bottom -= self.h/2.0

        if self.left < 0:
            self.left = 0
        if self.right > img_w - 1:
            self.right = img_w - 1
        if self.top < 0:
            self.top = 0
        if self.bottom > img_h - 1:
            self.bottom = img_h - 1

        self.left = int(self.left)
        self.right = int(self.right)
        self.bottom = int(self.bottom)
        self.top = int(self.top)
