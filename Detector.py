import Constant
import sys
import os
import cv2

import darknet


class Detector:
    def __init__(self, input_cfg, intput_weight, input_data):
        self.net = darknet.load_net(input_cfg, intput_weight, 0)
        self.meta = darknet.load_meta(input_data)

    def detect_image(self, image_url):
        return darknet.detect(self.net, self.meta, image_url)

    def detect_all_image_in_folder(self, folder_url, output_folder_url):
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
