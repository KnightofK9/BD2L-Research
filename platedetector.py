import cv2


class PlateDetector:
    def __init__(self):
        pass

    def load_and_detect_img(self, img_path):
        img = cv2.imread(img_path)
        self.detect_img(img)

    def detect_img(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adtv_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)
        retval = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        adtv_img = cv2.erode(adtv_img, retval)
        adtv_img = cv2.dilate(adtv_img, retval)
        adtv_img, contours, hierarchy = cv2.findContours(adtv_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                                         offset=(0, 0))
        if len(contours) <= 0:
            return False
        color = (0, 255, 0)
        height, width, depth = img.shape
        for contour in contours:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            if rw > width / 2.0 or rh > width / 2.0 or rw < 120 or rh < 20 or (rw * 1.0) / rh > 4.5 or (
                    rw * 1.0) / rh < 3.5:
                continue
            img = cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), color, 2)
            sub = adtv_img[ry:ry + rh, rx:rx + rw]
            sub, sub_contours, sub_hierarchy = cv2.findContours(sub, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                                               offset=(0, 0))
            if len(sub_contours) < 8:
                continue
            subr_list = []
            for sub_contour in sub_contours:
                srx, sry, srw, srh = cv2.boundingRect(sub_contour)
                if srh > rh/2 and srw < rw/8 :
                    ratio = srw*(1.0)/srh
                    if ratio > 0.2 and ratio < 0.7:
                        subr_list.append({
                            "x":srx,
                            "y":sry,
                            "w":srw,
                            "h":srh
                        })
            if len(subr_list) >= 7:
                subr_list.sort(key=lambda e: e["x"])
                last_subr = None
                for subr in subr_list[:]:
                    if last_subr == None:
                        last_subr = subr
                        continue
                    if subr["x"] > last_subr["x"] and subr["x"] + subr["w"] < last_subr["x"] + last_subr["w"]:
                        subr_list.remove(subr)
                        last_subr = None
                        continue
                    last_subr = subr

                for subr in subr_list:
                    srx = subr["x"]
                    sry = subr["y"]
                    srw = subr["w"]
                    srh = subr["h"]
                    cv2.rectangle(img, (rx + srx, ry + sry), (rx + srx + srw, ry + sry + srh), color, 2)

            # self.show_img(sub)
            # img = cv2.drawContours(img,sub_contours,-1, color)
        self.show_img(img)

    def show_img(self, img):
        cv2.imshow("Plate detector", img)
        cv2.waitKey(0)
