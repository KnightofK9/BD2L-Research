import cv2
import os
import shutil

from Extractor import Extractor


class VideoExtractor:

    def __init__(self):
        self.extractor = Extractor()

    def get_random_stream_url(self):
        return self.extractor.get_random_avail_stream_file()

    def reload_alive_stream_url(self):
        self.extractor.check_all_avail_stream_file()

    def extract_to(self, input_path, output_path, sampling_rate=0.5, max_capture_frame=0,
                   clear_input_folder_if_exists=False):
        if clear_input_folder_if_exists and os.path.isdir(output_path):
            print "Clearing output folder {}".format(output_path)
            shutil.rmtree(output_path)
            os.mkdir(output_path)
        skip_sec = 1 / sampling_rate
        video = cv2.VideoCapture(input_path)
        success, image = video.read()
        count = 0
        fps = 0
        (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
        if int(major_ver) < 3:
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
            print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)
            print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
        frame_rate = 1.0 / fps
        current_time = 0
        count_reading = 0
        print "Start capturing at url {}".format(input_path)
        print "FPS: {} ".format(fps)
        print "Output: {}".format(output_path)
        print "Sampling rate: {}".format(sampling_rate)
        while success:
            success, image = video.read()
            count_reading += 1
            current_time += frame_rate
            if current_time >= skip_sec:
                current_time = 0
                number_name = "{0:05d}".format(count)
                output_url = output_path + "frame_" + number_name + ".jpg";
                cv2.imwrite(output_url, image)
                print "Writing {}".format(output_url)
                count += 1
                if 0 < max_capture_frame <= count:
                    break
        print "Task completed, total {} files".format(count)
