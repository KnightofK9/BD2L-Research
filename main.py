import constant
# from darknetdetector import Detector
from videoextractor import VideoExtractor
from platedetector import PlateDetector

# video_extractor = VideoExtractor()
# video_extractor.extract_to(video_extractor.get_random_stream_url(), "./out/", 0.5, 10, True)

plate_detector = PlateDetector()
plate_detector.load_and_detect_img("./data/plate/001.JPG")

# detector = Detector(Constant.DARK_NET_CFG_PATH + "yolo.cfg",  "yolo.weights",
#                     Constant.DARK_NET_CFG_PATH + "coco.data")
# detector.detect_all_image_in_folder(Constant.IMAGE_DETECT_INPUT_PATH,
#                                     Constant.IMAGE_DETECT_OUTPUT_PATH)
