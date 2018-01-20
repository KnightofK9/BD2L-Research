import constant
# from darknetdetector import Detector
from videoextractor import VideoExtractor

video_extractor = VideoExtractor()
video_extractor.extract_to(video_extractor.get_random_stream_url(), "./out/", 0.5, 10, True)



# detector = Detector(Constant.DARK_NET_CFG_PATH + "yolo.cfg",  "yolo.weights",
#                     Constant.DARK_NET_CFG_PATH + "coco.data")
# detector.detect_all_image_in_folder(Constant.IMAGE_DETECT_INPUT_PATH,
#                                     Constant.IMAGE_DETECT_OUTPUT_PATH)
