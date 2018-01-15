import urllib2
import random


class Extractor:
    avail_list = []

    def __init__(self):
        self.init_default_url()

    def init_default_url(self):
        ids = [8, 9, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 48, 49, 52, 53, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68,
               69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 92, 93, 96]
        for i in ids:
            self.avail_list.append(Extractor._get_url(i))

    def check_all_avail_stream_file(self):
        min_id = 1
        max_id = 100
        print "Checking all avail stream url in range {}-{}".format(min_id, max_id)
        self.avail_list = []
        for i in range(min_id,max_id):
            url = Extractor._get_url(i)
            if Extractor._check_url_exists(url):
                self.avail_list.append(url)
            print "."

    def print_all_avail_stream_file(self):
        print '\n'.join(str(p) for p in self.avail_list)

    def get_random_avail_stream_file(self):
        return random.choice(self.avail_list)

    @staticmethod
    def _get_url(number):
        return "http://4co2.vp9.tv/chn/DNG{}/v.m3u8".format(str(number))

    @staticmethod
    def _check_url_exists(location):
        request = urllib2.Request(location)
        request.get_method = lambda: 'HEAD'
        try:
            response = urllib2.urlopen(request)
            return True
        except urllib2.HTTPError:
            return False
