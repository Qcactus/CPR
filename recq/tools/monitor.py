from time import time


class Timer(object):
    def start(self, msg):
        self.msg = msg
        self.start_time = time()

    def stop(self, msg_stop=""):
        self.stop_time = time()
        print(
            self.msg,
            ":",
            "{:10.5f}".format(self.stop_time - self.start_time),
            "s" + ((" | " + msg_stop) if msg_stop else ""),
        )
