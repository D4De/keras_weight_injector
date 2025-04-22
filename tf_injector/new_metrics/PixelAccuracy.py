from tf_injector.new_metrics.metric import Metric

class PixelAccuracy(Metric):

    def __init__(self):
        super().__init__()

    def chiSono(self):
        print("PixelAccuracy")