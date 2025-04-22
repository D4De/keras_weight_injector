from tf_injector.new_metrics.metric import Metric

def compute_IOU(x1, x2, numclass):
    # x1 = tf.argmax(x1, axis=-1, output_type=tf.dtypes.uint16)
    # x2 = tf.argmax(x2, axis=-1, output_type=tf.dtypes.uint16)
    ious = []
    for cls in range(numclass):
        clsx1 = x1 == cls
        clsx2 = x2 == cls
        inter = tf.reduce_sum(
            tf.cast(tf.logical_and(clsx1, clsx2), tf.uint64),
            axis=[1,2]
        )
        union = tf.reduce_sum( 
            tf.cast(tf.logical_or(clsx1, clsx2),tf.uint64),
            axis=[1,2]
        )
        iou = inter / union
        ious.append(iou)
    ious = tf.transpose(tf.stack(ious))
    return ious

class ImageIntersectionOverUnionMetric(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, clean_scores, clean_labels, labels):
        super().__init__(None, clean_labels, labels)
        return self
       
    def clean_metric(self):
        return compute_IOU(self.clean_scores, self.labels)
        
    def clean_output(self):
       return [1] * self.num_classes

    def faulty_output(self, faulty_scores, with_respect_to_labels=False):
        reference = self.clean_labels
        if with_respect_to_labels:
            reference = self.labels
        return compute_IOU(
            reference,
            faulty_scores,
            self.num_classes
        )
