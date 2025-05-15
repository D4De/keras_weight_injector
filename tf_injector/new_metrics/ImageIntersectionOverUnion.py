from tf_injector.new_metrics.metric import Metric
import tensorflow as tf

def compute_IOU(out, label, numclass):
    
    # out : (batch, 520, 520, 21)
    #out = out[0]
    #out = tf.argmax(out, axis=-1)
    # out : (batch, 520, 520)
    # label is already in the right format (batch, 520, 520)

    ious = []
    for cls in range(numclass):
        clsx1 = out == cls
        clsx2 = label == cls
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

    #print("out shape: ", out.shape)
    #print("label shape: ", label.shape)
    #print("IOU shape: ", ious.shape)
    #print("----------------------")
    return ious

class ImageIntersectionOverUnion(Metric):
    def __init__(self, clean_scores: tf.Tensor, labels: tf.Tensor, num_classes: int = 21):
        '''
        Tensor dimension of the parameters:
        clean_scores : (2, batch, height, width, num_classes)
        labels : (batch, height, width)
        '''
        super().__init__(clean_scores, labels, num_classes)
        if clean_scores is not None:
            self.num_classes = clean_scores.shape[-1] 
            self.clean_labels = self.__evaluate_clean_labels(clean_scores) 


    def __evaluate_clean_labels(self, x : tf.Tensor) -> tf.Tensor:
        """
        clean_score : (batch, height, width, num_classes)
        clean_label : (batch, height, width)
        """
        x = tf.argmax(x, axis=-1)
        return x

    def clean_output(self):
        metric = compute_IOU(self.clean_labels, self.labels, self.num_classes) # (dataset, 21)
        mean = tf.experimental.numpy.nanmean(metric, axis=0) # (21,)
        tupl = tuple(mean.numpy().tolist())

        return tupl + (None,) * self.num_classes

    def faulty_output(self, faulty_scores, with_respect_to_labels=False):

        faulty_label = self.__evaluate_clean_labels(faulty_scores)
        output = list()

        # metric on (faulty_scores, labels)
        metric_on_labels = compute_IOU(
            out = faulty_label,
            label = self.labels,
            numclass= self.num_classes
        )
        metric_on_labels = tf.experimental.numpy.nanmean(metric_on_labels, axis=0) # (data, 21) -> (21,)
        metric_on_labels = metric_on_labels.numpy().tolist()
        output.extend(metric_on_labels)

        # metric on (faulty_scores, clean_labels)
        metric_on_clean_score = compute_IOU(
            out = faulty_label,
            label = self.clean_labels,
            numclass= self.num_classes
        )
        metric_on_clean_score = tf.experimental.numpy.nanmean(metric_on_clean_score, axis=0) # (data, 21) -> (21,)
        metric_on_clean_score = metric_on_clean_score.numpy().tolist()
        output.extend(metric_on_clean_score)

        return tuple(output)

    def get_header(self):
        
        label = (f"IOU_on_label_{i}" for i in range(self.num_classes))
        golden = (f"IOU_on_golden_{i}" for i in range(self.num_classes))
        return tuple(label) + tuple(golden)