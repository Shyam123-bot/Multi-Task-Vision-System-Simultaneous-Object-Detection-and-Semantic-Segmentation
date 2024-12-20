from typing import Union
from numpy import ndarray
import tensorflow as tf

class DataEncoderDecoder:
    def __init__(
            self,
            num_classes: int,
            image_shape: tuple[int],
            xmin_boxes_default: ndarray[float] = None,
            ymin_boxes_default: ndarray[float] = None,
            xmax_boxes_default: ndarray[float] = None,
            ymax_boxes_default: ndarray[float] = None,
            center_x_boxes_default: ndarray[float] = None,
            center_y_boxes_default: ndarray[float] = None,
            width_boxes_default: ndarray[float] = None,
            height_boxes_default: ndarray[float] = None,
            iou_threshold: float = 0.5,
            standard_deviations_centroids_offsets: tuple[float] = (0.1, 0.1, 0.2, 0.2),
            augmentation_horizontal_flip: bool = False,
        ) -> None:
        """
        class for read and encode data, designed to work with tensorflow data pipelines
        you must pass default bounding boxes expressed in corners coordinates, or in centroids coordinates, or both
        (note for me -> probably not very efficient to store both coordinates! think about it in the future..)

        Args:
            num_classes (int): number of classes for object detection and segmentation problem, including background
            image_shape (tuple[int]): input image shape

            xmin_boxes_default (ndarray[float]): array of coordinates for xmin (corners coordinates)
            ymin_boxes_default (ndarray[float]): array of coordinates for ymin (corners coordinates)
            xmax_boxes_default (ndarray[float]): array of coordinates for xmax (corners coordinates)
            ymax_boxes_default (ndarray[float]): array of coordinates for ymax (corners coordinates)
            
            center_x_boxes_default (ndarray[float]): array of coordinates for center x (centroids coordinates)
            center_y_boxes_default (ndarray[float]): array of coordinates for center y (centroids coordinates)
            width_boxes_default (ndarray[float]): array of coordinates for width (centroids coordinates)
            height_boxes_default (ndarray[float]): array of coordinates for heigh (centroids coordinates)

            iou_threshold (float, optional): minimum intersection over union threshold with ground truth boxes to consider a default bounding box not background. Defaults to 0.5.
            standard_deviations_centroids_offsets (tuple[float], optional): standard deviations for offsets between ground truth and default bounding boxes, expected as (standard_deviation_center_x_offsets, standard_deviation_center_y_offsets, standard_deviation_width_offsets, standard_deviation_height_offsets). Defaults to (0.1, 0.1, 0.2, 0.2).

            augmentation_horizontal_flip (tuple[bool, float], optional): specify if horizontal flip data augmentation should be performed and the transformation probability. Defaults to (False, 0.5).
        """

        # set attributes
        self.num_classes = num_classes
        self.image_height, self.image_width = image_shape
        self.iou_threshold = iou_threshold
        self.standard_deviation_center_x_offsets, self.standard_deviation_center_y_offsets, self.standard_deviation_width_offsets, self.standard_deviation_height_offsets = standard_deviations_centroids_offsets

        # validation - only corners coordinates in input
        if all(centroids is None for centroids in (center_x_boxes_default, center_y_boxes_default, width_boxes_default, height_boxes_default)):
            if any(corners is None for corners in (xmin_boxes_default, ymin_boxes_default, xmax_boxes_default, ymax_boxes_default)):
                raise ValueError('you must pass all default bounding boxes corners coordinates!')
            else:
                # set corners coordinates
                self.xmin_boxes_default = tf.convert_to_tensor(xmin_boxes_default, dtype=tf.float32)
                self.ymin_boxes_default = tf.convert_to_tensor(ymin_boxes_default, dtype=tf.float32)
                self.xmax_boxes_default = tf.convert_to_tensor(xmax_boxes_default, dtype=tf.float32)
                self.ymax_boxes_default = tf.convert_to_tensor(ymax_boxes_default, dtype=tf.float32)

                # set centroids coordinates
                self.center_x_boxes_default, self.center_y_boxes_default, self.width_boxes_default, self.height_boxes_default = self._coordinates_corners_to_centroids(
                    xmin=self.xmin_boxes_default,
                    ymin=self.ymin_boxes_default,
                    xmax=self.xmax_boxes_default,
                    ymax=self.ymax_boxes_default
                )

        # validation - only centroids coordinates in input
        elif all(corners is None for corners in (xmin_boxes_default, ymin_boxes_default, xmax_boxes_default, ymax_boxes_default)):
            if any(centroids is None for centroids in (center_x_boxes_default, center_y_boxes_default, width_boxes_default, height_boxes_default)):
                raise ValueError('you must pass all default bounding boxes centroids coordinates!')
            else:
                # set centroids coordinates
                self.center_x_boxes_default = tf.convert_to_tensor(center_x_boxes_default, dtype=tf.float32)
                self.center_y_boxes_default = tf.convert_to_tensor(center_y_boxes_default, dtype=tf.float32)
                self.width_boxes_default = tf.convert_to_tensor(width_boxes_default, dtype=tf.float32)
                self.height_boxes_default = tf.convert_to_tensor(height_boxes_default, dtype=tf.float32)
                
                # set corners coordinates
                self.xmin_boxes_default, self.ymin_boxes_default, self.xmax_boxes_default, self.ymax_boxes_default = self._coordinates_centroids_to_corners(
                    center_x=self.center_x_boxes_default,
                    center_y=self.center_y_boxes_default,
                    width=self.width_boxes_default,
                    height=self.height_boxes_default
                )

        # validation - both corners and centroids coordinates in input
        elif (all(corners is None for corners in (xmin_boxes_default, ymin_boxes_default, xmax_boxes_default, ymax_boxes_default)) and
              all(centroids is None for centroids in (center_x_boxes_default, center_y_boxes_default, width_boxes_default, height_boxes_default))):
            # set corners coordinates
            self.xmin_boxes_default = tf.convert_to_tensor(xmin_boxes_default, dtype=tf.float32)
            self.ymin_boxes_default = tf.convert_to_tensor(ymin_boxes_default, dtype=tf.float32)
            self.xmax_boxes_default = tf.convert_to_tensor(xmax_boxes_default, dtype=tf.float32)
            self.ymax_boxes_default = tf.convert_to_tensor(ymax_boxes_default, dtype=tf.float32)
            
            # set centroids coordinates
            self.center_x_boxes_default = tf.convert_to_tensor(center_x_boxes_default, dtype=tf.float32)
            self.center_y_boxes_default = tf.convert_to_tensor(center_y_boxes_default, dtype=tf.float32)
            self.width_boxes_default = tf.convert_to_tensor(width_boxes_default, dtype=tf.float32)
            self.height_boxes_default = tf.convert_to_tensor(height_boxes_default, dtype=tf.float32)            
            
        # validation - some corners or centroids missing in input
        else:
            raise ValueError('you must pass all default bounding boxes centroids coordinates, or corners coordinates or both!') 

        # calculate area for default bounding boxes
        self.boxes_area_default = tf.expand_dims(
            input=(self.ymax_boxes_default - self.ymin_boxes_default + 1.0) * (self.xmax_boxes_default - self.xmin_boxes_default + 1.0),
            axis=1
        )

        # augmentation attributes
        self.augmentation_horizontal_flip = augmentation_horizontal_flip

    def _coordinates_corners_to_centroids(
            self,
            xmin: tf.Tensor,
            ymin: tf.Tensor,
            xmax: tf.Tensor,
            ymax: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        convert corners coordinates to centroids

        Args:
            xmin (tf.Tensor): xmin coordinates
            ymin (tf.Tensor): ymin coordinates
            xmax (tf.Tensor): xmax coordinates
            ymax (tf.Tensor): ymax coordinates

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: centroids coordinates (center_x, center_y, width, height)
        """

        # calculate centroids coordinates
        # note: pixels coordinates should be threated as image indexes, be careful with +-1 operations
        center_x = (xmax + xmin) / 2.0
        center_y = (ymax + ymin) / 2.0
        width = xmax - xmin + 1.0
        height = ymax - ymin + 1.0

        return center_x, center_y, width, height

    def _coordinates_centroids_to_corners(
            self,
            center_x: tf.Tensor,
            center_y: tf.Tensor,
            width: tf.Tensor,
            height: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        convert centroids coordinates to corners        

        Args:
            center_x (tf.Tensor): center_x coordinates
            center_y (tf.Tensor): center_y  coordinates
            width (tf.Tensor): width coordinates
            height (tf.Tensor): height coordinates

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: corners coordinates (xmin, ymin, xmax, ymax)
        """

        # calculate corners coordinates
        # note: pixels coordinates should be threated as image indexes, be careful with +-1 operations
        xmin = center_x - (width - 1.0) / 2.0
        ymin = center_y - (height - 1.0) / 2.0
        xmax = center_x + (width - 1.0) / 2.0
        ymax = center_y + (height - 1.0) / 2.0

        return xmin, ymin, xmax, ymax
    
    def _encode_ground_truth_labels_boxes(self, path_file_labels_boxes: str, augment_with_horizontal_flip: bool) -> tuple[tf.Tensor, tf.Tensor]:
        """
        encode ground truth data as required by a single-shot-detector network
        this means assign labels and calculate standardized offsets for each default bounding boxes

        Args:
            path_file_labels_boxes (str): path and filename for ground truth labels and boxes
            augment_with_horizontal_flip (bool): pass True if horizontal flip should be applied to input boxes, otherwise False

        Returns:
            tuple[tf.Tensor, tf.Tensor]:
                encoded data, a tuple with two tensor, respectively labels and offsets, with shape (total default boxes, num classes) and (total default boxes, 4)
                labels are one hot encoded
                offsets between ground truth and default bounding boxes are expressed as centroids offsets (center_x_offsets, center_y_offsets, width_offsets, height_offsets)
        """
        
        # read labels boxes csv file as text, split text by lines and then decode csv data to tensors
        labels_boxes = tf.strings.strip(tf.io.read_file(path_file_labels_boxes))
        labels_boxes = tf.strings.split(labels_boxes, sep='\r\n')
        labels_boxes = tf.io.decode_csv(labels_boxes, record_defaults=[int(), float(), float(), float(), float()])

        # create ground truth labels and boxes tensors
        labels_ground_truth, xmin_boxes_ground_truth, ymin_boxes_ground_truth, xmax_boxes_ground_truth, ymax_boxes_ground_truth = labels_boxes

        # augmentation - horizontal flip
        if augment_with_horizontal_flip:
            xmin_boxes_ground_truth, xmax_boxes_ground_truth = self.image_width - xmax_boxes_ground_truth, self.image_width - xmin_boxes_ground_truth

        # calculate area for ground truth bounding boxes
        boxes_area_ground_truth = (xmax_boxes_ground_truth - xmin_boxes_ground_truth + 1.0) * (ymax_boxes_ground_truth - ymin_boxes_ground_truth + 1.0)

        # coordinates of intersections between each default bounding box and all ground truth bounding boxes
        # it selects the maximum for xmin and ymin coordinates, the minimum for xmax and ymax coordinates
        xmin_boxes_intersection = tf.math.maximum(tf.expand_dims(self.xmin_boxes_default, axis=1), tf.transpose(xmin_boxes_ground_truth))
        ymin_boxes_intersection = tf.math.maximum(tf.expand_dims(self.ymin_boxes_default, axis=1), tf.transpose(ymin_boxes_ground_truth))
        xmax_boxes_intersection = tf.math.minimum(tf.expand_dims(self.xmax_boxes_default, axis=1), tf.transpose(xmax_boxes_ground_truth))
        ymax_boxes_intersection = tf.math.minimum(tf.expand_dims(self.ymax_boxes_default, axis=1), tf.transpose(ymax_boxes_ground_truth))

        # area of intersection between each default bounding box and all ground truth bounding boxes
        boxes_area_intersection = tf.math.maximum(0.0, xmax_boxes_intersection - xmin_boxes_intersection + 1.0) * tf.math.maximum(0.0, ymax_boxes_intersection - ymin_boxes_intersection + 1.0)

        # calculate intersection over union between each default bounding box and all ground truth bounding boxes
        # note that this is a matrix with shape (num default bounding boxes, num ground truth bounding boxes)
        iou = boxes_area_intersection / (self.boxes_area_default + tf.transpose(boxes_area_ground_truth) - boxes_area_intersection)

        # find best matches between ground truth and default bounding boxes with 3 steps, following original ssd paper suggestion
        # first one find a match for each ground truth box
        # second one find a match for each default box
        # third one put together results from previous steps, removing possible duplicates coming from the union operation

        # step 1 - find the best match between each ground truth box and all default bounding boxes
        # note that the output shape will be (num ground truth boxes with iou > 0 with at least one default box, 2)
        # this matrix-like tensor contains indexes for default boxes and ground truth boxes
        indexes_match_ground_truth = tf.stack([tf.math.argmax(iou, axis=0, output_type=tf.dtypes.int32), tf.range(len(xmin_boxes_ground_truth))], axis=1)
        indexes_match_ground_truth = tf.boolean_mask(tensor=indexes_match_ground_truth, mask=tf.math.greater(tf.math.reduce_max(iou, axis=0), 0.0), axis=0)

        # step 2 - find the best match between each default box and all ground truth bounding boxes
        # note that the output shape will be (num default truth boxes with iou > threshold with at least one ground truth box, 2)
        # this matrix-like tensor contains indexes for default boxes, ground truth boxes
        indexes_match_default = tf.stack([tf.range(len(self.xmin_boxes_default)), tf.math.argmax(iou, axis=1, output_type=tf.dtypes.int32)], axis=1)
        indexes_match_default = tf.boolean_mask(
            tensor=indexes_match_default,
            mask=tf.math.greater(tf.math.reduce_max(iou, axis=1), self.iou_threshold),
            axis=0
        )

        # step 3 - put all best matches together, removing possible duplicates
        indexes_match, _ = tf.raw_ops.UniqueV2(x=tf.concat([indexes_match_ground_truth, indexes_match_default], axis=0), axis=[0])

        # get the class labels for each best match and one-hot encode them (0 is reserved for the background class)
        labels_match = tf.gather(labels_ground_truth, indexes_match[:, 1])
        labels_match = tf.one_hot(labels_match, self.num_classes)

        # convert all bounding boxes coordinates from corners to centroids
        centroids_default_center_x, centroids_default_center_y, centroids_default_width, centroids_default_height = self._coordinates_corners_to_centroids(
            xmin=tf.gather(self.xmin_boxes_default, indexes_match[:, 0]),
            ymin=tf.gather(self.ymin_boxes_default, indexes_match[:, 0]),
            xmax=tf.gather(self.xmax_boxes_default, indexes_match[:, 0]),
            ymax=tf.gather(self.ymax_boxes_default, indexes_match[:, 0]),
        )
        centroids_ground_truth_center_x, centroids_ground_truth_center_y, centroids_ground_truth_width, centroids_ground_truth_height = self._coordinates_corners_to_centroids(
            xmin=tf.gather(xmin_boxes_ground_truth, indexes_match[:, 1]),
            ymin=tf.gather(ymin_boxes_ground_truth, indexes_match[:, 1]),
            xmax=tf.gather(xmax_boxes_ground_truth, indexes_match[:, 1]),
            ymax=tf.gather(ymax_boxes_ground_truth, indexes_match[:, 1]),
        )

        # calculate centroids offsets between ground truth and default boxes and standardize them
        # for standardization we are assuming that the mean zero and standard deviation given as input
        center_x_offsets = (centroids_ground_truth_center_x - centroids_default_center_x) / centroids_default_width / self.standard_deviation_center_x_offsets
        center_y_offsets = (centroids_ground_truth_center_y - centroids_default_center_y) / centroids_default_height / self.standard_deviation_center_y_offsets
        width_offsets = tf.math.log(centroids_ground_truth_width / centroids_default_width + 1.0) / self.standard_deviation_width_offsets
        height_offsets = tf.math.log(centroids_ground_truth_height / centroids_default_height + 1.0) / self.standard_deviation_height_offsets
        
        # ground truth data properly encoded
        # if a default bounding box was matched with ground truth, then proper labels and offsets centroids coordinates are assigned
        # otherwise background labels and zero offsets centroid coordinates are assigned
        ground_truth_encoded = tf.concat(
            values=[
                # all default bounding boxes are initially assigned to background class
                tf.ones(shape=(len(self.xmin_boxes_default), 1), dtype=tf.float32),
                # the remaining classes and offsets coordinates are equal to zero, becase all default bounding boxes are initially assigned to background class
                # note that the num_classes + 3 it's right, because we have previously created the class labels for background class
                tf.zeros(shape=(len(self.xmin_boxes_default), self.num_classes + 3), dtype=tf.float32)
            ],
            axis=1
        )        

        # update with proper values the indexes where a default bounding box matched a ground truth box
        ground_truth_encoded = tf.tensor_scatter_nd_update(
            tensor=ground_truth_encoded,
            indices=tf.expand_dims(indexes_match[:, 0], axis=1),
            updates=tf.concat(
                values=[
                    labels_match,
                    tf.expand_dims(center_x_offsets, axis=1),
                    tf.expand_dims(center_y_offsets, axis=1),
                    tf.expand_dims(width_offsets, axis=1),
                    tf.expand_dims(height_offsets, axis=1),
                ],
                axis=1)
        )

        return ground_truth_encoded[:, :-4], ground_truth_encoded[:, -4:]

    def read_and_encode(
            self,
            path_file_image: str,
            path_file_mask: str,
            path_file_labels_boxes: str,
        ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """
        read and encode ground truth data

        Args:
            path_file_image (str): path and filename for input image
            path_file_mask (str): path and filename for ground truth segmentation mask
            path_file_labels_boxes (str): path and filename for ground truth labels and boxes

        Returns:
            tuple[tf.Tensor, dict[str, tf.Tensor]]:
                a tuple containing data in form of (inputs, targets), as required by .fit method of a tensorflow keras Model class
                since i'm building a network with multiple outputs (mask for semantic segmentation, classification and regressione for object detection) i need multiple targets
                it's possible to return multiple targets using a dictionary
                keep in mind that the targets dictionary keys should match the output layers names in the network, in order to get the proper "y_true" data in the corresponding loss
        """

        # read the image
        image = tf.io.read_file(path_file_image)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, dtype=tf.float32)

        # read the segmentation mask, ignoring transparency channel in the png, one hot encode the classes, squeeze out unwanted dimension
        mask = tf.io.read_file(path_file_mask)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.one_hot(mask, depth=self.num_classes, dtype=tf.float32)
        mask = tf.squeeze(mask, axis=2)
        
        # horizontal random flip
        # this must determined in advance because the encoding process of the input bounding boxes it's in a separate method
        augment_with_horizontal_flip = True if self.augmentation_horizontal_flip and tf.random.uniform(shape=[], minval=0, maxval=1) >= 0.5 else False

        # augmentation - horizontal flip
        if augment_with_horizontal_flip:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        # encode ground truth labels and bounding boxes, applying horizontal flip if needed
        labels, boxes = self._encode_ground_truth_labels_boxes(path_file_labels_boxes=path_file_labels_boxes, augment_with_horizontal_flip=augment_with_horizontal_flip)

        return image, {'output-mask': mask, 'output-labels': labels, 'output-boxes': boxes}
    
    def decode_to_centroids(
            self,
            offsets_centroids: tf.Tensor,
            output_decoded_centroids_separately: bool = False
        ) -> Union[tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        decode standardized centroids offsets to centroids coordinates, where offsets means it's assumed to be equal to zero\n
        this method should used only to decode offsets for ground truth data encoded with this class\n
        do not use this for decode predictions from a network!

        Args:
            offsets_centroids (tf.Tensor): standardized offsets for centroids coordinates (center-x, center-y, width, height)
            output_decoded_centroids_separately (bool): if True will return a tuple with tensors (center-x, center-y, width, height), otherwise will return a single tensor with 4 values on last axis

        Returns:
            Union[tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]: decoded centroids coordinates
        """
 
        # split the offsets centroids coordinates
        center_x_offsets, center_y_offsets, width_offsets, height_offsets = [tf.squeeze(offsets_coordinates) for offsets_coordinates in tf.split(value=offsets_centroids, num_or_size_splits=4, axis=1)]

        # decode offsets to centroids coordinates
        center_x = center_x_offsets * self.standard_deviation_center_x_offsets * self.width_boxes_default + self.center_x_boxes_default
        center_y = center_y_offsets * self.standard_deviation_center_y_offsets * self.height_boxes_default + self.center_y_boxes_default
        width = (tf.math.exp(width_offsets * self.standard_deviation_width_offsets) - 1.0) * self.width_boxes_default
        height = (tf.math.exp(height_offsets * self.standard_deviation_height_offsets) - 1.0) * self.height_boxes_default

        # set to zero decoded coordinates for invalid boxes (default bounding boxes that were not matched to any ground truth)
        sum_of_coordinates_abs_value = tf.math.reduce_sum(tf.math.abs(offsets_centroids), axis=-1)
        not_background = tf.cast(tf.math.greater(sum_of_coordinates_abs_value, 0.0), dtype=tf.float32)
        center_x = center_x * not_background
        center_y = center_y * not_background
        width = width * not_background
        height = height * not_background

        # return data in desired format
        if output_decoded_centroids_separately:
            return center_x, center_y, width, height
        else:
            return tf.stack([center_x, center_y, width, height], axis=1)

    def decode_to_corners(
            self,
            offsets_centroids: tf.Tensor,
            output_decoded_corners_separately: bool = False
        ) -> Union[tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        decode standardized centroids offsets to corners coordinates, where offsets means it's assumed to be equal to zero\n
        this method should used only to decode offsets for ground truth data encoded with this class\n
        do not use this for decode predictions from a network!

        Args:
            offsets_centroids (tf.Tensor): standardized offsets for centroids coordinates (center-x, center-y, width, height)
            output_decoded_corners_separately (bool): if True will return a tuple with tensors (xmin, ymin, xmax, ymax), otherwise will return a single tensor with 4 values on last axis

        Returns:
            Union[tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]: decoded corners coordinates
        """        

        # decode offsets to centroids coordinates
        center_x, center_y, width, height = self.decode_to_centroids(offsets_centroids=offsets_centroids, output_decoded_centroids_separately=True)

        # convert to corners coordinates
        xmin, ymin, xmax, ymax = self._coordinates_centroids_to_corners(
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height
        )

        # set to zero decoded coordinates for invalid boxes (default bounding boxes that were not matched to any ground truth)
        centroids = tf.stack([center_x, center_y, width, height], axis=1)
        sum_of_coordinates_abs_value = tf.math.reduce_sum(tf.math.abs(centroids), axis=-1)
        not_background = tf.cast(tf.math.greater(sum_of_coordinates_abs_value, 0.0), dtype=tf.float32)
        xmin = xmin * not_background
        ymin = ymin * not_background
        xmax = xmax * not_background
        ymax = ymax * not_background

        # return data in desired format
        if output_decoded_corners_separately:
            return xmin, ymin, xmax, ymax
        else:
            return tf.stack([xmin, ymin, xmax, ymax], axis=1)

def augmentation_rgb_channels(image_batch: tf.Tensor, targets_batch: dict[str, tf.Tensor]) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    apply rgb channels augmentation to the image batch
    hue, saturation, contrast and brightness changes are randomly applied within a reasonable range
    transformed values are clipped between 0 and 1

    Args:
        image_batch (tf.Tensor): image batch tecnsor
        mask_batch (tf.Tensor): segmentation mask tensor
        labels_boxes_batch (tf.Tensor): encoded labels and bounding boxes tensor

    Returns:
        tuple[tf.Tensor, dict[str, tf.Tensor]]: 
        a tuple containing data in form of (inputs, targets)
        inputs it's the image_batch augmented with rgb channels transformations, targets it's the targets_batch unchanged
    """

    # small hue change
    image_batch = tf.image.random_hue(image_batch, max_delta=0.05)

    # small saturation change
    image_batch = tf.image.random_saturation(image_batch, lower=0.95, upper=1.05) 

    # small contrast change
    image_batch = tf.image.random_contrast(image_batch, lower=0.90, upper=1.10)

    # small brightness change
    image_batch = tf.image.random_brightness(image_batch, max_delta=0.10)

    # clip values out of range
    image_batch = tf.clip_by_value(image_batch, clip_value_min=0.0, clip_value_max=255.0)

    return image_batch, targets_batch

def read_image(path_file_image: str) -> tf.Tensor:
    """
    read an image given a path

    Args:
        path_file_image (str): path and filename for input image

    Returns:
        tf.Tensor: a tensor representing the input image
    """

    # read the image
    image = tf.io.read_file(path_file_image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)

    return image