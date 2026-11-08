"""TinyML-compatible model, loss, and metrics for UWB localisation."""

from __future__ import annotations

import itertools

import tensorflow as tf


ROOM_X_METERS = 4.8
ROOM_Y_METERS = 7.2
MAX_PEOPLE = 4
SLOT_PERMUTATIONS = tf.constant(
    list(itertools.permutations(range(MAX_PEOPLE))), dtype=tf.int32
)


@tf.keras.utils.register_keras_serializable(package="eeai")
class MultiPersonLoss(tf.keras.losses.Loss):
    """Binary confidence loss plus masked Smooth L1 coordinate loss."""

    def __init__(self, localization_weight: float = 5.0, delta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.localization_weight = float(localization_weight)
        self.delta = float(delta)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_confidence = y_true[..., 0]
        predicted_confidence = y_pred[..., 0]
        confidence_loss = tf.keras.backend.binary_crossentropy(
            true_confidence, predicted_confidence
        )

        coordinate_error = y_pred[..., 1:] - y_true[..., 1:]
        absolute_error = tf.abs(coordinate_error)
        smooth_l1 = tf.where(
            absolute_error < self.delta,
            0.5 * tf.square(absolute_error) / self.delta,
            absolute_error - 0.5 * self.delta,
        )
        smooth_l1 = tf.reduce_sum(smooth_l1, axis=-1)
        masked_localization = smooth_l1 * true_confidence
        people_per_frame = tf.reduce_sum(true_confidence, axis=-1)
        localization_loss = tf.math.divide_no_nan(
            tf.reduce_sum(masked_localization, axis=-1), people_per_frame
        )
        confidence_loss = tf.reduce_mean(confidence_loss, axis=-1)
        return confidence_loss + self.localization_weight * localization_loss

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "localization_weight": self.localization_weight,
                "delta": self.delta,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="eeai")
class PermutationInvariantMultiPersonLoss(tf.keras.losses.Loss):
    """Set-based confidence and localisation loss over all 4! assignments.

    The official metric treats localisations as an unordered set. This loss
    evaluates every permutation of the four target slots and backpropagates
    through the lowest-cost assignment for each frame.
    """

    def __init__(
        self,
        localization_weight: float = 8.0,
        confidence_weight: float = 1.0,
        count_weight: float = 0.0,
        delta: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.localization_weight = float(localization_weight)
        self.confidence_weight = float(confidence_weight)
        self.count_weight = float(count_weight)
        self.delta = float(delta)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        permuted_true = tf.gather(y_true, SLOT_PERMUTATIONS, axis=1)
        prediction = tf.expand_dims(y_pred, axis=1)

        true_confidence = permuted_true[..., 0]
        predicted_confidence = tf.clip_by_value(prediction[..., 0], 1e-6, 1.0 - 1e-6)
        confidence_loss = -(
            true_confidence * tf.math.log(predicted_confidence)
            + (1.0 - true_confidence) * tf.math.log(1.0 - predicted_confidence)
        )
        confidence_loss = tf.reduce_mean(confidence_loss, axis=-1)

        coordinate_error = prediction[..., 1:] - permuted_true[..., 1:]
        absolute_error = tf.abs(coordinate_error)
        smooth_l1 = tf.where(
            absolute_error < self.delta,
            0.5 * tf.square(absolute_error) / self.delta,
            absolute_error - 0.5 * self.delta,
        )
        smooth_l1 = tf.reduce_sum(smooth_l1, axis=-1) * true_confidence
        people_per_frame = tf.reduce_sum(true_confidence, axis=-1)
        localization_loss = tf.math.divide_no_nan(
            tf.reduce_sum(smooth_l1, axis=-1), people_per_frame
        )
        true_count = tf.reduce_sum(y_true[..., 0], axis=-1)
        predicted_count = tf.reduce_sum(y_pred[..., 0], axis=-1)
        count_error = tf.abs(predicted_count - true_count) / float(MAX_PEOPLE)
        assignment_cost = (
            self.confidence_weight * confidence_loss
            + self.localization_weight * localization_loss
            + self.count_weight * tf.expand_dims(count_error, axis=-1)
        )
        return tf.reduce_min(assignment_cost, axis=1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "localization_weight": self.localization_weight,
                "confidence_weight": self.confidence_weight,
                "count_weight": self.count_weight,
                "delta": self.delta,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="eeai")
class MetricAlignedPermutationLoss(tf.keras.losses.Loss):
    """Permutation-invariant objective expressed in physical metres.

    The official score accepts a Hungarian match only when its Euclidean
    distance is at most ``match_distance_m``.  Earlier objectives applied
    Smooth-L1 independently to normalised x/y coordinates, even though the
    room axes have different physical scales.  This objective performs both
    assignment and localisation optimisation in metres.  A small continuous
    tail term gives extra weight to errors beyond the official match radius.

    This class is used only during training.  It introduces no layer or
    operator into the exported inference graph.
    """

    def __init__(
        self,
        localization_weight: float = 2.0,
        confidence_weight: float = 1.0,
        huber_delta_m: float = 1.0,
        match_distance_m: float = 1.0,
        tail_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.localization_weight = float(localization_weight)
        self.confidence_weight = float(confidence_weight)
        self.huber_delta_m = float(huber_delta_m)
        self.match_distance_m = float(match_distance_m)
        self.tail_weight = float(tail_weight)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        permuted_true = tf.gather(y_true, SLOT_PERMUTATIONS, axis=1)
        prediction = tf.expand_dims(y_pred, axis=1)

        true_confidence = permuted_true[..., 0]
        predicted_confidence = tf.clip_by_value(prediction[..., 0], 1e-6, 1.0 - 1e-6)
        confidence_loss = -(
            true_confidence * tf.math.log(predicted_confidence)
            + (1.0 - true_confidence) * tf.math.log(1.0 - predicted_confidence)
        )
        confidence_loss = tf.reduce_mean(confidence_loss, axis=-1)

        room_scale = tf.constant([ROOM_X_METERS, ROOM_Y_METERS], dtype=tf.float32)
        coordinate_error_m = (prediction[..., 1:] - permuted_true[..., 1:]) * room_scale
        distance_m = tf.norm(coordinate_error_m, axis=-1)
        huber_distance = tf.where(
            distance_m < self.huber_delta_m,
            0.5 * tf.square(distance_m) / self.huber_delta_m,
            distance_m - 0.5 * self.huber_delta_m,
        )
        beyond_match_radius = tf.nn.relu(distance_m - self.match_distance_m)
        per_person_localization = (
            huber_distance + self.tail_weight * beyond_match_radius
        ) * true_confidence
        people_per_frame = tf.reduce_sum(true_confidence, axis=-1)
        localization_loss = tf.math.divide_no_nan(
            tf.reduce_sum(per_person_localization, axis=-1), people_per_frame
        )

        assignment_cost = (
            self.confidence_weight * confidence_loss
            + self.localization_weight * localization_loss
        )
        return tf.reduce_min(assignment_cost, axis=1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "localization_weight": self.localization_weight,
                "confidence_weight": self.confidence_weight,
                "huber_delta_m": self.huber_delta_m,
                "match_distance_m": self.match_distance_m,
                "tail_weight": self.tail_weight,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="eeai")
def masked_mean_distance_m(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean Euclidean error in metres on present ground-truth people."""
    mask = y_true[..., 0]
    scale = tf.constant([ROOM_X_METERS, ROOM_Y_METERS], dtype=tf.float32)
    distance = tf.norm((y_pred[..., 1:] - y_true[..., 1:]) * scale, axis=-1)
    return tf.math.divide_no_nan(tf.reduce_sum(distance * mask), tf.reduce_sum(mask))


@tf.keras.utils.register_keras_serializable(package="eeai")
def slot_confidence_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Slot-level presence accuracy at a diagnostic 0.5 threshold."""
    true_presence = y_true[..., 0] >= 0.5
    predicted_presence = y_pred[..., 0] >= 0.5
    return tf.reduce_mean(tf.cast(tf.equal(true_presence, predicted_presence), tf.float32))


@tf.keras.utils.register_keras_serializable(package="eeai")
def person_count_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Frame-level person-count accuracy at a diagnostic 0.5 threshold."""
    true_count = tf.reduce_sum(tf.cast(y_true[..., 0] >= 0.5, tf.int32), axis=-1)
    predicted_count = tf.reduce_sum(tf.cast(y_pred[..., 0] >= 0.5, tf.int32), axis=-1)
    return tf.reduce_mean(tf.cast(tf.equal(true_count, predicted_count), tf.float32))


def _separable_block(
    inputs: tf.Tensor,
    filters: int,
    block_index: int,
    pool_size: tuple[int, int] = (1, 2),
) -> tf.Tensor:
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3), padding="same", use_bias=False,
        name=f"depthwise_{block_index}",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"depthwise_bn_{block_index}")(x)
    x = tf.keras.layers.ReLU(name=f"depthwise_relu_{block_index}")(x)
    x = tf.keras.layers.Conv2D(
        filters, kernel_size=(1, 1), padding="same", use_bias=False,
        name=f"pointwise_{block_index}",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"pointwise_bn_{block_index}")(x)
    x = tf.keras.layers.ReLU(name=f"pointwise_relu_{block_index}")(x)
    return tf.keras.layers.AveragePooling2D(
        pool_size=pool_size, name=f"average_pool_{block_index}"
    )(x)


def build_baseline_model(
    input_shape: tuple[int, int, int] = (6, 3, 120),
    filters: tuple[int, ...] = (16, 24, 32),
    dense_units: int = 32,
    dropout_rate: float = 0.1,
    preserve_spatial_layout: bool = False,
    spatial_projection_filters: int | None = None,
    l2_regularization: float = 0.0,
) -> tf.keras.Model:
    """Build a small network using operations supported by TFLite Micro."""
    inputs = tf.keras.layers.Input(shape=input_shape, name="radar_input")
    # (radars, antennas, bins) -> (radars, bins, antenna channels)
    x = tf.keras.layers.Permute((1, 3, 2), name="orient_input")(inputs)
    x = tf.keras.layers.Conv2D(
        filters[0], kernel_size=(1, 7), strides=(1, 2), padding="same",
        use_bias=False, name="range_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="range_bn")(x)
    x = tf.keras.layers.ReLU(name="range_relu")(x)

    for block_index, block_filters in enumerate(filters, start=1):
        x = _separable_block(x, block_filters, block_index)

    if preserve_spatial_layout:
        # Pooling has already reduced the map substantially. Flattening here
        # preserves absolute radar/range-bin location needed for localisation.
        if spatial_projection_filters is not None:
            x = tf.keras.layers.Conv2D(
                spatial_projection_filters,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                name="spatial_projection",
            )(x)
            x = tf.keras.layers.BatchNormalization(name="spatial_projection_bn")(x)
            x = tf.keras.layers.ReLU(name="spatial_projection_relu")(x)
        x = tf.keras.layers.Flatten(name="spatial_flatten")(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    x = tf.keras.layers.Dense(
        dense_units,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
        name="bottleneck",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bottleneck_bn")(x)
    x = tf.keras.layers.ReLU(name="bottleneck_relu")(x)
    x = tf.keras.layers.Dense(MAX_PEOPLE * 3, activation="sigmoid", name="output_dense")(x)
    outputs = tf.keras.layers.Reshape((MAX_PEOPLE, 3), name="person_slots")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="uwb_tiny_baseline")
