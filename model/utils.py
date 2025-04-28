import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute focal loss for binary classification.

    Args:
        y_true: Ground truth labels, shape of [batch_size, num_classes].
        y_pred: Predicted probabilities, shape of [batch_size, num_classes].
        gamma: Focusing parameter to reduce the relative loss for well-classified examples.
        alpha: Balancing parameter to balance positive and negative examples.

    Returns:
        Tensor representing the focal loss.
    """
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Clip predictions to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Compute cross-entropy loss
    cross_entropy = -y_true * tf.math.log(y_pred)

    # Compute focal loss
    weight = alpha * tf.math.pow(1 - y_pred, gamma)
    focal_loss = weight * cross_entropy

    return tf.reduce_sum(focal_loss, axis=-1)