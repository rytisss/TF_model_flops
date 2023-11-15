import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Conv2D,
    LeakyReLU, Activation, UpSampling2D, BatchNormalization, AveragePooling2D, concatenate)
from tensorflow.keras.optimizers import Adam


def get_flops(model, model_inputs) -> float:
    """
    Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
    in inference mode. It uses tf.compat.v1.profiler under the hood.
    """
    # if not hasattr(model, "model"):
    #     raise wandb.Error("self.model must be set before using this method.")

    if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
    ):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # Compute FLOPs for one sample
    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        for inp in model_inputs
    ]

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    # convert to MFLOPs
    return (flops.total_float_ops / 1e6) / 2


def effiecientNetV2B0Classifier(input_size=(128, 128, 3), last_stage_out=False, weights_path=None):
    inputs = tf.keras.Input(shape=input_size)
    x = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False,
                                                    weights='imagenet',
                                                    input_tensor=inputs)
    if last_stage_out:
        x = x.get_layer('block6a_expand_activation').output
    else:
        x = x.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, x)
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(),
                  optimizer=Adam(learning_rate=1e-3),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                           tf.keras.metrics.Precision(thresholds=0.5),
                           tf.keras.metrics.Recall(thresholds=0.5)])
    if weights_path:
        model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    image_model = effiecientNetV2B0Classifier(last_stage_out=True)

    x = tf.constant(np.random.randn(1, 128, 128, 3))

    print(f'{get_flops(image_model, [x])} MFlops')
