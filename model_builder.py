import tensorflow as tf
import efficientnet as efn
import efficientnet.tfkeras

def build_model():
    base_model = efn.tfkeras.EfficientNetB0(input_shape=(224, 224, 3),
                                            include_top=False)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.2, name='top_dropout')(x)
    x = tf.keras.layers.Dense(10,
                     activation='softmax',
                     kernel_initializer=efn.model.DENSE_KERNEL_INITIALIZER,
                     name='probs')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x,
                                  name=base_model.name)

    return model
