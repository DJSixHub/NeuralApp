"""
Script para construir la U-Net del tutorial de TensorFlow (encoder MobileNetV2),
entrenar 5 epochs sobre Oxford-IIIT Pets (via TFDS) y convertir el modelo a TFLite
usando cuantización post-training (full integer / uint8).

Salida esperada:
 - <repo>/tools/saved_model/segmentation_saved_model/  (SavedModel)
 - <repo>/assets/segmentation_model_quant.tflite

Uso (PowerShell):
  python .\tools\train_convert.py

Nota: instalar dependencias en `requirements.txt` antes de ejecutar.
"""
from __future__ import annotations
import os
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds


TOOLS_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
SAVED_MODEL_DIR = TOOLS_DIR / "saved_model" / "segmentation_saved_model"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def upsample(filters, size):
    """Upsample block implemented with UpSampling2D + Conv2D to avoid TRANSPOSE_CONV in TFLite.
    Using Conv2DTranspose produces the builtin op TRANSPOSE_CONV which newer converters
    may emit with a version not present in some TFLite binaries. Replacing with
    resize+conv yields equivalent behavior while using supported ops.
    """
    result = tf.keras.Sequential([
        tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear'),
        tf.keras.layers.Conv2D(filters, size, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])
    return result


def unet_model(output_channels: int):
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),   # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=3, padding='same')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def representative_dataset_generator(ds, num_samples=100):
    """Yield samples for quantization calibration: returns batches of shape (1,128,128,3) in float32 [0,1]."""
    i = 0
    for image, _ in ds.unbatch().take(num_samples):
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]
        i += 1
        if i >= num_samples:
            break


def train_and_convert(epochs=5, batch_size=32):
    print('Cargando dataset Oxford-IIIT Pets via TFDS...')
    # current TFDS provides version 4.0.0; use a wildcard to accept 4.x
    dataset, info = tfds.load('oxford_iiit_pet:4.*.*', with_info=True)

    train = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = batch_size
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = max(1, TRAIN_LENGTH // BATCH_SIZE)

    train_batches = (
        train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    test_batches = test.batch(BATCH_SIZE)

    OUTPUT_CLASSES = 3

    out_tflite = ASSETS_DIR / 'segmentation_model_quant.tflite'
    # If tflite already exists, skip everything
    if out_tflite.exists():
        print(f'{out_tflite} ya existe — se omite entrenamiento/conversión.')
        return

    # If a SavedModel already exists, try to load it and skip training
    saved_model_pb = SAVED_MODEL_DIR / 'saved_model.pb'
    if saved_model_pb.exists():
        print('SavedModel encontrado — cargando para conversión...')
        try:
            model = tf.keras.models.load_model(str(SAVED_MODEL_DIR))
        except Exception:
            # fallback to loading via tf.saved_model.load and wrapping if needed
            model = tf.saved_model.load(str(SAVED_MODEL_DIR))
    else:
        model = unet_model(output_channels=OUTPUT_CLASSES)
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        print(f'Training {epochs} epochs, steps_per_epoch={STEPS_PER_EPOCH}...')
        model.fit(train_batches, epochs=epochs, steps_per_epoch=STEPS_PER_EPOCH, validation_data=test_batches, validation_steps=max(1, info.splits['test'].num_examples//BATCH_SIZE//5))

        print('Guardando modelo Keras (.h5) para referencia...')
        # Save a Keras HDF5 .h5 model (more compatible across TF versions)
        try:
            h5_path = str(SAVED_MODEL_DIR.with_suffix('.h5'))
            model.save(h5_path)
            print(f'Modelo guardado en {h5_path}')
        except Exception as e:
            print('No se pudo guardar .h5 del modelo:', e)

    print('Convirtiendo a TFLite con cuantización (full integer, uint8)...')
    # Prefer converter.from_keras_model when we have an in-memory Keras model
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    except Exception:
        # fallback to saved_model path when keras conversion not available
        converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Use a small representative dataset for calibration
    sample_ds = train.batch(1)
    converter.representative_dataset = lambda: representative_dataset_generator(sample_ds, num_samples=100)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print('Error during TFLite conversion:', e)
        raise

    out_path = ASSETS_DIR / 'segmentation_model_quant.tflite'
    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    print(f'Escrito {out_path}')


if __name__ == '__main__':
    train_and_convert(epochs=5, batch_size=32)
