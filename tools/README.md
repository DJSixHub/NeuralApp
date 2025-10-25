# Entrenamiento y conversión del modelo de segmentación (U-Net + MobileNetV2)

Este directorio contiene un script para reproducir la arquitectura del tutorial de TensorFlow (https://www.tensorflow.org/tutorials/images/segmentation), entrenar durante 4 epochs y convertir el modelo a TFLite usando cuantización post-training (full-integer, uint8).

Archivos añadidos:
- `train_convert.py` : script de entrenamiento y conversión.
- `../requirements.txt` : dependencias Python necesarias.

Instrucciones (PowerShell en Windows):

1) Crear y activar un entorno virtual (opcional pero recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3) Ejecutar el script (entrenamiento 4 epochs + conversión):

```powershell
python .\tools\train_convert.py
```

Salida esperada:
- `assets/segmentation_model_quant.tflite` : modelo TFLite cuantizado listo para integrar en la app Android.
- `tools/saved_model/segmentation_saved_model/` : SavedModel guardado.

Notas importantes:
- Instalar `tensorflow` puede requerir espacio en disco y tiempo. En CPU puede tardar mucho en entrenar; si tienes GPU y drivers adecuados, la instalación y entrenamiento será mucho más rápidos.
- Si la conversión a TFLite da errores por ops no soportadas en INT8, puede intentarse `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]` pero esto reduce la compatibilidad 100% local en todos los dispositivos.
- El modelo sigue la arquitectura exacta del tutorial (MobileNetV2 encoder + pix2pix upsample decoder). La cuantización se hace post-training con 100 imágenes de calibración del mismo dataset.
