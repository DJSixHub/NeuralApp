import 'dart:io';
import 'dart:typed_data';
import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Segmentation Demo',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  double _progress = 0.0;
  bool _processing = false;
  String? _savedPath;
  static const MethodChannel _storageChannel = MethodChannel('com.example.neural_app/storage');

  Future<void> _pickImage() async {
    // Try to pick image from gallery. Let the plugin handle permissions.
    try {
      final XFile? picked = await _picker.pickImage(source: ImageSource.gallery);
      if (!mounted) return;
      if (picked == null) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('No se seleccionó ninguna imagen')));
        return;
      }
      setState(() {
        _imageFile = File(picked.path);
        _savedPath = null;
      });
    } catch (e) {
      // Show a visible error so user knows something failed instead of failing silently
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error al abrir la galería: $e')));
      }
      print('pickImage error: $e');
    }
  }

  Future<ui.Image> _decodeImageFromListResize(Uint8List data, {int? targetWidth, int? targetHeight}) async {
    final codec = await ui.instantiateImageCodec(data, targetWidth: targetWidth, targetHeight: targetHeight);
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  Future<void> _runSegmentation() async {
    if (_imageFile == null) return;
    setState(() {
      _processing = true;
      _progress = 0.0;
    });

    // Step 1: load image bytes
    setState(() => _progress = 0.05);
    final bytes = await _imageFile!.readAsBytes();

    // Decode original to get size
    final ui.Image original = await _decodeImageFromListResize(bytes);
    final int origW = original.width;
    final int origH = original.height;

    // Step 2: resize to 128x128 (model input)
    setState(() => _progress = 0.15);
    final ui.Image img128 = await _decodeImageFromListResize(bytes, targetWidth: 128, targetHeight: 128);

    // Extract raw RGBA bytes from resized image
    final ByteData? byteData = await img128.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) {
      setState(() {
        _processing = false;
      });
      return;
    }
    final Uint8List rgba = byteData.buffer.asUint8List();

    // Build input tensor as nested List: [1][128][128][3]
    setState(() => _progress = 0.25);
    final input = List.generate(1, (_) => List.generate(128, (_) => List.generate(128, (_) => List.filled(3, 0))));
    for (int y = 0; y < 128; y++) {
      for (int x = 0; x < 128; x++) {
        final int idx = (y * 128 + x) * 4;
        final int r = rgba[idx];
        final int g = rgba[idx + 1];
        final int b = rgba[idx + 2];
        input[0][y][x][0] = r;
        input[0][y][x][1] = g;
        input[0][y][x][2] = b;
      }
    }

    // Step 3: load interpreter
    setState(() => _progress = 0.35);
    Interpreter interpreter;
    try {
      // First try the full asset path as declared in pubspec.yaml
      interpreter = await Interpreter.fromAsset('assets/segmentation_model_quant.tflite');
      print('Interpreter loaded: assets/segmentation_model_quant.tflite');
    } catch (e1) {
      print('Interpreter.fromAsset("assets/...") failed: $e1');
      // Fallback: some projects register assets without the leading folder
      try {
        interpreter = await Interpreter.fromAsset('segmentation_model_quant.tflite');
        print('Interpreter loaded: segmentation_model_quant.tflite');
      } catch (e2) {
        print('Failed to load interpreter with both names: $e1 ; $e2');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('No se pudo cargar el modelo: $e2')));
        }
        setState(() {
          _processing = false;
        });
        return;
      }
    }

    // Prepare output buffer: shape [1,128,128,3]
    final output = List.generate(1, (_) => List.generate(128, (_) => List.generate(128, (_) => List.filled(3, 0))));

    // Step 4: run inference
    setState(() => _progress = 0.5);
    try {
      interpreter.run(input, output);
    } catch (e) {
      print('Interpreter run error: $e');
      interpreter.close();
      setState(() {
        _processing = false;
      });
      return;
    }

    print('Interpreter run completed — postprocessing...');

    setState(() => _progress = 0.8);

    // Step 5: postprocess - create mask RGBA buffer with alpha for overlay
    final int maskW = 128;
    final int maskH = 128;
    final Uint8List maskBuffer = Uint8List(maskW * maskH * 4);
    for (int y = 0; y < maskH; y++) {
      for (int x = 0; x < maskW; x++) {
        final List<int> vals = output[0][y][x];
        int maxIdx = 0;
        int maxVal = vals[0];
        for (int c = 1; c < vals.length; c++) {
          if (vals[c] > maxVal) {
            maxVal = vals[c];
            maxIdx = c;
          }
        }
        final int base = (y * maskW + x) * 4;
        if (maxIdx == 0) {
          // outer background rendered purple
          maskBuffer[base] = 128;
          maskBuffer[base + 1] = 0;
          maskBuffer[base + 2] = 128;
          maskBuffer[base + 3] = 255;
        } else if (maxIdx == 1) {
          // border rendered solid yellow
          maskBuffer[base] = 255;
          maskBuffer[base + 1] = 255;
          maskBuffer[base + 2] = 0;
          maskBuffer[base + 3] = 255;
        } else {
          // inner region rendered solid red
          maskBuffer[base] = 255;
          maskBuffer[base + 1] = 0;
          maskBuffer[base + 2] = 0;
          maskBuffer[base + 3] = 255;
        }
      }
    }

    // Decode mask buffer into ui.Image
    final Completer<ui.Image> maskCompleter = Completer();
    ui.decodeImageFromPixels(maskBuffer, maskW, maskH, ui.PixelFormat.rgba8888, (ui.Image img) {
      maskCompleter.complete(img);
    });
    final ui.Image maskImg = await maskCompleter.future;

    // Step 6: compose original image and scaled mask onto a canvas and save
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder, Rect.fromLTWH(0, 0, origW.toDouble(), origH.toDouble()));
    // draw original
    final srcOrig = Rect.fromLTWH(0, 0, original.width.toDouble(), original.height.toDouble());
    final dstOrig = Rect.fromLTWH(0, 0, origW.toDouble(), origH.toDouble());
    final paint = Paint();
    canvas.drawImageRect(original, srcOrig, dstOrig, paint);
    // draw mask scaled to original size
    final srcMask = Rect.fromLTWH(0, 0, maskImg.width.toDouble(), maskImg.height.toDouble());
    final dstMask = Rect.fromLTWH(0, 0, origW.toDouble(), origH.toDouble());
    canvas.drawImageRect(maskImg, srcMask, dstMask, Paint());
    final picture = recorder.endRecording();
    final ui.Image composed = await picture.toImage(origW, origH);
    final ByteData? pngBytes = await composed.toByteData(format: ui.ImageByteFormat.png);
    if (pngBytes == null) {
      interpreter.close();
      setState(() {
        _processing = false;
      });
      return;
    }
    final outBytes = pngBytes.buffer.asUint8List();

    // Step 7: save result to the Downloads directory
    setState(() => _progress = 0.9);

  // Request the appropriate permission before writing to public storage.
    final bool canSave = await _ensureSavePermission();
    if (!canSave) {
      interpreter.close();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No se pudo guardar el resultado (permisos faltantes).')),
        );
      }
      setState(() {
        _processing = false;
      });
      return;
    }

    final String timestamp = DateTime.now().millisecondsSinceEpoch.toString();
    final String fileName = 'segmentation_result_$timestamp.png';
    String? savedPath = await _saveResultToDownloads(outBytes, fileName);
    if (savedPath == null) {
      interpreter.close();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No se pudo guardar el resultado en la carpeta Descargas.')),
        );
      }
      setState(() {
        _processing = false;
      });
      return;
    }
    if (savedPath.isEmpty) {
      savedPath = 'Carpeta Descargas / Galería';
    }

    interpreter.close();

    setState(() {
      _processing = false;
      _progress = 1.0;
      _savedPath = savedPath;
    });
  }

  Future<bool> _ensureSavePermission() async {
    if (Platform.isIOS) {
      final status = await Permission.photos.request();
      if (status.isGranted || status.isLimited) {
        return true;
      }
      if (status.isPermanentlyDenied) {
        await openAppSettings();
      }
      return false;
    }

    if (Platform.isAndroid) {
      final storageStatus = await Permission.storage.request();
      if (storageStatus.isGranted || storageStatus.isLimited) {
        return true;
      }
      if (storageStatus.isPermanentlyDenied) {
        await openAppSettings();
        return false;
      }
      // On Android 10+ scoped storage writes to Downloads without extra permission.
      return true;
    }

    return true;
  }

  Future<String?> _saveResultToDownloads(Uint8List bytes, String filename) async {
    try {
      final String? savedUri = await _storageChannel.invokeMethod<String>(
        'saveToDownloads',
        {
          'bytes': bytes,
          'name': filename,
        },
      );
      if (savedUri != null) {
        print('Imagen guardada en: $savedUri');
      }
      return savedUri;
    } on PlatformException catch (e) {
      print('Error al guardar imagen (PlatformException): ${e.message}');
      return null;
    } catch (e) {
      print('Error al guardar imagen: $e');
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Segmentación (local)')),
      body: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: _imageFile == null
                    ? const Text('Selecciona una imagen para empezar')
                    : Image.file(_imageFile!),
              ),
            ),
            if (_processing) ...[
              LinearProgressIndicator(value: _progress),
              const SizedBox(height: 8),
              Text('Procesando... ${(_progress * 100).toStringAsFixed(0)}%'),
            ] else if (_savedPath != null) ...[
              const SizedBox(height: 8),
              Text('Resultado guardado: $_savedPath'),
            ],
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _pickImage,
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Cargar foto'),
                ),
                ElevatedButton.icon(
                  onPressed: (_imageFile != null && !_processing) ? _runSegmentation : null,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Empezar'),
                ),
              ],
            ),
            const SizedBox(height: 12),
          ],
        ),
      ),
    );
  }
}
