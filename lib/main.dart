import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const MyApp());
}

Future<Interpreter?> _loadSegmentationInterpreter() async {
  try {
    return await Interpreter.fromAsset('assets/segmentation_model_quant.tflite');
  } catch (eAsset) {
    try {
      return await Interpreter.fromAsset('segmentation_model_quant.tflite');
    } catch (ePlain) {
      debugPrint('Failed to load interpreter: $eAsset ; $ePlain');
      return null;
    }
  }
}

List<int> _maskColorForClass(int classIndex, {int alpha = 255}) {
  switch (classIndex) {
    case 0:
      return <int>[128, 0, 128, alpha];
    case 1:
      return <int>[255, 255, 0, alpha];
    default:
      return <int>[255, 0, 0, alpha];
  }
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
  debugPrint('pickImage error: $e');
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
    final Interpreter? interpreter = await _loadSegmentationInterpreter();
    if (interpreter == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('No se pudo cargar el modelo para realizar la segmentación.')));
      }
      setState(() {
        _processing = false;
      });
      return;
    }

    // Prepare output buffer: shape [1,128,128,3]
    final output = List.generate(1, (_) => List.generate(128, (_) => List.generate(128, (_) => List.filled(3, 0))));

    // Step 4: run inference
    setState(() => _progress = 0.5);
    try {
      interpreter.run(input, output);
    } catch (e) {
  debugPrint('Interpreter run error: $e');
      interpreter.close();
      setState(() {
        _processing = false;
      });
      return;
    }

  debugPrint('Interpreter run completed — postprocessing...');

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
        final List<int> color = _maskColorForClass(maxIdx, alpha: 255);
        maskBuffer[base] = color[0];
        maskBuffer[base + 1] = color[1];
        maskBuffer[base + 2] = color[2];
        maskBuffer[base + 3] = color[3];
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

  Future<void> _openCameraSegmentation() async {
    if (!mounted) return;
    await Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const CameraSegmentationPage()));
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
        debugPrint('Imagen guardada en: $savedUri');
      }
      return savedUri;
    } on PlatformException catch (e) {
  debugPrint('Error al guardar imagen (PlatformException): ${e.message}');
      return null;
    } catch (e) {
  debugPrint('Error al guardar imagen: $e');
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
            Wrap(
              alignment: WrapAlignment.center,
              spacing: 12,
              runSpacing: 8,
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
                ElevatedButton.icon(
                  onPressed: _openCameraSegmentation,
                  icon: const Icon(Icons.videocam),
                  label: const Text('Usar cámara'),
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

class CameraSegmentationPage extends StatefulWidget {
  const CameraSegmentationPage({super.key});

  @override
  State<CameraSegmentationPage> createState() => _CameraSegmentationPageState();
}

class _CameraSegmentationPageState extends State<CameraSegmentationPage> {
  CameraController? _controller;
  Interpreter? _interpreter;
  bool _initializing = true;
  bool _processingFrame = false;
  Uint8List? _latestMaskBytes;
  int _frameCounter = 0;

  @override
  void initState() {
    super.initState();
    _initializePipeline();
  }

  @override
  void dispose() {
    _disposePipeline();
    super.dispose();
  }

  Future<void> _disposePipeline() async {
    try {
      if (_controller != null && _controller!.value.isStreamingImages) {
        await _controller!.stopImageStream();
      }
    } catch (_) {
      // Ignored: stopping the stream can fail if it was never started.
    }
    await _controller?.dispose();
    _controller = null;
    _interpreter?.close();
    _interpreter = null;
  }

  Future<void> _initializePipeline() async {
    final PermissionStatus camStatus = await Permission.camera.request();
    if (!camStatus.isGranted) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Se requiere permiso de cámara para el modo en vivo.')));
      Navigator.of(context).pop();
      return;
    }

    try {
      final List<CameraDescription> cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw Exception('No se encontraron cámaras disponibles.');
      }
      final CameraDescription selected = cameras.firstWhere(
        (CameraDescription cam) => cam.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final CameraController controller = CameraController(
        selected,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await controller.initialize();

      final Interpreter? interpreter = await _loadSegmentationInterpreter();
      if (interpreter == null) {
        await controller.dispose();
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('No se pudo cargar el modelo para la cámara.')));
        Navigator.of(context).pop();
        return;
      }

      _controller = controller;
      _interpreter = interpreter;

      if (mounted) {
        setState(() {
      _initializing = false;
        });
      }

      await controller.startImageStream(_processCameraImage);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('No se pudo iniciar la cámara: $e')));
      Navigator.of(context).pop();
    }
  }

  void _processCameraImage(CameraImage image) {
    if (!mounted || _interpreter == null) {
      return;
    }
    if (_processingFrame) {
      return;
    }
    _frameCounter = (_frameCounter + 1) % 4;
    if (_frameCounter != 0) {
      return;
    }
    _processingFrame = true;
    _handleFrame(image).whenComplete(() {
      _processingFrame = false;
    });
  }

  Future<void> _handleFrame(CameraImage image) async {
    final _MaskFrame? maskFrame = await _buildMaskForCameraImage(image);
    if (!mounted || maskFrame == null) {
      return;
    }
    setState(() {
      _latestMaskBytes = maskFrame.bytes;
    });
  }

  Future<_MaskFrame?> _buildMaskForCameraImage(CameraImage image) async {
    final Interpreter? interpreter = _interpreter;
    final CameraController? controller = _controller;
    if (interpreter == null || controller == null) {
      return null;
    }

    final img.Image? rgbImage = _convertYuv420ToImage(image);
    if (rgbImage == null) {
      return null;
    }

    final img.Image resized = img.copyResize(
      rgbImage,
      width: 128,
      height: 128,
      interpolation: img.Interpolation.average,
    );

    final List<List<List<List<int>>>> input =
        List.generate(1, (_) => List.generate(128, (_) => List.generate(128, (_) => List.filled(3, 0))));
    for (int y = 0; y < 128; y++) {
      for (int x = 0; x < 128; x++) {
  final img.Pixel pixel = resized.getPixel(x, y);
  input[0][y][x][0] = pixel.r.toInt();
  input[0][y][x][1] = pixel.g.toInt();
  input[0][y][x][2] = pixel.b.toInt();
      }
    }

    final List<List<List<List<int>>>> output =
        List.generate(1, (_) => List.generate(128, (_) => List.generate(128, (_) => List.filled(3, 0))));

    try {
      interpreter.run(input, output);
    } catch (e) {
      debugPrint('Interpreter run error (camera): $e');
      return null;
    }

    final img.Image mask = img.Image(width: 128, height: 128);
    for (int y = 0; y < 128; y++) {
      for (int x = 0; x < 128; x++) {
        final List<int> vals = output[0][y][x];
        int maxIdx = 0;
        int maxVal = vals[0];
        for (int c = 1; c < vals.length; c++) {
          if (vals[c] > maxVal) {
            maxVal = vals[c];
            maxIdx = c;
          }
        }
        final int alpha = maxIdx == 0 ? 160 : 220;
        final List<int> color = _maskColorForClass(maxIdx, alpha: alpha);
        mask.setPixelRgba(x, y, color[0], color[1], color[2], color[3]);
      }
    }

    img.Image expanded = img.copyResize(
      mask,
      width: rgbImage.width,
      height: rgbImage.height,
      interpolation: img.Interpolation.nearest,
    );

    expanded = _orientMaskForPreview(expanded, controller);

    final Uint8List bytes = Uint8List.fromList(img.encodePng(expanded));
    return _MaskFrame(bytes: bytes, width: expanded.width, height: expanded.height);
  }

  img.Image _orientMaskForPreview(img.Image mask, CameraController controller) {
    img.Image oriented = mask;
    final int rotation = controller.description.sensorOrientation % 360;
    if (rotation == 90) {
      oriented = img.copyRotate(oriented, angle: 90);
    } else if (rotation == 180) {
      oriented = img.copyRotate(oriented, angle: 180);
    } else if (rotation == 270) {
      oriented = img.copyRotate(oriented, angle: 270);
    }

    if (controller.description.lensDirection == CameraLensDirection.front) {
      oriented = img.flipHorizontal(oriented);
    }

    final Size? previewSize = controller.value.previewSize;
    if (previewSize != null) {
      final bool swap = rotation == 90 || rotation == 270;
      final int targetWidth = swap ? previewSize.height.round() : previewSize.width.round();
      final int targetHeight = swap ? previewSize.width.round() : previewSize.height.round();
      if (targetWidth > 0 && targetHeight > 0) {
        oriented = img.copyResize(
          oriented,
          width: targetWidth,
          height: targetHeight,
          interpolation: img.Interpolation.nearest,
        );
      }
    }

    return oriented;
  }

  img.Image? _convertYuv420ToImage(CameraImage image) {
    if (image.format.group != ImageFormatGroup.yuv420 || image.planes.length < 3) {
      return null;
    }

    final int width = image.width;
    final int height = image.height;
    final img.Image converted = img.Image(width: width, height: height);

    final Plane planeY = image.planes[0];
    final Plane planeU = image.planes[1];
    final Plane planeV = image.planes[2];

    final Uint8List bytesY = planeY.bytes;
    final Uint8List bytesU = planeU.bytes;
    final Uint8List bytesV = planeV.bytes;

    final int strideY = planeY.bytesPerRow;
    final int strideU = planeU.bytesPerRow;
    final int strideV = planeV.bytesPerRow;
    final int pixelStrideU = planeU.bytesPerPixel ?? 1;
    final int pixelStrideV = planeV.bytesPerPixel ?? 1;

    for (int y = 0; y < height; y++) {
      final int uvRow = (y >> 1);
      final int yRowOffset = y * strideY;
      final int uRowOffset = uvRow * strideU;
      final int vRowOffset = uvRow * strideV;
      for (int x = 0; x < width; x++) {
        final int yIndex = yRowOffset + x;
        final int uvColumn = (x >> 1);
        final int uIndex = uRowOffset + uvColumn * pixelStrideU;
        final int vIndex = vRowOffset + uvColumn * pixelStrideV;

        final int yValue = bytesY[yIndex];
        final int uValue = bytesU[uIndex];
        final int vValue = bytesV[vIndex];

        final int r = _clampToByte(yValue + 1.402 * (vValue - 128));
        final int g = _clampToByte(yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128));
        final int b = _clampToByte(yValue + 1.772 * (uValue - 128));

        converted.setPixelRgba(x, y, r, g, b, 255);
      }
    }

    return converted;
  }

  int _clampToByte(num value) {
    if (value < 0) {
      return 0;
    }
    if (value > 255) {
      return 255;
    }
    return value.round();
  }

  @override
  Widget build(BuildContext context) {
    final CameraController? controller = _controller;
    return Scaffold(
      appBar: AppBar(title: const Text('Segmentación en vivo')),
      body: _initializing
          ? const Center(child: CircularProgressIndicator())
          : (controller == null || _interpreter == null)
              ? const Center(child: Text('No se pudo inicializar la cámara'))
              : ColoredBox(
                  color: Colors.black,
                  child: Center(
                    child: AspectRatio(
                      aspectRatio: controller.value.aspectRatio,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          CameraPreview(controller),
                          if (_latestMaskBytes != null)
                            Image.memory(
                              _latestMaskBytes!,
                              fit: BoxFit.cover,
                              gaplessPlayback: true,
                            ),
                          Positioned(
                            left: 16,
                            bottom: 16,
                            child: Container(
                              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                              decoration: BoxDecoration(
                                color: Colors.black.withValues(alpha: 0.4),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  const Icon(Icons.pets, color: Colors.white),
                                  const SizedBox(width: 8),
                                  Text(
                                    _processingFrame ? 'Detectando…' : 'Listo',
                                    style: const TextStyle(color: Colors.white),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
    );
  }
}

class _MaskFrame {
  const _MaskFrame({required this.bytes, required this.width, required this.height});

  final Uint8List bytes;
  final int width;
  final int height;
}
