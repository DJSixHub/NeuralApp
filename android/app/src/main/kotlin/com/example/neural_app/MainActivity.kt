package com.example.neural_app

import android.content.ContentValues
import android.media.MediaScannerConnection
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream

class MainActivity : FlutterActivity() {
	private val channelName = "com.example.neural_app/storage"

	override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
		super.configureFlutterEngine(flutterEngine)
		MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName).setMethodCallHandler { call, result ->
			when (call.method) {
				"saveToDownloads" -> {
					val bytes = call.argument<ByteArray>("bytes")
					val name = call.argument<String>("name") ?: "segmentation_result_${System.currentTimeMillis()}.png"
					if (bytes == null) {
						result.error("INVALID_INPUT", "bytes missing", null)
						return@setMethodCallHandler
					}
					try {
						val savedPath = saveToDownloads(bytes, name)
						if (savedPath != null) {
							result.success(savedPath)
						} else {
							result.error("SAVE_FAILED", "No path returned", null)
						}
					} catch (e: Exception) {
						result.error("SAVE_FAILED", e.localizedMessage, null)
					}
				}
				else -> result.notImplemented()
			}
		}
	}

	private fun saveToDownloads(bytes: ByteArray, fileName: String): String? {
		val resolver = applicationContext.contentResolver
		val mimeType = "image/png"

		return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
			val collection = MediaStore.Downloads.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
			val values = ContentValues().apply {
				put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
				put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
				put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
				put(MediaStore.MediaColumns.IS_PENDING, 1)
			}
			val uri = resolver.insert(collection, values) ?: return null
			try {
				resolver.openOutputStream(uri)?.use { stream ->
					stream.write(bytes)
					stream.flush()
				} ?: throw IllegalStateException("Unable to open output stream")
				values.clear()
				values.put(MediaStore.MediaColumns.IS_PENDING, 0)
				resolver.update(uri, values, null, null)
				uri.toString()
			} catch (e: Exception) {
				resolver.delete(uri, null, null)
				throw e
			}
		} else {
			@Suppress("DEPRECATION")
			val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
			if (!downloadsDir.exists()) {
				downloadsDir.mkdirs()
			}
			val outputFile = File(downloadsDir, fileName)
			FileOutputStream(outputFile).use { stream ->
				stream.write(bytes)
				stream.flush()
			}
			val values = ContentValues().apply {
				put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
				put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
				put(MediaStore.MediaColumns.DATA, outputFile.absolutePath)
			}
			resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
			MediaScannerConnection.scanFile(applicationContext, arrayOf(outputFile.absolutePath), arrayOf(mimeType), null)
			outputFile.absolutePath
		}
	}
}
