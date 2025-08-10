package com.example.imageclassificationqat

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.util.concurrent.Executors
import androidx.core.graphics.createBitmap
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var viewFinder: PreviewView
    private lateinit var classificationResultText: TextView
    private var mBitmap: Bitmap? = null
    private var mModule: Module? = null

    fun rotateBitmap(original: Bitmap, degrees: Double): Bitmap? {
        val matrix: Matrix = Matrix()
        matrix.preRotate(degrees.toFloat())
        val rotatedBitmap: Bitmap? = Bitmap.createBitmap(
            original,
            0,
            0,
            original.getWidth(),
            original.getHeight(),
            matrix,
            true
        )
        return rotatedBitmap
    }
    @OptIn(ExperimentalGetImage::class)
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.surfaceProvider = viewFinder.surfaceProvider
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // Efficient format
                .build()
                .also {
                    it.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                        //analyze image
                        mModule?.destroy()
                        mModule = Module.load("/data/local/tmp/QAT_Model_Actual.pte")
                        val image: Image = imageProxy.image as Image
                        val buffer: ByteBuffer = imageProxy.planes[0].buffer
                        // Create an ARGB_8888 Bitmap
                        val bitmap : Bitmap = createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
                        buffer.rewind() // Rewind buffer to start
                        bitmap.copyPixelsFromBuffer(buffer)
                        mBitmap = bitmap
                        mBitmap = rotateBitmap(mBitmap!!.scale(320, 320), imageProxy.imageInfo.rotationDegrees.toDouble())

                        val inputTensor: Tensor =
                            TensorImageUtils.bitmapToFloat32Tensor(
                                mBitmap,
                                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                                TensorImageUtils.TORCHVISION_NORM_STD_RGB)

                        val outputTensor: Tensor = mModule!!.forward(EValue.from(inputTensor))[0].toTensor()
                        val scores = outputTensor.dataAsFloatArray
                        var maxScore = -Float.MAX_VALUE
                        var maxScoreIndex = -1
                        var i = 0
                        for (score in scores)
                        {
                            if(score > maxScore){
                                maxScore = score
                                maxScoreIndex = i
                            }
                            i++
                        }

                        Log.i("Detection: ", "Detected - " + class_names[maxScoreIndex])
                        //output text
                        runOnUiThread { classificationResultText.text = class_names[maxScoreIndex] }
                        imageProxy.close() // IMPORTANT: Close the imageProxy
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("Camera", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        viewFinder = findViewById(R.id.viewFinder)
        classificationResultText = findViewById(R.id.classificationResultText)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        mModule = Module.load("/data/local/tmp/QAT_Model.pte")
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }
    }
    private fun requestPermissions() {
        ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, RequestCodePermission)
    }

    private fun allPermissionsGranted(): Boolean {
        for (permission in REQUIRED_PERMISSIONS){
            if(ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED){
                return false
            }
        }
        return true
    }
    private var RequestCodePermission = 101
    private val REQUIRED_PERMISSIONS = arrayOf("android.permission.CAMERA")
    private val class_names = arrayOf(
        "komondor",
        "German_shepherd",
        "toy_poodle",
        "pug",
        "Yorkshire_terrier",
        "Doberman",
        "Bernese_mountain_dog",
        "French_bulldog",
        "chow",
        "Chihuahua",
        "Eskimo_dog",
    )
}

