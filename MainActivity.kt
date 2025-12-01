package com.em.predictor

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.work.*
import kotlinx.coroutines.launch
import java.util.concurrent.TimeUnit

/**
 * Cliente Android MOCK para prototipo
 * Genera datos sintéticos y los envía al backend
 */
class MainActivity : ComponentActivity() {
    
    private lateinit var dataGenerator: SyntheticDataGenerator
    private lateinit var apiClient: EMPredictorApiClient
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize
        dataGenerator = SyntheticDataGenerator()
        apiClient = EMPredictorApiClient(
            baseUrl = "https://api.empredictor.dev",
            deviceToken = getDeviceToken()
        )
        
        setContent {
            EMPredictorTheme {
                MainScreen()
            }
        }
    }
    
    @Composable
    fun MainScreen() {
        var isSimulating by remember { mutableStateOf(false) }
        var lastSyncTime by remember { mutableStateOf("Never") }
        var dataPointCount by remember { mutableStateOf(0) }
        val scope = rememberCoroutineScope()
        
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("EM Predictor - Prototype") }
                )
            }
        ) { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            "Simulación de Datos",
                            style = MaterialTheme.typography.titleLarge
                        )
                        Text(
                            "Este cliente genera datos sintéticos para validar el backend",
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Divider(modifier = Modifier.padding(vertical = 8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Estado:")
                            Text(
                                if (isSimulating) "Activo" else "Pausado",
                                color = if (isSimulating) 
                                    MaterialTheme.colorScheme.primary 
                                else 
                                    MaterialTheme.colorScheme.error
                            )
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Última sincronización:")
                            Text(lastSyncTime)
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Puntos enviados:")
                            Text(dataPointCount.toString())
                        }
                    }
                }
                
                Button(
                    onClick = {
                        if (isSimulating) {
                            stopSimulation()
                            isSimulating = false
                        } else {
                            startSimulation()
                            isSimulating = true
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp)
                ) {
                    Text(if (isSimulating) "Detener Simulación" else "Iniciar Simulación")
                }
                
                OutlinedButton(
                    onClick = {
                        scope.launch {
                            val dataPoint = dataGenerator.generateDataPoint()
                            val success = apiClient.sendDataPoint(dataPoint)
                            if (success) {
                                dataPointCount++
                                lastSyncTime = java.time.LocalDateTime.now()
                                    .format(java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss"))
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Enviar Punto Manual")
                }
                
                Spacer(modifier = Modifier.weight(1f))
                
                Text(
                    "⚠️ Modo Prototipo - Datos Sintéticos",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
        }
    }
    
    private fun startSimulation() {
        val workRequest = PeriodicWorkRequestBuilder<DataGenerationWorker>(
            15, TimeUnit.MINUTES
        )
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.CONNECTED)
                    .build()
            )
            .build()
        
        WorkManager.getInstance(this).enqueueUniquePeriodicWork(
            "data_generation",
            ExistingPeriodicWorkPolicy.REPLACE,
            workRequest
        )
    }
    
    private fun stopSimulation() {
        WorkManager.getInstance(this).cancelUniqueWork("data_generation")
    }
    
    private fun getDeviceToken(): String {
        val prefs = getSharedPreferences("em_prefs", MODE_PRIVATE)
        var token = prefs.getString("device_token", null)
        if (token == null) {
            token = java.util.UUID.randomUUID().toString()
            prefs.edit().putString("device_token", token).apply()
        }
        return token
    }
}

/**
 * Generador de datos sintéticos realistas
 */
class SyntheticDataGenerator {
    
    private val random = kotlin.random.Random.Default
    
    data class SyntheticDataPoint(
        val userIdHash: String,
        val timestamp: String,
        val embedding: FloatArray,
        val numericFeatures: NumericFeatures
    )
    
    data class NumericFeatures(
        val sentimentScore: Float,
        val avgSentenceLen: Float,
        val typeTokenRatio: Float,
        val numMessages: Int,
        val responseLatency: Float?,
        val steps: Int?,
        val hrMean: Float?,
        val sleepHours: Float?
    )
    
    fun generateDataPoint(): SyntheticDataPoint {
        // Simula un usuario específico (en producción: user real)
        val userId = "simulated_user_001"
        val userIdHash = hashUserId(userId)
        
        // Genera embedding sintético (768 dims)
        val embedding = FloatArray(768) { random.nextFloat() * 0.1f - 0.05f }
        
        // Genera features con variación realista
        val baselineSentiment = -0.1f  // Ligeramente negativo (fatiga)
        val sentiment = baselineSentiment + (random.nextFloat() * 0.4f - 0.2f)
        
        val features = NumericFeatures(
            sentimentScore = sentiment.coerceIn(-1f, 1f),
            avgSentenceLen = 8f + random.nextFloat() * 8f,
            typeTokenRatio = 0.6f + random.nextFloat() * 0.2f,
            numMessages = random.nextInt(5, 20),
            responseLatency = if (random.nextFloat() > 0.3f) 
                120f + random.nextFloat() * 300f else null,
            steps = if (random.nextFloat() > 0.2f)
                3000 + random.nextInt(7000) else null,
            hrMean = if (random.nextFloat() > 0.2f)
                65f + random.nextFloat() * 25f else null,
            sleepHours = if (random.nextFloat() > 0.1f)
                5.5f + random.nextFloat() * 3f else null
        )
        
        return SyntheticDataPoint(
            userIdHash = userIdHash,
            timestamp = java.time.Instant.now().toString(),
            embedding = embedding,
            numericFeatures = features
        )
    }
    
    private fun hashUserId(userId: String): String {
        val bytes = java.security.MessageDigest.getInstance("SHA-256")
            .digest(userId.toByteArray())
        return bytes.joinToString("") { "%02x".format(it) }
    }
}

/**
 * Cliente API para enviar datos al backend
 */
class EMPredictorApiClient(
    private val baseUrl: String,
    private val deviceToken: String
) {
    suspend fun sendDataPoint(dataPoint: SyntheticDataGenerator.SyntheticDataPoint): Boolean {
        return try {
            // En producción: usar Retrofit/OkHttp
            // Aquí: mock success
            kotlinx.coroutines.delay(500)
            true
        } catch (e: Exception) {
            false
        }
    }
}

/**
 * Worker para generación periódica de datos
 */
class DataGenerationWorker(
    context: android.content.Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result {
        val generator = SyntheticDataGenerator()
        val apiClient = EMPredictorApiClient(
            baseUrl = "https://api.empredictor.dev",
            deviceToken = "device_token_placeholder"
        )
        
        val dataPoint = generator.generateDataPoint()
        val success = apiClient.sendDataPoint(dataPoint)
        
        return if (success) Result.success() else Result.retry()
    }
}

@Composable
fun EMPredictorTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(),
        content = content
    )
}
