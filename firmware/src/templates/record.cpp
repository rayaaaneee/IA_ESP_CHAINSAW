#include <Arduino.h>
#include <driver/i2s_std.h>

// --- Configuration I2S (Pins pour l'ESP32) ---
#define I2S_WS  4    // LRCL
#define I2S_SCK 5    // BCLK
#define I2S_SD  6    // DOUT
#define SOUND_SAMPLE_RATE 8000

// Configuration des buffers
const int BUF_SAMPLES = 4092; 
const int GAIN_SHIFT  = 0;

int32_t rawBuffer[BUF_SAMPLES];
int32_t s24[BUF_SAMPLES];
int16_t outBuffer[BUF_SAMPLES];

i2s_chan_handle_t rx_handle;
int32_t prevSample = 0;

// ---------------- Filtre médian ----------------
int32_t med3(int32_t a, int32_t b, int32_t c) {
  int32_t t;
  if (a > b) { t = a; a = b; b = t; }
  if (b > c) { t = b; b = c; c = t; }
  if (a > b) { t = a; a = b; b = t; }
  return b;
}

// ---------------- Initialisation I2S ----------------
void i2s_init() {
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  i2s_new_channel(&chan_cfg, NULL, &rx_handle);

  i2s_std_config_t std_cfg = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SOUND_SAMPLE_RATE),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
    .gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = (gpio_num_t)I2S_SCK,
      .ws   = (gpio_num_t)I2S_WS,
      .dout = I2S_GPIO_UNUSED,
      .din  = (gpio_num_t)I2S_SD,
      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
    },
  };
  std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;  

  i2s_channel_init_std_mode(rx_handle, &std_cfg);
  i2s_channel_enable(rx_handle);
}

// ---------------- Setup ----------------
void setup() {
  Serial.begin(115200);
  delay(2000); // Laisse le temps au port série de s'ouvrir proprement
  
  Serial.println("\n===============================================");
  Serial.println("=== Test I2S INMP441 - Capture et Traitement ===");
  Serial.println("===============================================\n");

  i2s_init();
  Serial.println("Microphone I2S initialisé. Début de la lecture...\n");
}

// ---------------- Loop ----------------
void loop() {
  size_t bytesRead = 0;
  
  // 1. Lecture I2S (bloque jusqu'à ce que le buffer soit rempli)
  i2s_channel_read(rx_handle, rawBuffer, sizeof(rawBuffer), &bytesRead, portMAX_DELAY);
  int n = bytesRead / 4;

  if (n == 0) return;

  // 2. Décalage pour obtenir les données sur 24 bits
  for (int i = 0; i < n; i++) {
    s24[i] = rawBuffer[i] >> 8;
  }

  // 3. Application du filtre médian et conversion 16 bits
  for (int i = 0; i < n; i++) {
    int32_t a = (i == 0)     ? prevSample : s24[i - 1];
    int32_t b = s24[i];
    int32_t c = (i == n - 1) ? s24[i]     : s24[i + 1];
    
    int32_t v = med3(a, b, c) >> GAIN_SHIFT;
    
    // Clamping pour rester dans les limites d'un int16_t
    if (v > 32767)  v = 32767;
    else if (v < -32768) v = -32768;
    
    outBuffer[i] = (int16_t)v;
  }
  
  // Sauvegarde du dernier échantillon pour la prochaine boucle
  prevSample = s24[n - 1];

  // 4. Affichage des 10 premières valeurs de chaque lot
  Serial.println("--- Nouveau buffer (Affichage de 10 échantillons sur 4092) ---");
  for(int i = 0; i < 10 && i < n; i++) {
    Serial.print("Brut 32b: "); 
    Serial.print(rawBuffer[i]);
    Serial.print("\t| Décalé 24b: "); 
    Serial.print(s24[i]);
    Serial.print("\t| Sortie 16b: "); 
    Serial.println(outBuffer[i]);
  }
  Serial.println("..."); // Indique qu'il y a d'autres données
}