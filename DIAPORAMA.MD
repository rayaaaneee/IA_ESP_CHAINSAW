# Acoustic Chainsaw Detection on ESP32 — 15-minute presentation

**Deliverable:** slide-by-slide content + speaker script (English)
**Format:** 15 slides / 15 minutes (~1 min per slide)
**Project:** `IA_ESP_CHAINSAW` — TensorFlow → TFLite Micro → ESP32 edge-AI node

---

## Timing plan

| Slide | Title | Time | Cumulative |
|---|---|---|---|
| 1 | Who I am & context | 1:00 | 1:00 |
| 2 | Developer experience | 0:45 | 1:45 |
| 3 | Project overview | 1:00 | 2:45 |
| 4 | TRAINING — pipeline architecture | 1:00 | 3:45 |
| 5 | TRAINING — the dataset | 1:00 | 4:45 |
| 6 | TRAINING — features & model | 1:00 | 5:45 |
| 7 | TRAINING — Keras results | 1:15 | 7:00 |
| 8 | TRAINING — Keras vs TFLite parity | 1:00 | 8:00 |
| 9 | TRAINING — threshold study | 1:15 | 9:15 |
| 10 | TRAINING — firmware generation (the bridge) | 1:00 | 10:15 |
| 11 | FIRMWARE — architecture | 1:00 | 11:15 |
| 12 | FIRMWARE — DSP on the microcontroller | 1:00 | 12:15 |
| 13 | FIRMWARE — real-time detection loop | 1:00 | 13:15 |
| 14 | Results & current limitation | 1:00 | 14:15 |
| 15 | Future work & conclusion | 0:45 | 15:00 |

---

## SLIDE 1 — Detecting chainsaws in the rainforest, on a €5 microcontroller

**On-slide text**

- **Rayan** — engineering student, **CPE Lyon**, France
- Apprentice at **Energy Pool** (Lyon) — energy management / demand-response
- Research internship — **Universidad Militar Nueva Granada**, Bogotá
- **15/06/2026 → 14/08/2026**
- Supervisors: **Edward Guillen** & **Juan Carlos Martínez**
- Goal: an autonomous acoustic sensor against illegal logging

**Visuals**

- Top-left: portrait photo (round crop).
- Logo strip along the bottom: CPE Lyon · Energy Pool · Universidad Militar Nueva Granada.
- Right half: world map with two pins (Lyon → Bogotá) and a dotted flight path.
- Bottom-right corner: photo placeholder — the UMNG campus, or you working at the lab bench with the breadboard + ESP32.
- Optional background: a low-opacity rainforest photo, so the title reads over green.

**Script (~1:00)**

> Good morning everyone, and thank you for having me. My name is Rayan, I come from Lyon, in France, where I study engineering at CPE Lyon. In parallel with my studies I work as an apprentice at Energy Pool, in Lyon, a company operating in the energy-management sector — mostly demand-response and flexibility for electrical grids.
> For the last two months I have been here in Bogotá, at the Universidad Militar Nueva Granada, from the 15th of June to the 14th of August 2026, under the supervision of Edward Guillen and Juan Carlos Martínez, whom I would like to thank for their guidance.
> The project I am going to present today sits between artificial intelligence and embedded systems: building a small autonomous sensor that can hear a chainsaw in the rainforest and raise an alert — without internet, without GPS, and on a microcontroller that costs a few euros.

---

## SLIDE 2 — Developer experience: one command to rule them all

**On-slide text**

- `./init.sh` / `init.ps1` / `init.bat` → Python 3.11 **venv** created + dependencies installed, cross-platform
- **pip-tools**: `requirements.in` (hand-written) → `requirements.txt` (locked, reproducible)
- All actions exposed as **Invoke** tasks in `tasks.py`, with short aliases
- Python side: `inv extract` · `inv train` · `inv convert` · `inv validate-tflite` · `inv generate-firmware`
- Embedded side: `inv build` · `inv upload` · `inv monitor`
- **`inv full-pipeline`** = extract → train → convert → validate → generate firmware
- VS Code workspace + PlatformIO, recommended extensions pre-declared

**Visuals**

- A terminal mock-up showing `inv --list` output (task names + aliases), styled like a real shell.
- Below it, a horizontal "one-liner" flow: `init` → `venv` → `inv full-pipeline` → `inv upload` → `inv monitor`.
- Small icons: Python, VS Code, PlatformIO, Invoke.

**Script (~0:45)**

> Before the technical content, one word about developer experience, because it is a real part of the work.
> The repository ships with `init` scripts for Windows, WSL and Linux: they create the Python virtual environment and install every dependency, so a new developer is operational in one command. Dependencies are managed with pip-tools — I only maintain a short `requirements.in`, and the locked `requirements.txt` is compiled from it, which makes the environment reproducible.
> Then every single action in the project — feature extraction, training, conversion, validation, firmware generation, compiling and flashing the board — is a task in `tasks.py`, using the Invoke library, each with a short alias. In practice, going from a fresh dataset to a flashed microcontroller is three commands: `inv full-pipeline`, `inv upload`, `inv monitor`.

---

## SLIDE 3 — Project overview: from a WAV file to a listening device

**On-slide text**

- **Problem:** illegal logging in the Amazon is detected too late — no network, no power, no infrastructure
- **Idea:** the chainsaw *sounds* — a steady two-stroke engine signature, audible far away
- **Objective:** train a TensorFlow model to recognise chainsaw sound, then run it *on-device*
- **Chain:** WAV dataset → Keras model → **TFLite** conversion → embedded in an **ESP32-WROOM-32**
- Audio captured by an **INMP441** I2S digital microphone, 8 kHz, 2-second windows
- 100 % offline inference: no internet, no GPS, no cloud
- Two decoupled modules: **`training/`** (Python, on a PC) and **`firmware/`** (C++, on the board)

**Visuals**

- Central end-to-end diagram, left to right:
  `🎵 WAV dataset → 🧠 Keras (.h5) → 📦 TFLite (.tflite) → ⚙️ C++ header files → 🔌 ESP32 + 🎤 INMP441 → 🚨 alert`
- Photo placeholder: your actual breadboard — ESP32 + INMP441 wired up.
- A small "8 kHz · 2 s window · 1 s hop · 194 features" badge.

**Script (~1:00)**

> Here is the problem. In remote forest areas, illegal logging is usually discovered weeks later, from satellite imagery, when the trees are already gone. There is no mobile network, no power grid, no infrastructure. But a chainsaw makes a very characteristic noise: a steady two-stroke engine, with a stable harmonic structure, audible over a long distance.
> So the objective of the project is to train a TensorFlow neural network to recognise that sound, and then to run this network directly inside a microcontroller placed in the forest.
> The chain is the following: I start from a labelled dataset of WAV files, I train a Keras model on a PC, I convert it to TensorFlow Lite, and I embed it into an ESP32, which listens continuously through an INMP441 digital microphone at 8 kilohertz.
> Everything runs on-device. The project is split into two completely decoupled modules: the Python training pipeline, and the C++ firmware.

---

## SLIDE 4 — TRAINING module: a five-stage pipeline

**On-slide text**

- `extract_features.py` → `data/processed/feature_dataset.npz` + manifest
- `train.py` → `model/model.h5` + `training_report_NN.json`
- `convert_to_tflite.py` → `model/model.tflite`
- `validate_tflite.py` → `tflite_report_NN.json`
- `generate_firmware.py` → the three C++ files consumed by the firmware
- Each stage **reads the previous stage's output from disk** → any stage can be re-run alone
- **Dataset signature** (hash of every raw file: path, size, mtime) → automatic stale-cache detection
- Side tools: `check.py` (cache & label consistency), `chart.py` (learning curves, history across runs)

**Visuals**

- Reproduce the `mermaid` flowchart of `training/README.md` as a clean 5-box horizontal pipeline, each box labelled with its output artifact underneath (`.npz`, `.h5`, `.tflite`, `.json`, `.h/.cpp`).
- Off to the side, two dashed boxes `check.py` and `chart.py` marked "diagnostics, not part of the pipeline".
- A small key icon on the arrow between stages: "dataset signature = cache invalidation".

**Script (~1:00)**

> Let's start with the training module. It is a five-stage pipeline, and the important design decision is that the stages are decoupled: each one reads the previous stage's output from disk, so I can re-run any stage independently without redoing the whole chain.
> First, feature extraction turns the WAV folder into a compressed NumPy dataset plus a JSON manifest. Second, training produces the Keras model and a timestamped training report. Third, conversion produces the TensorFlow Lite model. Fourth, validation compares the two models. And fifth, firmware generation writes the C++ files.
> To avoid working on a stale cache, the extraction step computes a dataset signature — a hash of every raw file's path, size and modification time — so the downstream stages can automatically detect that the dataset changed and re-extract.
> Alongside the pipeline, two diagnostic tools check the cache and the labels, and plot the learning curves.

---

## SLIDE 5 — The dataset: 1,911 recordings, 5,963 windows

**On-slide text**

- **Positives — chainsaw: 492 files** — 3 "unique" sources (generic, Oregon, Stihl) + 2 "mixed" sets recorded over jungle ambience
- **Negatives — environment: 1,419 files** — birds, jaguar, monkey, snake, rain, tropical rain, rainforest ambience ×2, motocross
- Sliced into **2 s windows with 1 s hop** → **5,963 feature vectors** (3,580 train / 1,197 val / 1,186 test)
- **Automatic labelling from the folder path** (`chainsaw`/`motosierra` → 1, `birds`/`lluvia`/`jaguar`… → 0); an unmatched file raises an error
- **Grouped split**: all windows of one recording stay on the same side → no data leakage
- Class imbalance ≈ **26 % positive** → handled with class weights (0.68 / 1.90)
- *Motocross is deliberately in the negatives: a two-stroke engine that is not a chainsaw*

**Visuals**

- Donut chart: positives vs negatives (492 / 1,419 files).
- Horizontal bar chart of negative categories (rain 306, rainforest ambience 306, birds 201, motocross 201, jaguar 122, monkey 100, ambience beta 72, snake 60, tropical rain 51).
- Small schematic of the sliding window: a waveform with three overlapping 2 s brackets shifted by 1 s.
- Icons: 🪚 🐦 🐒 🐆 🌧️ 🐍 🏍️.

**Script (~1:00)**

> Now the data. The dataset contains about 1,900 recordings: roughly 490 chainsaw recordings — from three different sources, including Stihl and Oregon saws, plus two sets where the chainsaw is mixed into jungle ambience — and about 1,400 negative recordings of the environment: birds, monkeys, a jaguar, snakes, rain, tropical rain, rainforest ambience, and motocross.
> That last one is deliberate: a motocross bike is a two-stroke engine that is not a chainsaw, so it is exactly the kind of hard negative the model must learn to reject.
> Each file is cut into two-second windows with a one-second hop, which gives almost six thousand feature vectors. Labelling is automatic, derived from the folder path, and a file that matches no known token raises an error — so the folder structure is the single source of truth.
> One important detail: the split is grouped by source recording, so all windows coming from the same file stay on the same side of the split. Otherwise the model would be evaluated on windows almost identical to those it trained on.

---

## SLIDE 6 — Features & model: 194 numbers, 35 k parameters

**On-slide text**

**Feature vector — 194 values per 2 s window**
- 20 **MFCC** + delta + delta² (librosa)
- 32-band **mel spectrogram** in dB
- spectral **centroid**, **bandwidth**, **rolloff**, **zero-crossing rate**, **RMS energy**
- each series reduced to **(mean, std)** over time → fixed-size vector

**Model — a small dense MLP**
```
Input(194) → Normalization (adapted on train split)
→ Dense(128, relu) → Dropout
→ Dense(64, relu)  → Dropout
→ Dense(32, relu)
→ Dense(1, sigmoid)
```
- ≈ **35,300 parameters**, Adam + binary cross-entropy
- `EarlyStopping` + `ReduceLROnPlateau`
- Single source of truth: the **`FeatureConfig`** dataclass (8 kHz, 2 s / 1 s, `n_mfcc=20`, `n_mels=32`, `n_fft=1024`, `hop=256`)

**Visuals**

- Left: a mel-spectrogram image of a chainsaw clip next to one of rainforest ambience — the visual contrast sells the whole approach.
- Right: vertical network diagram 194 → 128 → 64 → 32 → 1.
- A "194 = 20×2 MFCC + 20×2 Δ + 20×2 Δ² + 32×2 mel + 5×2 spectral… " decomposition bar, colour-coded by block.

**Script (~1:00)**

> How do we describe two seconds of sound to a neural network? Not with raw samples — sixteen thousand samples would be far too much for this microcontroller. Instead I extract a compact acoustic description: twenty MFCC coefficients, which capture the timbre, plus their first and second derivatives, which capture how that timbre evolves; a thirty-two-band mel spectrogram; and five classical spectral descriptors — centroid, bandwidth, rolloff, zero-crossing rate and energy.
> Each of these is a time series, and I summarise each one by its mean and its standard deviation, which gives a fixed-size vector of one hundred and ninety-four values.
> The model itself is deliberately small: a dense network, one hundred and ninety-four inputs, three hidden layers, one sigmoid output — about thirty-five thousand parameters. All the signal-processing parameters come from one single dataclass, `FeatureConfig`, and that will matter a lot in three slides.

---

## SLIDE 7 — Training results (report #06)

**On-slide text**

- 11 epochs (early-stopped), best weights restored, LR halved by `ReduceLROnPlateau`
- **Validation:** accuracy **98.3 %** · precision **97.8 %** · recall **95.9 %** · **AUC 0.994**
- **Test:** accuracy **97.9 %** · precision **96.1 %** · recall **95.8 %** · **AUC 0.995**
- Test confusion matrix @ 0.5 → TP 297 · TN 864 · **FP 12** · **FN 13**
- Train/val gap stays small → no visible overfitting on this dataset

**Visuals**

- Learning curves (two side-by-side line charts, from `chart.py --latest`): loss & val_loss; accuracy & val_accuracy. Loss drops 0.27 → 0.013, val_loss flattens ~0.06–0.07.
- A 2×2 confusion-matrix heatmap for the test set (297 / 864 / 12 / 13).
- A metric "scorecard" row of four big numbers: 97.9 % · 96.1 % · 95.8 % · 0.995.

**Script (~1:15)**

> These are the results of the training run I kept, report number six. Training stops after eleven epochs thanks to early stopping, and the learning-rate scheduler halves the learning rate at the end.
> On the test set — remember, recordings the model has never seen — accuracy is around ninety-eight percent, precision ninety-six, recall ninety-six, and the area under the ROC curve is 0.995. In absolute numbers, on eleven hundred and eighty-six test windows, the model makes twelve false positives and thirteen false negatives.
> The gap between training and validation curves stays small, so there is no obvious overfitting on this dataset.
> I want to be honest about what these numbers mean, though: they are computed on the same *kind* of audio the model was trained on. They tell us the architecture and the features are appropriate. They do not yet tell us how the device behaves in a real forest — I will come back to that at the end.

---

## SLIDE 8 — Keras vs TFLite: the conversion is numerically lossless

**On-slide text**

- `validate_tflite.py` runs **both models on the exact same splits** and compares them
- **Prediction mismatches: 0** on validation and test
- Mean absolute probability difference: **4 × 10⁻⁹** (val) / **6 × 10⁻⁹** (test); max **8 × 10⁻⁷**
- Metric deltas ≈ 10⁻⁷–10⁻⁸ → pure float32 rounding
- ⚠️ **No `Optimize.DEFAULT`**: dynamic-range int8 quantization worked on desktop but produced **NaN** on TFLite Micro / ESP32 → the exported model stays **plain float32**
- Report caching: a report is written only for a new `(training report, threshold, tflite hash)` triplet → no redundant reports

**Visuals**

- Two overlaid histograms of predicted probabilities (Keras vs TFLite) — they superimpose perfectly; caption "0 mismatches / 2,383 windows".
- A log-scale bar chart of the absolute probability difference (mean and max) with a reference line at the 0.5 decision threshold, to show the difference is 6 orders of magnitude below anything that matters.
- A red "❌ Optimize.DEFAULT → NaN on MCU" callout box next to a green "✅ float32" box.

**Script (~1:00)**

> Converting to TensorFlow Lite is not automatically safe, so I built a validation step that runs both models — the Keras one and the converted one — on exactly the same validation and test splits, and compares them window by window.
> The result is that the conversion is essentially lossless: zero prediction mismatches out of more than two thousand windows, and a mean absolute difference on the output probability of the order of ten to the minus nine. That is pure floating-point rounding.
> There is one important negative result here. I initially enabled the default optimisation, which applies dynamic-range int8 quantization. It works perfectly with the desktop interpreter, but on the microcontroller, TensorFlow Lite Micro produced NaN outputs. So I deliberately export the model in plain float32, and I documented why — the model is small enough that we can afford it.

---

## SLIDE 9 — Choosing the decision threshold: trading false positives against false negatives

**On-slide text**

Same TFLite model, four validation reports at four thresholds — **test set**:

| Threshold | FP | FN | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| 0.30 | 20 | 12 | 93.7 % | **96.1 %** | 97.3 % |
| 0.40 | 17 | 13 | 94.6 % | 95.8 % | 97.5 % |
| 0.50 | 12 | 13 | 96.1 % | 95.8 % | 97.9 % |
| 0.60 | **9** | 14 | **97.1 %** | 95.5 % | **98.1 %** |

- Raising the threshold: **FP 20 → 9** (−55 %) for the cost of **only 2 extra FN**
- **False negative** = a chainsaw we miss — a tree is cut
- **False positive** = a false alarm → wasted ranger patrol, and loss of trust in the system
- In the field, alarm fatigue kills the system faster than a missed event → the firmware runs at **0.65**

**Visuals**

- Main chart: dual-axis line plot — X = threshold (0.3 → 0.6), one descending line for false positives, one ascending line for false negatives; the "operating point" at 0.65 marked with a vertical dashed line. This is *the* chart of the presentation.
- Secondary: grouped bars precision / recall / F1 per threshold.
- Two icons with one-line captions: 🌲🪚 "FN = a tree is lost" · 🚙💨 "FP = a patrol sent for nothing".

**Script (~1:15)**

> Once the model is validated, there is still a free parameter: the decision threshold on the output probability. I generated four validation reports for the same model, at thresholds 0.3, 0.4, 0.5 and 0.6.
> The trend is clear. At 0.3 we get twenty false positives and twelve false negatives; at 0.6, only nine false positives, and just two more false negatives. In other words, raising the threshold removes more than half of the false alarms at almost no cost in detection.
> Why does this matter? The two errors do not cost the same thing. A false negative means a chainsaw we did not hear — a tree gets cut. A false positive means a false alarm, and in a real deployment, sending a ranger patrol through the jungle for nothing is extremely expensive; more importantly, a device that cries wolf will simply be ignored, and then it protects nothing.
> That is why the firmware operates at 0.65 — slightly more conservative than the best offline accuracy point.

---

## SLIDE 10 — The bridge: `generate_firmware.py` — Python writes the C++

**On-slide text**

The `generate/` sub-module produces **three files**, from the trained model and `FeatureConfig`:

| Generator | Output | Content |
|---|---|---|
| `generate_feature_config.py` | `include/app/config.h` | `AUDIO_CONFIG` — sample rate, window/hop, `n_mfcc`, `n_mels`, FFT & hop length |
| `generate_dsp_constants.py` | `src/app/dsp_constants.cpp` | **Mel filterbank** + **DCT** matrices as flat `float` arrays |
| `generate_inference.py` | `src/model/inference.cpp` | the `.tflite` bytes as `const uint8_t g_model_data[]` + SHA-256 for traceability |

- ⛔ **These three files must never be hand-edited** — silently overwritten at each regeneration
- Everything that must survive regeneration lives in `board_config.h` (pins, thresholds, watchdog)
- **This is the contract between the two modules**: the Python extractor and the C++ extractor mathematically *cannot* diverge, because both derive from the same `FeatureConfig`
- Consequence: the firmware needs **no Python, no TensorFlow, no filesystem** — everything is compiled in

**Visuals**

- A central "bridge" metaphor: left bank = Python/`training`, right bank = C++/`firmware`, and `generate_firmware.py` as the bridge deck, with three parallel arrows labelled `config.h`, `dsp_constants.cpp`, `inference.cpp`.
- A padlock icon on `FeatureConfig` labelled "single source of truth".
- A small code snippet of the generated `g_model_data[] = {0x1c, 0x00, ...}` with the SHA-256 comment above it.

**Script (~1:00)**

> This slide is, for me, the most important one of the training module. The generation sub-module is the bridge between Python and C++.
> It writes three files. The first one exports the feature configuration — sample rate, window size, number of MFCCs, FFT size — as a C header. The second precomputes the mel filterbank matrix and the DCT matrix, the same ones librosa builds internally, and writes them as plain float arrays, so the firmware can compute MFCCs without linking any heavy math library. The third embeds the TensorFlow Lite model itself as a byte array, with its SHA-256 for traceability.
> These three files are generated, never hand-written, and everything that must survive a regeneration — pin numbers, detection thresholds, watchdog timeout — lives in a separate, hand-maintained header.
> This is what guarantees coherence between the two modules: the Python and the C++ feature extractors cannot drift apart, because both are derived from one single configuration object.

---

## SLIDE 11 — FIRMWARE module: architecture

**On-slide text**

- **PlatformIO / Arduino framework**, C++17, board `nodemcu-32s` (**ESP32-WROOM-32**)
- Libraries: `TensorFlowLite_ESP32` (TFLite Micro runtime) + `arduinoFFT` (on-device FFT)
- Layered layout:
  - `drivers/` — I2S microphone (**INMP441**)
  - `services/` — `mfcc` (feature extraction), `watchdog`
  - `model/` — generated model bytes + TFLite Micro interpreter
  - `app/` — `config.h` (generated), `board_config.h` (hand-maintained), `flags.h`
  - `utils/debug.h` — verbose serial output, gated by `ENABLE_DEBUG_LOGS` (off by default)
- **Compile-time flags** → disabled features cost zero bytes and zero cycles
- Fail-safe philosophy: if the watchdog cannot be configured, the firmware warns once and keeps running instead of blocking

**Visuals**

- Layered architecture diagram (a stack of coloured bands): `main.cpp` on top, then `services`, then `drivers` + `model`, then `app/config`.
- Colour-code each file as 🟩 generated / 🟦 hand-maintained — a legend makes slide 10 pay off visually.
- Photo placeholder: close-up of the wired ESP32 + INMP441 module.

**Script (~1:00)**

> On the embedded side, the firmware is a PlatformIO project, C++17, targeting an ESP32-WROOM-32. It uses only two external libraries: the TensorFlow Lite Micro runtime for the ESP32, and a small FFT library.
> The code is organised in layers: a driver layer for the I2S microphone; a services layer containing the feature extraction and the watchdog; a model layer with the generated model and the interpreter; and a configuration layer where you can clearly see which files are generated by Python and which ones are hand-maintained.
> Features are controlled by compile-time flags, so a disabled feature costs literally zero bytes of flash and zero CPU cycles. And the services are written to fail safe: if something cannot be initialised, the firmware logs a warning once and keeps running with that feature disabled, rather than blocking at start-up. On a device meant to sit alone in a forest, never blocking is more valuable than being complete.

---

## SLIDE 12 — Re-implementing librosa in C++, exactly

**On-slide text**

- `extract_features_from_audio()` reproduces the **exact same 194-value vector** as the Python extractor — without librosa, without NumPy
- Per window: **63 STFT frames**, Hann-windowed, **reflect-padded** at the edges
- FFT → mel energies (`MEL_MATRIX`) → 20 MFCC (`DCT_MATRIX`) → Δ and Δ² with librosa-equivalent Savitzky–Golay coefficients
- Reduced on the fly with a streaming **`RunningStats`** accumulator → no full-length buffers in RAM
- 🐛 **The bug that cost me the most:** librosa uses **two different dB references** — `ref=1.0` inside `mfcc()`, `ref=np.max` for the standalone mel block. Mixing them up made the model return a near-zero probability *whatever* the input.
- Lesson: on embedded ML, **most bugs are not in the model — they are in the feature parity**

**Visuals**

- Side-by-side code comparison: a Python line (`librosa.feature.mfcc(...)`) facing its ~10-line C++ equivalent.
- A bar diagram of the 194-value memory layout: `[mfcc µ/σ ×20][Δ ×20][Δ² ×20][mel µ/σ ×32][centroid][bandwidth][rolloff][zcr][rms]`.
- A "before / after" plot of output probability on the same chainsaw clip: flat near 0 → correct high value, captioned with the dB-reference fix.

**Script (~1:00)**

> This is where most of the engineering time actually went. The firmware has to compute *exactly* the same one hundred and ninety-four numbers as the Python extractor — otherwise the model receives inputs from a different distribution than the one it was trained on, and it simply fails, silently.
> So I re-implemented, in C++, the whole chain: sixty-three overlapping FFT frames with a Hann window and reflect padding at the edges, the mel filterbank, the DCT to get the MFCCs, and the delta coefficients with the same convolution kernels librosa uses. Everything is accumulated in a streaming way, so I never keep full-length buffers in memory.
> And here is the bug that cost me the most time: librosa internally uses two different decibel references — an absolute one inside the MFCC computation, and a window-relative one for the standalone mel block. I was using the same one for both, and the model returned almost zero probability no matter what I played at it. The model was fine. The features were not.

---

## SLIDE 13 — The real-time detection loop

**On-slide text**

- `setup()`: load the embedded model, build a `MicroInterpreter` over a **6 KiB tensor arena**, `AllocateTensors()`, init microphone, DSP and watchdog
- `loop()`: **sliding window** — the first pass records a full 2 s; after that the buffer is shifted left by 1 s and only the newest second is recorded
  → inference **every ~1 s** instead of every 2 s → **detection latency halved**
- 16,000 `int16` samples → 194 floats → input tensor → `Invoke()` → probability
- **Debounce / hysteresis state machine** before raising anything:

| Constant | Value | Role |
|---|---|---|
| `kDetectionEnterThreshold` | **0.65** | probability needed to start counting |
| `kDetectionConsecutiveFrames` | **2** | consecutive windows required to confirm |
| `kDetectionExitThreshold` | **0.35** | probability must fall below this to clear |
| `kWatchdogTimeoutSeconds` | **10** | the board reboots if the loop hangs |

- Two thresholds ≠ one: hysteresis prevents the verdict from flickering around the decision boundary

**Visuals**

- A timeline chart: probability over time (noisy curve), with the 0.65 and 0.35 lines, and below it a binary "ALERT" band showing the state machine confirming after two frames and holding until the probability drops below 0.35.
- A sliding-window animation strip: `[■■]` → shift → `[■■]`, with the "only 1 new second recorded" arrow.
- Small chip icon labelled "6 KiB arena — inference entirely in RAM".

**Script (~1:00)**

> Here is what happens on the device, once per second, forever.
> At start-up the firmware loads the embedded model and allocates the TensorFlow Lite Micro interpreter in a six-kilobyte tensor arena. Then the loop captures audio into a sliding window: the very first pass blocks for a full two seconds, but after that it shifts the buffer left by one second and only records the newest second. That single optimisation halves the worst-case detection latency, because inference now runs every second instead of every two.
> The sixteen thousand samples become one hundred and ninety-four features, we run the interpreter, and we get a probability.
> But a single probability is never enough to raise an alert. It goes through a debounce and hysteresis state machine: two consecutive windows above 0.65 to confirm a detection, and the probability must drop below 0.35 to clear it. Using two different thresholds instead of one prevents the verdict from flickering when the signal hovers around the boundary.
> And a hardware watchdog reboots the board if the loop ever hangs.

---

## SLIDE 14 — Results: excellent on chainsaws, too eager on the rest

**On-slide text**

- ✅ On real chainsaw sound, the latest model is **very reliable** — recall ≈ 96 %, detection confirmed in ~2 s
- ✅ Keras → TFLite → ESP32 chain validated end to end, running fully offline
- ⚠️ **The remaining weakness is false positives**: on sounds that are *not* chainsaws, the device sometimes fires
- Why?
  - The dataset contains **no wind**, **no insects**, **no human voices**, **no microphone self-noise**, **no non-chainsaw small engines** (brush cutters, generators)
  - The firmware **peak-normalises each 2 s window** → in a near-silent window, the microphone's own noise floor is amplified to full scale, and the model has never been shown that pattern as a negative
- The model is not "wrong" — it is being asked about sounds it was never taught to reject

**Visuals**

- A two-column "✅ / ⚠️" panel: what works vs what does not.
- Illustration of the peak-normalisation effect: a tiny noise waveform on the left, and the same waveform stretched to full scale after normalisation, with a "the model sees this" caption.
- A "missing from the dataset" grid of greyed-out icons: 💨 wind · 🦗 insects · 🗣️ voices · 🔇 mic self-noise · 🌩️ thunder · ⚙️ brush cutter.

**Script (~1:00)**

> So where does the project stand today? The last dataset gave a model that works very well on real chainsaw sounds — when there is a chainsaw, it is detected, reliably, in about two seconds. The full chain, from Keras to TensorFlow Lite to the ESP32, is validated and runs completely offline.
> The remaining weakness is the opposite direction: false positives. On sounds that are not chainsaws, the device sometimes fires.
> And when you look at the dataset, this is not surprising. There is no wind in it — and the microphone has no wind screen. There are no insects, although cicadas are one of the loudest and most constant sounds in a tropical forest. No human voices, no non-chainsaw small engines like brush cutters, which are acoustically the closest thing to a chainsaw. And no recordings of the microphone's own noise floor — which matters, because the firmware peak-normalises every window, so a nearly silent window gets its own hiss amplified to full scale, and the model has simply never been shown that pattern labelled as "not a chainsaw".

---

## SLIDE 15 — Future work & conclusion

**On-slide text**

**1. Close the dataset gap (highest priority)**
- Record **on the deployed hardware itself**: microphone self-noise, wind, insects, rain on the enclosure, human voices, brush cutters/generators
- Cheap multiplier: **data augmentation** — mix noise at several SNRs, pitch/time-shift chainsaw clips, low-pass filter them to simulate distance

**2. Tune the threshold on real field data**
- Today 0.65 is chosen from an *offline* study; the right way is to **log probabilities in situ** for several days and pick the operating point that minimises false alarms while keeping recall high
- The `validate-tflite` task already produces a report per threshold → the method is in place, it needs field recordings

**3. Then, and only then, deploy**
- Target: a node the rangers **trust** — few enough false alarms that every alert is acted upon

> **Conclusion:** the pipeline, the tooling and the embedded chain work end to end. What separates this prototype from a field-ready device is not more model capacity — it is **more representative negative data, and a threshold calibrated in the Amazon rather than on a desk.**

**Visuals**

- A three-step roadmap arrow: "Record real negatives → Re-train → Calibrate threshold in situ → Deploy".
- Re-use the FP/FN threshold curve from slide 9, greyed out, with a question mark on the operating point, captioned "to be re-measured in the field".
- Closing image: rainforest photo + the node's silhouette; final line "Thank you — questions?" plus your contact details and the three logos again.

**Script (~0:45)**

> To conclude, the most valuable next step is not a bigger model. It is better negative data — and specifically, negatives recorded on the deployed hardware itself: wind, insects, human voices, brush cutters, and the microphone's own noise floor. Data augmentation, mixing noise at several signal-to-noise ratios and low-pass filtering the chainsaw clips to simulate distance, is a cheap way to multiply that coverage.
> The second step is threshold optimisation based on real field conditions. Today the value 0.65 comes from an offline study on my dataset. The right method is to install the node, log its probabilities for several days in the actual environment, and then choose the operating point that minimises false positives while keeping the detection rate high. The tooling for that already exists — every validation run produces a report per threshold.
> Because in the end, the criterion for success is not accuracy on a test set. It is whether a ranger in the Amazon trusts the alert enough to walk into the forest. Thank you very much — I'll be happy to take your questions.

---

## Appendix — reusable numbers

| Item | Value |
|---|---|
| Sample rate / window / hop | 8 kHz / 2.0 s / 1.0 s |
| Feature vector | 194 values, 63 STFT frames, `n_fft=1024`, `hop=256`, `n_mfcc=20`, `n_mels=32` |
| Dataset | 1,911 WAV files (492 chainsaw / 1,419 environment) → 5,963 windows |
| Splits | 3,580 train / 1,197 validation / 1,186 test |
| Class balance (train) | ≈ 2,638 negative / 942 positive → weights 0.68 / 1.90 |
| Model | 194→128→64→32→1, ≈ 35,300 parameters, float32 |
| Keras test | acc 97.89 % · prec 96.12 % · rec 95.81 % · AUC 0.9952 |
| Keras validation | acc 98.33 % · prec 97.77 % · rec 95.94 % · AUC 0.9941 |
| TFLite parity | 0 mismatches, mean │Δp│ ≈ 5 × 10⁻⁹, max ≈ 8 × 10⁻⁷ |
| Test FP/FN @ 0.3 / 0.4 / 0.5 / 0.6 | 20/12 · 17/13 · 12/13 · 9/14 |
| Runtime thresholds | enter 0.65 · exit 0.35 · 2 consecutive frames |
| Tensor arena / watchdog | 6 KiB / 10 s |

### Suggested charts to generate from the JSON reports

1. **Learning curves** — `training_report_06.json` → `history.loss` vs `history.val_loss`, and `accuracy` vs `val_accuracy` (11 epochs).
2. **Threshold trade-off** (the key chart) — X = 0.3/0.4/0.5/0.6, two series: `tflite_test_metrics.false_positives` and `false_negatives`.
3. **Precision / Recall / F1 vs threshold** — grouped bars: 93.7/96.1/94.9 · 94.6/95.8/95.2 · 96.1/95.8/96.0 · 97.1/95.5/96.3.
4. **Keras vs TFLite parity** — log-scale bar of `mean_abs_probability_diff` and `max_abs_probability_diff` against a 0.5 reference line; annotate "prediction_mismatch_count = 0".
5. **Confusion matrices** — 2×2 heatmaps for the test set at 0.5 and at 0.6, side by side, to visualise the 12 → 9 false-positive reduction.
6. **Dataset composition** — donut (positive/negative) + horizontal bars per environment category.