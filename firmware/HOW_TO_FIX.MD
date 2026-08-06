# How To Fix the Firmware DRAM Overflow

## Problem observed

The firmware build fails at link time with:

```text
section `.dram0.bss' will not fit in region `dram0_0_seg`
region `dram0_0_seg' overflowed by 93744 bytes
```

This means the firmware is not failing because of a syntax error or a missing include. It is failing because the static RAM usage is too large for the ESP32 DRAM region used by `.bss`.

## What is causing it in this firmware

The current firmware places too many large arrays in global/static RAM.

The biggest contributors are in `firmware/src/services/mfcc_utils.cpp`:

- `g_signal[AUDIO_WINDOW_SAMPLES]`
- `g_frame_real[kFftLength]`
- `g_frame_imag[kFftLength]`
- `g_magnitude[kFftBins]`
- `g_mel_energies[kMelCount][FEATURE_FRAME_COUNT]`
- `g_mel_db[kMelCount][FEATURE_FRAME_COUNT]`
- `g_mfcc_matrix[kMfccCount][FEATURE_FRAME_COUNT]`
- `g_delta_matrix[kMfccCount][FEATURE_FRAME_COUNT]`
- `g_delta2_matrix[kMfccCount][FEATURE_FRAME_COUNT]`
- `g_centroid[FEATURE_FRAME_COUNT]`
- `g_bandwidth[FEATURE_FRAME_COUNT]`
- `g_rolloff[FEATURE_FRAME_COUNT]`
- `g_zcr[FEATURE_FRAME_COUNT]`
- `g_rms[FEATURE_FRAME_COUNT]`

There is also the tensor arena in `firmware/src/main.cpp`:

- `tensor_arena[kTensorArenaSize]` with `kTensorArenaSize = 50 * 1024`

On top of that, the firmware also needs the normal Arduino / TensorFlow Lite / PlatformIO runtime memory.

## Why this overflows

The board is `nodemcu-32s`, which is an ESP32 with limited internal DRAM.

The rough static RAM footprint is already very high:

- `g_signal`: about 64 KB
- `g_frame_real` + `g_frame_imag`: about 8 KB
- mel / MFCC / delta matrices: several tens of KB
- tensor arena: 50 KB
- other runtime data: additional KBs

So the linker cannot place everything in `.dram0.bss`.

## What to do first

### 1. Reduce the tensor arena size

The fastest thing to test is lowering:

```cpp
constexpr int kTensorArenaSize = 50 * 1024;
```

Try values like:

- `40 * 1024`
- `32 * 1024`
- `24 * 1024`

If the model still allocates tensors, keep the smallest value that works.

### 2. Move big working buffers off static RAM if possible

The big arrays in `mfcc_utils.cpp` are the main RAM pressure.

Best options:

- allocate them dynamically only when needed,
- use `heap_caps_malloc()` if you want explicit ESP32 heap control,
- or reduce the number of intermediate arrays.

For tomorrow's test, the practical priority is to reduce memory pressure, not to redesign the entire pipeline.

## Best fixes in order

### Option A: Reduce arrays and tensor arena

This is the quickest path.

Actions:

- lower `kTensorArenaSize`,
- reduce one or more of the largest feature buffers,
- keep only the arrays strictly needed for the final 194-feature vector.

### Option B: Stream the computation

Instead of storing all intermediate matrices at once:

- compute a frame,
- extract MFCC / delta / delta2 summary values,
- write directly into the final feature vector,
- reuse the same temporary buffers.

This reduces RAM a lot.

### Option C: Use external PSRAM if the board has it

If you switch to an ESP32 variant with PSRAM, you can place large buffers there.

But for `nodemcu-32s`, do not assume PSRAM exists.

## Concrete places to edit

### `firmware/src/main.cpp`

Check:

```cpp
constexpr int kTensorArenaSize = 50 * 1024;
```

Lower this first if the model still runs.

### `firmware/src/services/mfcc_utils.cpp`

This file is the main RAM consumer.

The large static arrays should be reviewed first.

The biggest one is:

```cpp
float g_signal[AUDIO_WINDOW_SAMPLES];
```

Then the FFT and feature matrices.

## Suggested reduction strategy

### Step 1

Try building with a smaller tensor arena.

### Step 2

If the build still fails, reduce the number of persistent global arrays in `mfcc_utils.cpp`.

### Step 3

If needed, convert some buffers to local scratch buffers or reuse a single scratch buffer for multiple stages.

### Step 4

Rebuild until the link step succeeds.

## How to verify the fix

After each change, run:

```bash
inv run
```

If you only want to check compilation without flashing:

```bash
inv build
```

If the build succeeds, the linker overflow is resolved.

Then run the monitor:

```bash
inv monitor
```

You should see:

- boot logs,
- tensor allocation success,
- audio capture success,
- feature extraction success,
- chainsaw verdict output.

## What success looks like

You want the build to reach:

- `Linking .pio\\build\\nodemcu-32s\\firmware.elf`
- then complete without `dram0_0_seg` overflow
- then flash successfully if you use `inv run`

## What not to change first

Do not start by rewriting the MFCC math.

The current problem is memory layout, not algorithm correctness.

So the right order is:

1. shrink memory usage,
2. confirm the build links,
3. only then validate runtime output.

## Quick diagnosis summary

- The code structure is mostly correct.
- The firmware fails because too many large static buffers live in DRAM.
- The fix is to reduce RAM consumption, starting with the tensor arena and the biggest feature buffers.
- Once the linker overflow disappears, the firmware is much closer to being ready for the test.
