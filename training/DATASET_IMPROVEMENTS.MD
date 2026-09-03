# Dataset improvements for field robustness

The current dataset (`training/data/raw/`) only has two top-level classes:

- `chainsaw/` (`mixed/`, `unique/`) — positive samples.
- `environment/` — negative samples, currently limited to `bird/`, `jaguar/`, `monkey/`, `motocross/`, `rain/`, `rainforest_ambience/`, `rainforest_ambience_beta/`, `snake/`, `tropical_rainforest_rain/`.

This covers wildlife and rain reasonably well, but it's missing several sound categories the ESP32 node will actually encounter once deployed in the field, and this gap is a likely source of false positives/negatives on real hardware. This document lists the additions worth prioritizing.

Whenever you add a new folder, remember to register a matching keyword in `POSITIVE_TOKENS`/`NEGATIVE_TOKENS` in [`extract_features.py`](extract_features.py) (or nest it under an existing recognized folder name) so `infer_label()` can classify it automatically — otherwise `build_feature_dataset()` raises a `ValueError`.

## Missing negative categories to add

| Category | Suggested folder | Why it matters |
| --- | --- | --- |
| **Human voices** | `environment/human_voice/` (talking, shouting, footsteps, radio chatter) | Rangers, loggers, or hikers near the node must never be misclassified as a chainsaw; voices and chainsaw drones don't share much spectral structure, but the model has never been shown an example to explicitly reject. |
| **Wind / breath noise on the microphone** | `environment/wind/` | The INMP441 has no wind screen; direct airflow across the diaphragm produces loud, broadband, low-frequency rumble that can resemble a running engine. This is one of the most common false-positive sources for any outdoor microphone and is completely absent from the dataset today. |
| **Microphone self-noise / true silence** | `environment/silence/` (or `environment/mic_noise/`) | Recorded directly from the deployed hardware (INMP441 + ESP32) in a quiet room/enclosure, not downloaded stock audio. This directly addresses the bug diagnosed in this session: firmware peak-normalizes each 2s window, so a near-silent window gets its noise floor amplified to full scale. Without real "amplified self-noise" examples labeled as negative, the model has no basis to reject that specific input pattern. |
| **Insects (cicadas, crickets)** | `environment/insects/` | Very loud, broadband/tonal, and extremely common in tropical rainforest — currently not represented at all despite being one of the most frequent ambient sounds on-site. |
| **Small engines that aren't chainsaws** | `environment/machinery/` (brush cutters/weed trimmers, generators, water pumps, leaf blowers, drones) | Two-stroke brush cutters in particular have a spectral signature very close to a chainsaw's (similar engine type, similar RPM range) and are commonly used alongside real logging/poaching activity — a much harder negative than birds or rain, and a realistic field mix-up. |
| **Thunder / storm** | `environment/thunder/` | Sudden broadband transients distinct from steady rain; not covered by the existing `rain/` clips. |
| **Rain hitting the enclosure directly** | `environment/rain_on_case/` | Distinct from ambient/distant rain (`rain/`, `tropical_rainforest_rain/`): rain drops hitting the physical housing produce sharp, close-mic transients with a different spectral profile than rain heard from a distance. |

## Chainsaw-side improvements

- **Distance/attenuation variety**: most positive clips likely capture the chainsaw close-up. Add far-field and through-foliage recordings (attenuated, more low-pass filtered) so the model generalizes to realistic detection ranges rather than only near-field, unattenuated sound.
- **Operating phases**: idle vs. actively cutting through wood vs. revving — these have different spectral envelopes and both should trigger detection.
- **Multiple chainsaw models/brands**: reduces the risk of overfitting to one engine's specific harmonic signature.

## Data augmentation (cheap way to multiply coverage)

Rather than only sourcing new raw recordings, augmenting existing clips is a fast way to widen coverage of the categories above:

- Mix low-level wind/self-noise or pink/white noise into existing clips at a few SNR levels, to teach the model that noisy-but-quiet input isn't automatically "interesting" (positive or negative — this in particular targets the peak-normalization false-positive issue).
- Time-stretch / pitch-shift chainsaw clips slightly to simulate different engine models and recording distances.
- Apply a low-pass filter to a copy of each positive clip to approximate far-field attenuation.

## Priority order

1. Microphone self-noise/silence (directly tied to a confirmed firmware bug pattern).
2. Wind noise (most likely outdoor false-positive source).
3. Human voices (safety-relevant: must not misclassify people as chainsaws, and vice versa).
4. Small engines / brush cutters (hardest realistic negative).
5. Insects, thunder, rain-on-case (fill remaining ambient-sound gaps).
6. Chainsaw distance/phase/model variety, then augmentation.
