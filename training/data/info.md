# Chainsaw datasets information

Filenames for the chainsaw dataset can be fetched using the following pattern:

```python
nb_files = 50
filenames = [f"motosierra_digital_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/chainsaw/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "motosierra_digital_001.wav" to "motosierra_digital_050.wav".
```

# Environment datasets information

## Motocross

Filenames for the motocross dataset can be fetched using the following pattern:

```python
nb_files = 201
filenames = [f"motocross_digital_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/environment/motocross_digital/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
```This will fetch all dataset files from 'motocross_digital_001.wav' to 'motocross_digital_201.wav'.```
```

## Rain

Filenames for the rain dataset can be fetched using the following pattern:

```python
nb_files = 306
filenames = [f"Lluvia_digital_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/environment/lluvia_digital/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
```This will fetch all dataset files from 'Lluvia_digital_001.wav' to 'Lluvia_digital_306.wav'.```
```

## Rainforest atmosphere

Filenames for the rainforest atmosphere dataset can be fetched using the following pattern:

```python
nb_files = 306
filenames = [f"digital_ambience_rainforest_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/environment/digital_ambience_rainforest_/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "digital_ambience_rainforest_001.wav" to "digital_ambience_rainforest_306.wav".```
```

## Rainforest atmosphere beta

Filenames for the rainforest atmosphere beta dataset can be fetched using the following pattern:

```python
nb_files = 73
filenames = [f"rainforest_atmosphere_beta_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/environment/rainforest_atmosphere_beta/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
```This will fetch all dataset files from 'rainforest_atmosphere_beta_001.wav' to 'rainforest_atmosphere_beta_073.wav'.```
```

## Tropical rainforest rain

Filenames for the rain in rainforest dataset can be fetched using the following pattern:

```python
nb_files = 51
filenames = [f"Lluvia_selva_tropical_digital_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/environment/lluvia_selva_tropical_digital/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
```This will fetch files from 'Lluvia_selva_tropical_digital_001.wav' to 'Lluvia_selva_tropical_digital_051.wav'.```
```