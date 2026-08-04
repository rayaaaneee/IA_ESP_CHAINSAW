# Chainsaw datasets information

## Unique

### Normal

Filenames for the chainsaw dataset can be fetched using the following pattern:

```python
nb_files = 130
filenames = [f"motosierra_digital_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/chainsaw/unique/normal/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "motosierra_digital_001.wav" to "motosierra_digital_130.wav".
```

### Oregon

Filenames for the oregon dataset can be fetched using the following pattern:

```python
nb_files = 7
filenames = [f"motosierra_f_oregon_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/chainsaw/unique/oregon/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "motosierra_f_oregon_001.wav" to "motosierra_f_oregon_007.wav".
```

### Stihl

Filenames for the stihl dataset can be fetched using the following pattern:

```python
nb_files = 130
filenames = [f"chainsaw_stihl_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/chainsaw/unique/stihl/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "chainsaw_stihl_001.wav" to "chainsaw_stihl_130.wav".
```
## Mixed

Filenames for the mixed dataset can be fetched using the following pattern:

```python
nb_files = 54
filenames = [f"chainsaw_jungle_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/chainsaw/mixed/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "chainsaw_jungle_001.wav" to "chainsaw_jungle_054.wav".
```

### AND


```python
nb_files = 171
filenames = [f"motosierra_jungla_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/chainsaw/mixed/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "motosierra_jungla_001.wav" to "motosierra_jungla_171.wav".
```

# Environment datasets information

## Bird

Filenames for the bird dataset can be fetched using the following pattern:

```python
nb_files = 201
filenames = [f"Birds_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/environment/birds/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "Birds_001.wav" to "Birds_201.wav".
```

## Jaguar

Filenames for the jaguar dataset can be fetched using the following pattern:

```python
nb_files = 122
filenames = [f"Jaguar_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/environment/jaguar/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "Jaguar_001.wav" to "Jaguar_122.wav".
```

## Monkey

Filenames for the monkey dataset can be fetched using the following pattern:

```python
nb_files = 100
filenames = [f"monkey_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/environment/monkey/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "monkey_001.wav" to "monkey_100.wav".
```

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
    with open(f"training/data/raw/environment/rainforest_ambience/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
``` This will fetch all dataset files from "digital_ambience_rainforest_001.wav" to "digital_ambience_rainforest_306.wav".```
```

## Rainforest atmosphere beta

Filenames for the rainforest atmosphere beta dataset can be fetched using the following pattern:

```python
nb_files = 72
filenames = [f"rainforest_ambience_beta_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/environment/rainforest_ambience_beta/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
```This will fetch all dataset files from 'rainforest_ambience_beta_001.wav' to 'rainforest_ambience_beta_072.wav'.```
```

## Snake

Filenames for the snake dataset can be fetched using the following pattern:

```python
nb_files = 60
filenames = [f"SNAKE_{i:03d}.wav" for i in range(1, nb_files + 1)]
for filename in filenames:
    with open(f"training/data/raw/environment/snake/{filename}", "r") as f:
        pass  # Create an empty file for demonstration purposes
```This will fetch all dataset files from 'SNAKE_001.wav' to 'SNAKE_060.wav'.```
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