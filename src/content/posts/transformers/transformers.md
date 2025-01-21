---
title: Transformers Quickstart
published: 2025-01-21
tags: [Artigos]
category: Artigos
draft: false
---

Essa é uma traduçao (com pequenos ajustes) do material [transformers quicktour](https://huggingface.co/docs/transformers/quicktour) do Hugging Face.

## Introdução

Este post mostrará como usar o ```pipeline()``` para inferência, como carregar um modelo pré-treinado e um pré-processador e treinar um modelo com **PyTorch**.

Antes de começar, certifique-se de ter todas as bibliotecas necessárias instaladas:

```bash
pip install transformers datasets evaluate accelerate torch
```

O ```pipeline()``` é a maneira mais fácil e rápida de usar um modelo pré-treinado para inferência.
Você pode usar o ```pipeline()``` pronto para uso para muitas Tarefas (Tabela 1) em diferentes modalidades, algumas das quais são mostradas na tabela abaixo:

<center>Tabela 1 - Tarefas possíveis com o pipeline do Transformers.</center>

|Descriçao da Tarefa|Identificador do Pipeline|
|----|----|
|Análise de sentimento|pipeline(task=“sentiment-analysis”)|
|Geração de Texto|pipeline(task=“text-generation”)|
|reconhecimento automático de fala|pipeline(task=“automatic-speech-recognition”)|

## Exemplo análise de sentimento

Comece criando uma instância de ```pipeline()``` e especificando uma tarefa para a qual você deseja usá-lo.
O ```pipeline()``` baixa e armazena em cache um modelo pré-treinado padrão e um tokenizador para análise de sentimento.
Agora você pode usar o classificador no seu texto de destino.
Neste guia, você usará o ```pipeline()``` para análise de sentimentos como um exemplo:

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
classifier("We are very happy to show you the Transformers library.")

Saída: 
    [{'label': 'POSITIVE', 'score': 0.9997795224189758}]
```

Se você tiver mais de uma entrada, passe suas entradas como uma lista para o ```pipeline()``` para retornar uma lista de dicionários:

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
results = classifier(["We are very happy to show you the Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

Saída:
    label: POSITIVE, with score: 0.9998
    label: NEGATIVE, with score: 0.5309
```

## Exemplo reconhecimento automático de fala

O ```pipeline()``` também pode iterar um conjunto de dados inteiro para qualquer tarefa que você desejar. Para este exemplo, vamos escolher o **reconhecimento automático de fala** como nossa tarefa:

```python
import torch
from transformers import pipeline
from datasets import load_dataset, Audio

speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# Carregue um conjunto de dados de áudio que você gostaria de iterar. 
# Por exemplo, carregue o conjunto de dados MInDS-14.
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

# You need to make sure the sampling rate of the dataset 
# matches the sampling rate facebook/wav2vec2-base-960h was trained on:
dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

# The audio files are automatically loaded and resampled when calling the "audio" column. 
# Extract the raw waveform arrays from the first 4 samples and pass it as a list to the pipeline:
result = speech_recognizer(dataset[:4]["audio"])
print([d["text"] for d in result])
```

For larger datasets where the inputs are big (like in speech or vision), you’ll want to pass a generator instead of a list to load all the inputs in memory. Take a look at the pipeline API reference for more information.
