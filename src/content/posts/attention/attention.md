---
title: Attention is all you need
published: 2025-01-20
tags: [Artigos]
category: Artigos
draft: false
---

Título: Attention Is All You Need

Tradução do artigo com google tradutor

[Download do Original](https://arxiv.org/abs/1706.03762)

Autores:

- Ashish Vaswani <avaswani@google.com>
- Noam Shazeer <noam@google.com>
- Niki Parmar <nikip@google.com>
- Jakob Uszkoreit <usz@google.com>
- Llion Jones <llion@google.com>
- Aidan N. Gomez <aidan@cs.toronto.edu>
- Łukasz Kaiser <lukaszkaiser@google.com>
- Illia Polosukhin <illia.polosukhin@gmail.com>

## Vídeo resources

<iframe width="100%" height="468" src="https://www.youtube.com/embed/wjZofJX0v4M" title="Transformers (how LLMs work) explained visually | DL5" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## abstract

:::tip[Original]
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English- to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
:::

:::note[Tradução]
Os modelos de **transduction** de sequência dominante são baseados em redes neurais recorrentes ou convolucionais complexas que incluem um codificador e um decodificador. Os modelos de melhor desempenho também conectam o codificador e o decodificador por meio de um mecanismo de atenção. Propomos uma nova arquitetura de rede simples, o Transformer, baseada somente em mecanismos de atenção, dispensando totalmente a recorrência e as convoluções. Experimentos em duas tarefas de tradução automática mostram que esses modelos são superiores em qualidade, ao mesmo tempo em que são mais paralelizáveis ​​e exigem significativamente menos tempo para treinar. Nosso modelo atinge 28,4 BLEU na tarefa de tradução do inglês para o alemão do WMT 2014, melhorando os melhores resultados existentes, incluindo conjuntos, em mais de 2 BLEU. Na tarefa de tradução do inglês para o francês do WMT 2014, nosso modelo estabelece uma nova pontuação BLEU de última geração de modelo único de 41,8 após treinamento por 3,5 dias em oito GPUs, uma pequena fração dos custos de treinamento dos melhores modelos da literatura. Mostramos que o Transformer generaliza bem para outras tarefas, aplicando-o com sucesso à análise de constituintes ingleses, tanto com dados de treinamento grandes quanto limitados.
:::

## Introdução

:::tip[Original]
Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].
:::

:::note[Tradução]
Redes neurais recorrentes, memória de curto prazo longa [13] e redes neurais recorrentes com portas [7] em particular, foram firmemente estabelecidas como abordagens de última geração em modelagem de sequências e problemas de transdução, como modelagem de linguagem e tradução automática [35, 2, 5]. Desde então, vários esforços continuaram a expandir os limites dos modelos de linguagem recorrente e arquiteturas de codificador-decodificador [38, 24, 15].
:::

:::tip[Original]
Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.
:::

:::note[Tradução]
Modelos recorrentes tipicamente fatoram a computação ao longo das posições de símbolo das sequências de entrada e saída. Alinhando as posições a passos no tempo de computação, eles geram uma sequência de estados ocultos $ht$, como uma função do estado oculto anterior $ht−1$ e a entrada para a posição $t$. Essa natureza inerentemente sequencial impede a paralelização dentro de exemplos de treinamento, o que se torna crítico em comprimentos de sequência maiores, pois as restrições de memória limitam o loteamento entre os exemplos. Trabalhos recentes alcançaram melhorias significativas na eficiência computacional por meio de truques de fatoração [21] e computação condicional [32], ao mesmo tempo em que melhoram o desempenho do modelo no caso do último. A restrição fundamental da computação sequencial, no entanto, permanece.
:::

:::tip[Original]
Attention mechanisms have become an integral part of compelling sequence modeling and transduc- tion models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network. In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
:::

:::note[Tradução]
Os mecanismos de atenção tornaram-se parte integrante da modelagem de sequências convincentes e modelos de transdução em várias tarefas, permitindo a modelagem de dependências sem levar em conta sua distância nas sequências de entrada ou saída [2, 19]. Em todos os casos, exceto alguns [27], no entanto, esses mecanismos de atenção são usados ​​em conjunto com uma rede recorrente.
:::

:::tip[Original]
In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
:::

:::note[Tradução]
Neste trabalho, propomos o Transformer, uma arquitetura de modelo que evita recorrência e, em vez disso, depende inteiramente de um mecanismo de atenção para desenhar dependências globais entre entrada e saída. O Transformer permite significativamente mais paralelização e pode atingir um novo estado da arte em qualidade de tradução após ser treinado por apenas doze horas em oito GPUs P100.
:::

## Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence- aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34]. To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].
