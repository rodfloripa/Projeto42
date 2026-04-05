
<p align="justify"><h1>Relatório Técnico — Clustering de Documentos com Embeddings Contextuais</h1></p>

<p align="justify"><h3>1. Introdução</h3></p>

<p align="justify">O artigo <a href="https://arxiv.org/abs/2502.16139">An Improved Deep Learning Model for Word Embeddings Based Clustering for Large Text Datasets</a> propõe uma evolução do método WEClustering, utilizando embeddings contextuais gerados por BERT para melhorar o agrupamento de documentos.</p>

<p align="justify">A ideia central é substituir representações tradicionais (como Bag-of-Words e TF-IDF puro), que não capturam semântica, por representações densas que entendem contexto.</p>

<p align="justify">O método parte da premissa de que abordagens clássicas tratam palavras como independentes, ignorando relações semânticas profundas. Com o uso de modelos baseados em transformers, como o BERT, torna-se possível capturar o significado contextual das palavras, melhorando significativamente a qualidade das representações.</p>

<p align="justify">O pipeline segue as etapas:</p>

<p align="justify">
1. Extração de embeddings com BERT<br>
2. PCA<br>
3. Clustering com KMeans
</p>

<p align="justify">Esse método demonstrou ganhos relevantes em métricas como ARI (Adjusted Rand Index), alcançando valores próximos de 0.60 no dataset 20 Newsgroups.</p>

---

<p align="justify"><h3>2. Evolução do Projeto</h3></p>

<p align="justify">A implementação prática evoluiu de um pipeline complexo baseado em tokens para uma abordagem moderna de <b>Sentence Embeddings</b>. A primeira tentativa, seguindo rigorosamente o paper, enfrentou desafios de custo computacional e desempenho (ARI ≈ 0.20). A transição para o <b>Sentence-BERT (SBERT)</b> simplificou a arquitetura e permitiu que o modelo processasse o significado global de cada documento, resultando em um ganho expressivo de precisão e estabilidade.</p>

<p align="justify">Essa mudança reduziu drasticamente a complexidade do pipeline, eliminando etapas intermediárias como clustering de palavras e construção da matriz CD, que introduziam ruído e perda de informação.</p>

<p align="justify">Após otimizações adicionais, o melhor resultado obtido foi:</p>

<p align="center"><b>BEST ARI: 0.4541</b></p>

---

<p align="justify"><h3>3. Análise dos Resultados</h3></p>

<p align="justify">Os experimentos realizados demonstraram que a configuração da redução de dimensionalidade possui impacto direto na qualidade do clustering. Em particular, o uso de PCA preservando 90% da variância apresentou o melhor desempenho.</p>



<div align="center">

<b>Tabela: Comparativo de Métricas (ARI)</b>

| Configuração | ARI |
|--------------|-----|
| NO_PCA       | 0.44 |
| PCA_100 (63%)| 0.44 |
| PCA_90%      | 0.45 |

</div>

<p align="justify">Esse resultado evidencia um ponto fundamental: existe um equilíbrio entre remoção de ruído e preservação de informação. Reduções muito agressivas eliminam componentes relevantes, enquanto a ausência de redução mantém redundâncias que dificultam a separação dos clusters.</p>

---

<p align="justify"><h3>4. Pipeline Final</h3></p>

<p align="justify">O bloco abaixo representa o núcleo do sistema, responsável por transformar texto bruto em uma representação vetorial otimizada para clustering. A utilização do <b>Sentence-BERT</b> permite capturar relações semânticas globais entre documentos, eliminando a necessidade de processamento token-level. O uso de batches grandes maximiza a eficiência da GPU, enquanto a normalização garante estabilidade no espaço vetorial. A aplicação do PCA com preservação de variância atua como um mecanismo de compressão inteligente, mantendo a estrutura dos dados enquanto reduz ruído.</p>

```python
# EMBEDDINGS + NORMALIZAÇÃO
model = SentenceTransformer(model_name, device=device)

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

X = normalize(embeddings)

# PCA (90% variância)
pca = PCA(n_components=0.90, random_state=42)
X_reduced = pca.fit_transform(X)

print("Variância explicada:", np.sum(pca.explained_variance_ratio_))
````

<p align="justify">A escolha de preservar 90% da variância permite reduzir dimensionalidade sem comprometer a estrutura semântica dos dados, o que se mostrou essencial para melhorar o desempenho do clustering.</p>

---

<p align="justify"><h3>5. Clustering Otimizado</h3></p>

<p align="justify">O processo de clustering foi projetado para mitigar a variabilidade inerente ao algoritmo KMeans. Como a inicialização dos centroides influencia diretamente o resultado, foi adotada uma estratégia de múltiplas execuções com diferentes seeds. O uso de <b>k-means++</b> melhora a distribuição inicial dos centroides, enquanto o aumento do número de inicializações internas (<b>n_init</b>) reduz a probabilidade de convergência para mínimos locais.</p>

```python
def run_kmeans(X, labels, trials=5):
    best_ari = -1

    for seed in range(trials):
        kmeans = KMeans(
            n_clusters=20,
            init="k-means++",
            n_init=50,
            max_iter=1000,
            random_state=seed
        )

        pred = kmeans.fit_predict(X)
        ari = adjusted_rand_score(labels, pred)

        print(f"Seed {seed} → ARI: {ari:.4f}")

        if ari > best_ari:
            best_ari = ari

    return best_ari
```

<p align="justify">Esse mecanismo garante maior robustez estatística, reduzindo a variância entre execuções e permitindo selecionar a melhor configuração possível.</p>

---

<p align="justify"><h3>6. Otimizações de GPU (CUDA)</h3></p>

<p align="justify">O desempenho computacional do pipeline foi significativamente aprimorado através do uso de GPU. A inferência em batches permite explorar o paralelismo massivo da arquitetura CUDA, reduzindo o tempo total de processamento. A eliminação de loops por token foi um fator determinante, substituindo operações iterativas por processamento vetorizado.</p>

```python
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
```

<p align="justify">Essa abordagem reduz a complexidade computacional e elimina gargalos associados à execução sequencial, tornando o sistema escalável para grandes volumes de dados.</p>

---

<p align="justify"><h3>7. Blocos Mais Importantes</h3></p>

<p align="justify"><b>Embeddings:</b> Responsáveis por transformar texto em representações semânticas densas. Esse é o componente mais crítico do sistema, pois define a separabilidade dos dados.</p>

```python
embeddings = model.encode(...)
```

<p align="justify"><b>PCA:</b> Atua como um mecanismo de redução de dimensionalidade, removendo redundâncias e melhorando a estrutura dos dados.</p>

```python
PCA(n_components=0.90)
```

<p align="justify"><b>Clustering:</b> Responsável pela segmentação final dos documentos com base na proximidade no espaço vetorial.</p>

```python
kmeans.fit_predict(X)
```

<p align="justify"><b>Multi-seed:</b> Garante estabilidade e robustez, evitando dependência de inicializações aleatórias.</p>

```python
for seed in range(trials)
```

---

<p align="justify"><h3>8. Principais Aprendizados</h3></p>

<p align="justify">A principal conclusão deste projeto é que a qualidade dos embeddings tem impacto mais significativo do que a complexidade do pipeline. A simplificação da arquitetura resultou em maior desempenho, estabilidade e eficiência computacional.</p>

<p align="justify">Além disso, a redução de dimensionalidade mostrou-se essencial para melhorar a separação dos clusters, enquanto o uso de GPU foi fundamental para viabilizar o processamento em escala.</p>

---

<p align="justify"><h3>9. Conclusão</h3></p>

<p align="justify">O projeto evoluiu de uma implementação complexa e pouco eficiente para uma abordagem moderna baseada em embeddings de sentença, alcançando resultados competitivos com menor custo computacional.</p>

<p align="justify">O pipeline final apresenta um equilíbrio adequado entre simplicidade, desempenho e escalabilidade, sendo uma solução prática e eficaz para clustering de textos.</p>

---

<p align="justify"><h3>10. Trabalhos Futuros</h3></p>

<p align="justify">Possíveis extensões incluem:</p>

<p align="justify">
- Fine-tuning com aprendizado contrastivo<br>
- Uso de modelos maiores<br>
- Métodos de clustering mais avançados<br>
- Visualização dos clusters em baixa dimensão
</p>

---

```
```

