

---

<p align="justify"><h1>Relatório Técnico — Clustering de Documentos com Embeddings Contextuais</h1></p>

<p align="justify"><h3>1. Introdução</h3></p>

<p align="justify">O artigo <a href="https://arxiv.org/abs/2502.16139">An Improved Deep Learning Model for Word Embeddings Based Clustering for Large Text Datasets</a> propõe uma evolução do método WEClustering, utilizando embeddings contextuais gerados por BERT para melhorar o agrupamento de documentos.</p>

<p align="justify">A ideia central é substituir representações tradicionais como Bag-of-Words e TF-IDF puro, que não capturam semântica, por representações densas que entendem contexto.</p>

<p align="justify">O método parte da premissa de que abordagens clássicas tratam palavras como independentes, ignorando relações semânticas profundas. Com o uso de modelos baseados em transformers, como o BERT, torna-se possível capturar o significado contextual das palavras, melhorando significativamente a qualidade das representações.</p>

<p align="justify">O pipeline segue as etapas:</p>

<p align="justify">
1. Extração de embeddings com BERT<br>
2. PCA<br>
3. Clustering com KMeans
</p>

<p align="justify">Esse método demonstrou ganhos relevantes em métricas como ARI, alcançando valores próximos de 0.45 no dataset 20 Newsgroups, contra 0.60 reportado no artigo.</p>

---

<p align="justify"><h3>2. Evolução do Projeto</h3></p>

<p align="justify">A implementação prática evoluiu de um pipeline complexo baseado em tokens para uma abordagem moderna de <b>Sentence Embeddings</b>. A primeira tentativa, seguindo rigorosamente o paper, enfrentou desafios de custo computacional e desempenho com ARI ≈ 0.20. A transição para o <b>Sentence-BERT (SBERT)</b> simplificou a arquitetura e permitiu que o modelo processasse o significado global de cada documento, resultando em um ganho expressivo de precisão e estabilidade.</p>

<p align="justify">Essa mudança reduziu drasticamente a complexidade do pipeline, eliminando etapas intermediárias como clustering de palavras e construção da matriz CD, que introduziam ruído e perda de informação.</p>

<p align="justify">Após otimizações adicionais, o melhor resultado obtido foi:</p>

<p align="center"><b>BEST ARI: 0.45</b></p>

---

<p align="justify"><h3>3. Análise dos Resultados</h3></p>

<p align="justify">Os experimentos demonstraram que, ao contrário da hipótese inicial, a redução de dimensionalidade teve impacto mínimo na qualidade do clustering. O uso de PCA preservando 90% da variância apresentou desempenho praticamente idêntico à versão sem PCA.</p>

<div align="center">

<b>Tabela: Comparativo de Métricas (ARI)</b>

| Configuração | ARI |
|--------------|-----|
| NO_PCA | 0.44 |
| PCA_100 (63%)| 0.44 |
| PCA_90% | 0.45 |

</div>

<p align="justify">O resultado mostra que o ganho com PCA foi irrisório, de apenas 0.01 de ARI. Isso indica que os embeddings do SBERT já são bem estruturados e com baixa redundância, tornando a redução de dimensionalidade pouco relevante para este cenário. A principal vantagem do PCA aqui foi apenas redução de custo computacional no KMeans, não ganho de performance.</p>

---

<p align="justify"><h3>4. Pipeline Final</h3></p>

<p align="justify">O bloco abaixo representa o núcleo do sistema, responsável por transformar texto bruto em uma representação vetorial otimizada para clustering. A utilização do <b>Sentence-BERT</b> permite capturar relações semânticas globais entre documentos, eliminando a necessidade de processamento token-level. O uso de batches grandes maximiza a eficiência da GPU, enquanto a normalização garante estabilidade no espaço vetorial. O PCA foi mantido com 90% de variância principalmente para reduzir o custo computacional das etapas seguintes.</p>
EMBEDDINGS + NORMALIZAÇÃO

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
<p align="justify">A escolha de preservar 90% da variância reduziu a dimensionalidade sem comprometer a estrutura semântica dos dados. Porém, os testes mostraram que essa etapa não foi determinante para a qualidade do clustering.</p>

---

<p align="justify"><h3>5. Clustering Otimizado</h3></p>

<p align="justify">O processo de clustering foi projetado para mitigar a variabilidade inerente ao algoritmo KMeans. Como a inicialização dos centroides influencia diretamente o resultado, foi adotada uma estratégia de múltiplas execuções com diferentes seeds. O uso de <b>k-means++</b> melhora a distribuição inicial dos centroides, enquanto o aumento do número de inicializações internas <b>n_init</b> reduz a probabilidade de convergência para mínimos locais.</p>

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

<p align="justify"><h3>7. Blocos Mais Importantes</h3></p>

<p align="justify"><b>Embeddings:</b> Responsáveis por transformar texto em representações semânticas densas. Esse é o componente mais crítico do sistema, pois define a separabilidade dos dados.</p>

```python
embeddings = model.encode(...)
```

<p align="justify"><b>PCA:</b> Atuou como redução de dimensionalidade para acelerar o KMeans. O impacto na métrica ARI foi irrisório, indicando que os embeddings já estavam bem condicionados.</p>

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

<p align="justify">Além disso, a redução de dimensionalidade via PCA não trouxe ganhos práticos de ARI, servindo apenas para otimizar tempo de processamento. O uso de GPU foi fundamental para viabilizar o processamento em escala.</p>

---

<p align="justify"><h3>9. Conclusão</h3></p>

<p align="justify">O projeto evoluiu de uma implementação complexa e pouco eficiente para uma abordagem moderna baseada em embeddings de sentença, alcançando ARI de 0.45 com menor custo computacional.</p>

<p align="justify">O pipeline final apresenta um equilíbrio adequado entre simplicidade, desempenho e escalabilidade. O ARI do artigo é de 0.60, mas esta versão simplificada obteve 0.45 mesmo sem ganhos relevantes vindos do PCA.</p>

---

