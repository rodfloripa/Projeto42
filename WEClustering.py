

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# ==============================
# 1. CONFIG
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 TESTE MODELOS AQUI
model_name = "sentence-transformers/all-mpnet-base-v2"
# model_name = "sentence-transformers/gtr-t5-base"  # mais forte (se tiver GPU boa)

n_clusters = 20
batch_size = 64
kmeans_trials = 5

print("Device:", device)
print("Modelo:", model_name)

# ==============================
# 2. DATASET
# ==============================
print("Carregando dataset...")
newsgroups = fetch_20newsgroups(
    subset='all',
    remove=('headers','footers','quotes')
)

valid_idx = [i for i, t in enumerate(newsgroups.data) if len(t.strip()) > 100]
texts = [newsgroups.data[i] for i in valid_idx]
real_labels = [newsgroups.target[i] for i in valid_idx]

print(f"Total de documentos: {len(texts)}")

# ==============================
# 3. EMBEDDINGS (GPU)
# ==============================
print("Carregando modelo...")
model = SentenceTransformer(model_name, device=device)

print("Gerando embeddings...")
embeddings = model.encode(
    texts,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# normalização extra
X = normalize(embeddings)

# ==============================
# 4. FUNÇÃO KMEANS
# ==============================
def run_kmeans(X, labels, trials=5):
    best_ari = -1

    for seed in range(trials):
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=50,
            max_iter=1000,
            random_state=seed
        )

        pred = kmeans.fit_predict(X)
        ari = adjusted_rand_score(labels, pred)

        print(f"   Seed {seed} → ARI: {ari:.4f}")

        if ari > best_ari:
            best_ari = ari

    return best_ari

# ==============================
# 5. TESTE DE CONFIGURAÇÕES
# ==============================
configs = [
    ("NO_PCA", None),
    ("PCA_100", PCA(n_components=100, random_state=42)),
    ("PCA_90%", PCA(n_components=0.90, random_state=42)),
]

results = {}

print("\nTestando configurações...\n")

for name, pca in configs:
    print(f"=== {name} ===")

    if pca:
        X_test = pca.fit_transform(X)
        print(f"Variância explicada: {np.sum(pca.explained_variance_ratio_):.4f}")
    else:
        X_test = X

    ari = run_kmeans(X_test, real_labels, trials=kmeans_trials)
    results[name] = ari

    print(f"➡️ BEST ARI ({name}): {ari:.4f}\n")

# ==============================
# 6. RESULTADO FINAL
# ==============================
best_config = max(results, key=results.get)

print("\n==============================")
print("RESULTADOS FINAIS:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

print("\n🏆 MELHOR CONFIG:", best_config)
print("🏆 BEST ARI:", results[best_config])
print("==============================")
              
""" 
Device: cuda
Modelo: sentence-transformers/all-mpnet-base-v2
Carregando dataset...
Total de documentos: 17056
Carregando modelo...

Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]

MPNetModel LOAD REPORT from: sentence-transformers/all-mpnet-base-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

Gerando embeddings...

Batches:   0%|          | 0/267 [00:00<?, ?it/s]

Testando configurações...

=== NO_PCA ===
   Seed 0 → ARI: 0.4263
   Seed 1 → ARI: 0.4442
   Seed 2 → ARI: 0.4415
   Seed 3 → ARI: 0.4075
   Seed 4 → ARI: 0.4430
➡️ BEST ARI (NO_PCA): 0.4442

=== PCA_100 ===
Variância explicada: 0.6367
   Seed 0 → ARI: 0.4274
   Seed 1 → ARI: 0.4281
   Seed 2 → ARI: 0.4424
   Seed 3 → ARI: 0.4418
   Seed 4 → ARI: 0.4426
➡️ BEST ARI (PCA_100): 0.4426

=== PCA_90% ===
Variância explicada: 0.9002
   Seed 0 → ARI: 0.4541
   Seed 1 → ARI: 0.4411
   Seed 2 → ARI: 0.4254
   Seed 3 → ARI: 0.4475
   Seed 4 → ARI: 0.4494
➡️ BEST ARI (PCA_90%): 0.4541


==============================
RESULTADOS FINAIS:
NO_PCA: 0.4442
PCA_100: 0.4426
PCA_90%: 0.4541

🏆 MELHOR CONFIG: PCA_90%
🏆 BEST ARI: 0.45410041667456513
==============================
"""


