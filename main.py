import os
import cv2
import numpy as np
import insightface
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

PASTA_IMAGENS = "fotos"
PASTA_SAIDA = "saida"
THRESHOLD_RECONHECIMENTO = 0.55  # mais tolerante para reconhecer

os.makedirs(PASTA_SAIDA, exist_ok=True)

app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

rostos = []
embeddings = []

print("Detectando rostos...")

for arquivo in os.listdir(PASTA_IMAGENS):
    caminho = os.path.join(PASTA_IMAGENS, arquivo)

    if not caminho.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img = cv2.imread(caminho)
    faces = app.get(img)

    if len(faces) == 0:
        continue

    nome_base = os.path.splitext(arquivo)[0]

    for face in faces:
        emb = face.embedding

        # recortar rosto
        x1, y1, x2, y2 = map(int, face.bbox)
        rosto_crop = img[y1:y2, x1:x2]
        rosto_crop = cv2.cvtColor(rosto_crop, cv2.COLOR_BGR2RGB)

        if not nome_base.lower().startswith("grupo"):
            nome = nome_base
        else:
            nome = f"Desconhecido_{len(rostos)}"

        rostos.append({
            "nome": nome,
            "embedding": emb,
            "imagem": rosto_crop
        })

        embeddings.append(emb)

print(f"{len(rostos)} rostos detectados.\n")

# ----------------------------
# MATRIZ DE SIMILARIDADE
# ----------------------------

matriz = cosine_similarity(embeddings)

# ----------------------------
# RENOMEAR DESCONHECIDOS
# ----------------------------

for i in range(len(rostos)):
    if not rostos[i]["nome"].startswith("Desconhecido"):
        continue

    melhor_sim = 0
    melhor_nome = None

    for j in range(len(rostos)):
        if i == j:
            continue

        nome_j = rostos[j]["nome"]

        if nome_j.startswith("Desconhecido"):
            continue

        sim = matriz[i][j]

        if sim > melhor_sim:
            melhor_sim = sim
            melhor_nome = nome_j

    if melhor_sim > THRESHOLD_RECONHECIMENTO:
        rostos[i]["nome"] = melhor_nome

nomes = [r["nome"] for r in rostos]

# ----------------------------
# CRIAR GRAFO
# ----------------------------

G = nx.Graph()

for i, r in enumerate(rostos):
    G.add_node(i, label=r["nome"], image=r["imagem"])

for i in range(len(rostos)):
    for j in range(i + 1, len(rostos)):
        sim = matriz[i][j]
        if sim > 0.40:
            G.add_edge(i, j, weight=sim)

pos = nx.spring_layout(G, seed=42, weight="weight")

# ----------------------------
# DESENHAR COM ROSTOS
# ----------------------------

fig, ax = plt.subplots(figsize=(10, 8))

for node in G.nodes:
    (x, y) = pos[node]
    img = G.nodes[node]['image']

    imagebox = plt.imshow(img, extent=(x-0.05, x+0.05, y-0.05, y+0.05))
    ax.text(x, y-0.07, G.nodes[node]['label'], ha='center')

# desenhar conexões
for (u, v, d) in G.edges(data=True):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    ax.plot([x1, x2], [y1, y2])

    peso = d["weight"]
    ax.text((x1+x2)/2, (y1+y2)/2, f"{peso:.2f}", fontsize=8)

ax.set_title("Mapa de proximidade facial")
ax.axis("off")

# salvar
saida_path = os.path.join(PASTA_SAIDA, "grafo.png")
plt.savefig(saida_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nGrafo salvo em: {saida_path}")