import os
import cv2
import numpy as np
import insightface
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS

PASTA_IMAGENS = "fotos"
PASTA_SAIDA = "saida"
os.makedirs(PASTA_SAIDA, exist_ok=True)

print("Carregando modelo...")

# tenta GPU primeiro
try:
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    print("GPU ativa")
except:
    app.prepare(ctx_id=-1)
    print("Rodando em CPU")

rostos = []
embeddings = []

def recortar_rosto(img, bbox, margem=0.35, tamanho=160):
    h, w, _ = img.shape
    x1, y1, x2, y2 = map(int, bbox)

    largura = x2 - x1
    altura = y2 - y1

    x1 = max(0, int(x1 - largura * margem))
    y1 = max(0, int(y1 - altura * margem))
    x2 = min(w, int(x2 + largura * margem))
    y2 = min(h, int(y2 + altura * margem))

    rosto = img[y1:y2, x1:x2]

    size = max(rosto.shape[:2])
    square = np.ones((size, size, 3), dtype=np.uint8) * 255

    y_offset = (size - rosto.shape[0]) // 2
    x_offset = (size - rosto.shape[1]) // 2
    square[y_offset:y_offset+rosto.shape[0], x_offset:x_offset+rosto.shape[1]] = rosto

    square = cv2.resize(square, (tamanho, tamanho))
    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)

    return square

print("Detectando rostos...")

for arquivo in os.listdir(PASTA_IMAGENS):
    caminho = os.path.join(PASTA_IMAGENS, arquivo)

    if not caminho.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img = cv2.imread(caminho)
    faces = app.get(img)

    for face in faces:
        rosto_crop = recortar_rosto(img, face.bbox)

        rostos.append({
            "imagem": rosto_crop,
            "arquivo": arquivo
        })

        embeddings.append(face.embedding)

print(f"{len(rostos)} rostos detectados")

embeddings = np.array(embeddings)

# similaridade
similaridade = cosine_similarity(embeddings)

# distância real
distancias = 1 - similaridade

# MDS preserva distância
mds = MDS(dissimilarity='precomputed', random_state=42)
posicoes = mds.fit_transform(distancias)

# normalizar posições
posicoes = (posicoes - posicoes.min()) / (posicoes.max() - posicoes.min())

fig, ax = plt.subplots(figsize=(12, 10))

tamanho = 0.05

# desenhar rostos
for i, rosto in enumerate(rostos):
    x, y = posicoes[i]
    img = rosto["imagem"]

    ax.imshow(img, extent=(x-tamanho, x+tamanho, y-tamanho, y+tamanho))

# desenhar distâncias
for i in range(len(rostos)):
    for j in range(i+1, len(rostos)):
        sim = similaridade[i][j]

        x1, y1 = posicoes[i]
        x2, y2 = posicoes[j]

        ax.plot([x1, x2], [y1, y2], linewidth=0.5, alpha=0.3)
        ax.text((x1+x2)/2, (y1+y2)/2, f"{sim:.2f}", fontsize=8)

ax.set_title("Mapa de Similaridade Facial (distância real)")
ax.axis("off")

saida = os.path.join(PASTA_SAIDA, "mapa_distancias.png")
plt.savefig(saida, dpi=300, bbox_inches="tight")
plt.show()

print("Mapa salvo em:", saida)
