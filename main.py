import os
import math
import sys
import pickle
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS

_DLL_HANDLES = []


def configurar_dlls_nvidia_windows():
    """
    Em Windows, adiciona DLL dirs dos pacotes nvidia-* instalados no venv.
    Necessario para o ONNX Runtime carregar CUDA/cuDNN.
    """
    if os.name != "nt":
        return

    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    dll_dirs = [
        base / "cudnn" / "bin",
        base / "cublas" / "bin",
        base / "cuda_runtime" / "bin",
        base / "cufft" / "bin",
        base / "curand" / "bin",
        base / "cusolver" / "bin",
        base / "cusparse" / "bin",
        base / "nvjitlink" / "bin",
    ]
    for d in dll_dirs:
        if d.exists():
            _DLL_HANDLES.append(os.add_dll_directory(str(d)))


configurar_dlls_nvidia_windows()

try:
    import onnxruntime as ort

    # Forca preload de DLLs CUDA/cuDNN dos pacotes nvidia-* no site-packages.
    ort.preload_dlls(directory="")
except Exception:
    ort = None

import insightface

# pasta de imagens e saida
PASTA_IMAGENS = "fotos"
PASTA_SAIDA = "saida"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# parametros
MAX_SIDE = 1200
CROP_SIZE = 180
MARGIN = 0.18  # margem menor para o rosto preencher melhor o circulo
MASK_BLUR_KSIZE = 9  # odd number (7,9...) para anti-alias
MDS_N_INIT = 1
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
CACHE_VERSION = 2
CACHE_PATH = os.path.join(PASTA_SAIDA, "face_cache.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Mapa de similaridade facial")
    parser.add_argument("--images-dir", type=str, default=PASTA_IMAGENS)
    parser.add_argument("--query-image", type=str, default=None)
    parser.add_argument("--query-top-k", type=int, default=25)
    parser.add_argument("--face-mode", choices=["all", "largest", "best-score"], default="all")
    parser.add_argument("--min-det-score", type=float, default=0.0)
    parser.add_argument("--focus-person", type=str, default=None)
    parser.add_argument("--focus-top-k", type=int, default=30)
    parser.add_argument("--max-edges-per-node", type=int, default=1)
    parser.add_argument("--min-similarity-edge", type=float, default=0.0)
    parser.add_argument("--hover-min-similarity", type=float, default=0.25)
    parser.add_argument("--only-different-people", action="store_true", default=True)
    parser.add_argument("--person-separator", type=str, default="_")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--interactive", dest="interactive", action="store_true")
    parser.add_argument("--no-interactive", dest="interactive", action="store_false")
    parser.set_defaults(interactive=True)
    return parser.parse_args()


def carregar_imagem(path: str, max_side: int = MAX_SIDE):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Erro ao ler {path}")

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def recortar_quadrado_cover(
    img: np.ndarray,
    bbox,
    margem: float = MARGIN,
    tamanho: int = CROP_SIZE,
) -> np.ndarray:
    """
    Recorta um quadrado centrado no rosto, com margem,
    e redimensiona para (tamanho, tamanho).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    largura, altura = x2 - x1, y2 - y1

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(largura, altura) * (1.0 + 2.0 * margem)
    side = max(2.0, side)
    half = side * 0.5

    xs = int(math.floor(cx - half))
    ys = int(math.floor(cy - half))
    xe = int(math.ceil(cx + half))
    ye = int(math.ceil(cy + half))

    # Pad reflexivo para evitar bordas brancas quando crop sai da imagem.
    pad_left = max(0, -xs)
    pad_top = max(0, -ys)
    pad_right = max(0, xe - w)
    pad_bottom = max(0, ye - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img_pad = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        xs += pad_left
        xe += pad_left
        ys += pad_top
        ye += pad_top
    else:
        img_pad = img

    roi = img_pad[ys:ye, xs:xe]
    if roi.size == 0:
        roi = cv2.resize(img, (tamanho, tamanho), interpolation=cv2.INTER_AREA)
    else:
        roi = cv2.resize(roi, (tamanho, tamanho), interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)


def aplicar_mascara_circular_suave(img_rgb: np.ndarray, blur_ksize: int = MASK_BLUR_KSIZE) -> np.ndarray:
    """
    Recebe imagem RGB (HxWx3) e retorna RGBA com alpha circular anti-aliased.
    """
    h, w = img_rgb.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    r = min(w, h) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (dist <= r).astype(np.float32)
    mask_u8 = (mask * 255).astype(np.uint8)

    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    mask_blur = cv2.GaussianBlur(mask_u8, (k, k), 0)
    alpha_u8 = mask_blur.astype(np.uint8)
    return np.dstack((img_rgb, alpha_u8))


def min_pairwise_distance(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return 1.0

    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    d[d == 0] = np.nan
    m = np.nanmin(d)
    return float(m) if not np.isnan(m) else 1.0


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {"version": CACHE_VERSION, "model_name": MODEL_NAME, "det_size": DET_SIZE, "images": {}}
    try:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
    except Exception:
        return {"version": CACHE_VERSION, "model_name": MODEL_NAME, "det_size": DET_SIZE, "images": {}}

    if (
        cache.get("version") != CACHE_VERSION
        or cache.get("model_name") != MODEL_NAME
        or tuple(cache.get("det_size", ())) != tuple(DET_SIZE)
    ):
        return {"version": CACHE_VERSION, "model_name": MODEL_NAME, "det_size": DET_SIZE, "images": {}}
    cache.setdefault("images", {})
    return cache


def save_cache(cache):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def filter_faces(faces, face_mode: str, min_det_score: float):
    filtered = [f for f in faces if float(getattr(f, "det_score", 1.0)) >= min_det_score]
    if not filtered:
        return []
    if face_mode == "largest":
        return [max(filtered, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))]
    if face_mode == "best-score":
        return [max(filtered, key=lambda f: float(getattr(f, "det_score", 1.0)))]
    return filtered


def filter_cached_faces(cached_faces, face_mode: str, min_det_score: float):
    filtered = [f for f in cached_faces if float(f.get("det_score", 1.0)) >= min_det_score]
    if not filtered:
        return []
    if face_mode == "largest":
        return [max(filtered, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))]
    if face_mode == "best-score":
        return [max(filtered, key=lambda f: float(f.get("det_score", 1.0)))]
    return filtered


def abrir_arquivo(path: str):
    if not path:
        return
    try:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print("Falha ao abrir arquivo:", path, "|", e)


def extrair_face_consulta(app, query_path: str):
    img = carregar_imagem(query_path)
    faces = app.get(img)
    if len(faces) == 0:
        raise RuntimeError(f"Nenhum rosto encontrado em --query-image: {query_path}")
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = np.asarray(face.embedding, dtype=np.float32)
    crop_rgb = recortar_quadrado_cover(img, face.bbox, margem=MARGIN, tamanho=CROP_SIZE)
    crop_rgba = aplicar_mascara_circular_suave(crop_rgb, blur_ksize=MASK_BLUR_KSIZE)
    return emb, crop_rgba


def infer_person_id(filename: str, separator: str) -> str:
    stem = Path(filename).stem
    if separator and separator in stem:
        return stem.split(separator)[0]
    return stem


def build_edges(
    sim: np.ndarray,
    max_edges_per_node: int,
    min_similarity_edge: float,
    people_ids=None,
    only_different_people: bool = False,
):
    n = sim.shape[0]
    edge_set = set()
    if max_edges_per_node <= 0:
        return []
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        if only_different_people and people_ids is not None:
            same_person = np.array([pid == people_ids[i] for pid in people_ids], dtype=bool)
            row[same_person] = -np.inf
        order = np.argsort(row)[::-1]
        added = 0
        for j in order:
            if row[j] < min_similarity_edge:
                break
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            edge_set.add((a, b))
            added += 1
            if added >= max_edges_per_node:
                break
    return sorted(edge_set)


def gerar_relatorio_pessoas(rostos, sim: np.ndarray):
    person_to_indices = {}
    for idx, r in enumerate(rostos):
        person_to_indices.setdefault(r["pessoa"], []).append(idx)

    pessoas = sorted(person_to_indices.keys())
    linhas = []
    for pessoa in pessoas:
        idx_a = person_to_indices[pessoa]
        melhor = None
        for outra in pessoas:
            if outra == pessoa:
                continue
            idx_b = person_to_indices[outra]
            bloco = sim[np.ix_(idx_a, idx_b)]
            s = float(np.max(bloco))
            if melhor is None or s > melhor[1]:
                melhor = (outra, s)
        if melhor is not None:
            linhas.append((pessoa, melhor[0], melhor[1]))

    linhas.sort(key=lambda x: x[2], reverse=True)
    out_csv = os.path.join(PASTA_SAIDA, "similaridade_pessoas.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("pessoa,mais_parecida,similaridade\n")
        for p, q, s in linhas:
            f.write(f"{p},{q},{s:.6f}\n")

    print("Resumo de similaridade entre pessoas diferentes:")
    for p, q, s in linhas:
        print(f"  {p} -> {q}: {s:.3f}")
    print("Relatorio salvo em:", out_csv)
    return linhas


def get_hover_index(event, pos_norm: np.ndarray, node_half: float):
    if event.xdata is None or event.ydata is None:
        return None
    x, y = float(event.xdata), float(event.ydata)
    d = np.linalg.norm(pos_norm - np.array([x, y]), axis=1)
    idx = int(np.argmin(d))
    return idx if d[idx] <= node_half else None


def habilitar_hover_interativo(
    fig,
    ax,
    pos_norm: np.ndarray,
    node_half: float,
    sim: np.ndarray,
    rostos,
    min_similarity_hover: float,
):
    state = {"idx": None, "artists": [], "min_similarity": float(np.clip(min_similarity_hover, -1.0, 1.0))}

    def limpar_artistas():
        for art in state["artists"]:
            try:
                art.remove()
            except Exception:
                pass
        state["artists"] = []

    def desenhar_relacoes(idx: int):
        x0, y0 = pos_norm[idx]
        indices_ativos = {idx}
        highlight = Circle((x0, y0), radius=node_half * 1.04, fill=False, edgecolor="gold", linewidth=2.0, zorder=8)
        ax.add_patch(highlight)
        state["artists"].append(highlight)
        title = ax.text(
            0.02,
            0.98,
            f"Similaridade minima (hover): {state['min_similarity']:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            zorder=9,
            bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2", linewidth=0),
        )
        state["artists"].append(title)

        for j in range(sim.shape[0]):
            if j == idx:
                continue
            score = float(sim[idx, j])
            if score < state["min_similarity"]:
                continue
            indices_ativos.add(j)
            x1, y1 = pos_norm[j]
            dx, dy = x1 - x0, y1 - y0
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            ux, uy = dx / dist, dy / dist
            sx = x0 + ux * node_half
            sy = y0 + uy * node_half
            ex = x1 - ux * node_half
            ey = y1 - uy * node_half

            lw = 0.9 + 3.2 * max(0.0, score)
            line = ax.plot([sx, ex], [sy, ey], linewidth=lw, color=(0.05, 0.05, 0.05, 0.7), zorder=6)[0]
            state["artists"].append(line)

            tx, ty = (sx + ex) / 2.0, (sy + ey) / 2.0
            label = ax.text(
                tx,
                ty,
                f"{score:.3f}",
                fontsize=8,
                ha="center",
                va="center",
                zorder=9,
                bbox=dict(facecolor="white", alpha=0.75, boxstyle="round,pad=0.08", linewidth=0),
            )
            state["artists"].append(label)

        # Redesenha rostos ligados por cima, para evitar sobreposicao por outros rostos.
        for k in sorted(indices_ativos):
            xk, yk = pos_norm[k]
            left, right = xk - node_half, xk + node_half
            bottom, top = yk - node_half, yk + node_half
            img = rostos[k]["image_rgba"]
            face_overlay = ax.imshow(img, extent=(left, right, bottom, top), interpolation="bilinear", zorder=8)
            state["artists"].append(face_overlay)

        # Reforca destaque do rosto sob hover no topo.
        highlight_top = Circle((x0, y0), radius=node_half * 1.06, fill=False, edgecolor="gold", linewidth=2.3, zorder=10)
        ax.add_patch(highlight_top)
        state["artists"].append(highlight_top)

    def on_move(event):
        if event.inaxes != ax:
            if state["idx"] is not None:
                state["idx"] = None
                limpar_artistas()
                fig.canvas.draw_idle()
            return

        hovered = get_hover_index(event, pos_norm, node_half)
        if hovered == state["idx"]:
            return

        state["idx"] = hovered
        limpar_artistas()
        if hovered is not None:
            desenhar_relacoes(hovered)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        idx = get_hover_index(event, pos_norm, node_half)
        if idx is None:
            return
        path = rostos[idx].get("path")
        if path:
            print("Abrindo arquivo:", path)
            abrir_arquivo(path)

    def on_slider(_value):
        if state["idx"] is None:
            return
        limpar_artistas()
        desenhar_relacoes(state["idx"])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Slider para ajustar limiar de similaridade em tempo real.
    fig.subplots_adjust(bottom=0.10)
    ax_slider = fig.add_axes([0.20, 0.03, 0.60, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="Limiar Similaridade",
        valmin=-1.0,
        valmax=1.0,
        valinit=state["min_similarity"],
        valstep=0.01,
    )
    state["slider"] = slider

    def on_slider_changed(value):
        state["min_similarity"] = float(value)
        on_slider(value)

    slider.on_changed(on_slider_changed)


def inicializar_face_analysis():
    print("Verificando providers do onnxruntime e inicializando modelo...")
    use_gpu_requested = False

    try:
        providers = ort.get_available_providers()
        print("Providers onnxruntime disponiveis:", providers)

        if "CUDAExecutionProvider" in providers:
            app = insightface.app.FaceAnalysis(
                name=MODEL_NAME,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=DET_SIZE)
            use_gpu_requested = True
        else:
            app = insightface.app.FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=DET_SIZE)
            print("CUDAExecutionProvider nao disponivel, usando CPU.")
    except Exception as e:
        print("Falha ao inicializar InsightFace com GPU, fallback CPU:", e)
        app = insightface.app.FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=DET_SIZE)

    # Diagnostico: providers ativos por modelo.
    cuda_ativo = False
    try:
        for nome, model in app.models.items():
            sess = getattr(model, "session", None)
            if sess is not None:
                ps = sess.get_providers()
                print(f"Modelo {nome} providers ativos:", ps)
                if "CUDAExecutionProvider" in ps:
                    cuda_ativo = True
    except Exception:
        pass

    if cuda_ativo:
        print("InsightFace inicializado com GPU (CUDAExecutionProvider ativo).")
    elif use_gpu_requested:
        print("GPU foi solicitada, mas os modelos abriram em CPU. Verifique DLLs CUDA/cuDNN.")

    return app, cuda_ativo


args = parse_args()
app, use_gpu = inicializar_face_analysis()
cache = None if args.no_cache else load_cache()

# detectar rostos e extrair embeddings
files = sorted([f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
print("Arquivos encontrados:", len(files))

rostos = []
embeddings = []

for fn in files:
    path = os.path.join(args.images_dir, fn)
    try:
        img = carregar_imagem(path)
    except Exception as e:
        print("Erro lendo", fn, e)
        continue

    mtime_ns = os.stat(path).st_mtime_ns
    file_size = os.path.getsize(path)
    faces_data = None
    loaded_from_cache = False

    if cache is not None:
        entry = cache["images"].get(fn)
        if entry and entry.get("mtime_ns") == mtime_ns and entry.get("file_size") == file_size:
            faces_data = filter_cached_faces(entry.get("faces_all", []), args.face_mode, args.min_det_score)
            loaded_from_cache = True

    if faces_data is None:
        faces_detected = app.get(img)
        faces_all_data = []
        for face in faces_detected:
            faces_all_data.append(
                {
                    "bbox": [float(x) for x in face.bbox],
                    "det_score": float(getattr(face, "det_score", 1.0)),
                    "embedding": face.embedding.astype(np.float32),
                }
            )

        faces_data = filter_cached_faces(faces_all_data, args.face_mode, args.min_det_score)
        if len(faces_data) == 0:
            print("Nenhum rosto em", fn)
            if cache is not None:
                cache["images"][fn] = {"mtime_ns": mtime_ns, "file_size": file_size, "faces_all": faces_all_data}
            continue

        if cache is not None:
            cache["images"][fn] = {"mtime_ns": mtime_ns, "file_size": file_size, "faces_all": faces_all_data}

    if len(faces_data) == 0:
        print("Nenhum rosto em", fn)
        continue

    if loaded_from_cache:
        print(f"Usando cache em {fn}: {len(faces_data)} rosto(s)")

    for face_data in faces_data:
        emb = np.asarray(face_data["embedding"], dtype=np.float32)
        bbox = face_data["bbox"]
        crop_rgb = recortar_quadrado_cover(img, bbox, margem=MARGIN, tamanho=CROP_SIZE)
        crop_rgba = aplicar_mascara_circular_suave(crop_rgb, blur_ksize=MASK_BLUR_KSIZE)
        person_id = infer_person_id(fn, args.person_separator)
        rostos.append({"arquivo": fn, "pessoa": person_id, "path": path, "image_rgba": crop_rgba})
        embeddings.append(emb)

if cache is not None:
    save_cache(cache)

n = len(rostos)
print("Rostos detectados:", n)
if n < 2:
    raise SystemExit("Poucos rostos para comparar.")

emb_np = np.vstack(embeddings)

# Foco opcional para reduzir lotacao no grafico:
# mantém a pessoa alvo + top-k rostos mais similares a ela.
if args.focus_person:
    alvo = args.focus_person.strip().lower()
    idx_focus = [i for i, r in enumerate(rostos) if alvo in r["pessoa"].lower() or alvo in r["arquivo"].lower()]
    if not idx_focus:
        raise SystemExit(f"Nenhum rosto encontrado para --focus-person '{args.focus_person}'.")

    focus_emb = emb_np[idx_focus].mean(axis=0, keepdims=True)
    sim_focus = cosine_similarity(emb_np, focus_emb).ravel()
    idx_ordenados = np.argsort(sim_focus)[::-1]

    selected = set(idx_focus)
    for idx in idx_ordenados:
        selected.add(int(idx))
        if len(selected) >= len(idx_focus) + max(0, args.focus_top_k):
            break

    selected_idx = sorted(selected)
    rostos = [rostos[i] for i in selected_idx]
    emb_np = emb_np[selected_idx]
    print(
        f"Modo foco ativo: '{args.focus_person}' | rostos da pessoa: {len(idx_focus)} | "
        f"total exibido: {len(selected_idx)}"
    )

    n = len(rostos)
    if n < 2:
        raise SystemExit("Poucos rostos para comparar apos filtro de foco.")

# Busca por rosto de consulta (imagem externa).
if args.query_image:
    query_emb, query_rgba = extrair_face_consulta(app, args.query_image)
    sim_query = cosine_similarity(emb_np, query_emb.reshape(1, -1)).ravel()
    order = np.argsort(sim_query)[::-1]
    top_k = max(1, int(args.query_top_k))
    selected_idx = [int(i) for i in order[:top_k]]

    rostos = [rostos[i] for i in selected_idx]
    emb_np = emb_np[selected_idx]
    sim_query = sim_query[selected_idx]

    rostos.insert(
        0,
        {
            "arquivo": f"QUERY: {Path(args.query_image).name}",
            "pessoa": "query",
            "path": os.path.abspath(args.query_image),
            "image_rgba": query_rgba,
        },
    )
    emb_np = np.vstack([query_emb.reshape(1, -1), emb_np])

    out_query = os.path.join(PASTA_SAIDA, "resultado_query.csv")
    with open(out_query, "w", encoding="utf-8") as f:
        f.write("rank,arquivo,similaridade\n")
        for rank, (r, s) in enumerate(zip(rostos[1:], sim_query), start=1):
            f.write(f"{rank},{r['arquivo']},{float(s):.6f}\n")

    print(f"Busca por imagem ativa: {Path(args.query_image).name} | top_k={top_k}")
    print("Ranking salvo em:", out_query)

    n = len(rostos)
    if n < 2:
        raise SystemExit("Poucos rostos para comparar apos filtro por query.")

sim = cosine_similarity(emb_np)
dist = 1.0 - sim
people_ids = [r["pessoa"] for r in rostos]
gerar_relatorio_pessoas(rostos, sim)

# MDS para conservar distancias
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=MDS_N_INIT, max_iter=300)
pos = mds.fit_transform(dist)

# normalizar posicoes para 0..1
min_xy = pos.min(axis=0)
max_xy = pos.max(axis=0)
span = max_xy - min_xy
span[span == 0] = 1.0
pos_norm = (pos - min_xy) / span

# calcular tamanho automatico do no
min_dist = min_pairwise_distance(pos_norm)
node_half = float(np.clip(0.35 * min_dist, 0.03, 0.10))
print(f"min_dist={min_dist:.4f}, node_half={node_half:.4f}")

# desenhar
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_title("Mapa de Similaridade Facial", fontsize=16)
ax.set_aspect("equal")
ax.axis("off")

# desenhar linhas estaticas (quando nao interativo)
if not args.interactive:
    edges = build_edges(
        sim,
        args.max_edges_per_node,
        args.min_similarity_edge,
        people_ids=people_ids,
        only_different_people=args.only_different_people,
    )
    print("Arestas desenhadas:", len(edges))
    for i, j in edges:
            x1, y1 = pos_norm[int(i)]
            x2, y2 = pos_norm[int(j)]
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy)
            if d == 0:
                continue

            ux, uy = dx / d, dy / d
            sx = x1 + ux * node_half
            sy = y1 + uy * node_half
            ex = x2 - ux * node_half
            ey = y2 - uy * node_half

            w = float(sim[int(i), int(j)])
            linewidth = 0.6 + 3.0 * max(0.0, w)
            ax.plot([sx, ex], [sy, ey], linewidth=linewidth, color=(0.15, 0.15, 0.15, 0.55), zorder=1)

            tx, ty = (sx + ex) / 2.0, (sy + ey) / 2.0
            ax.text(
                tx,
                ty,
                f"{w:.3f}",
                fontsize=7,
                ha="center",
                va="center",
                zorder=2,
                bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.08", linewidth=0),
            )

# desenhar imagens
for i in range(n):
    x, y = pos_norm[i]
    img_rgba = rostos[i]["image_rgba"]
    left, right = x - node_half, x + node_half
    bottom, top = y - node_half, y + node_half

    ax.imshow(img_rgba, extent=(left, right, bottom, top), interpolation="bilinear", zorder=3)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

out_path = os.path.join(PASTA_SAIDA, "mapa_distancias_fillcircle.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Mapa salvo em:", out_path)

if args.interactive:
    print("Modo interativo ativo: passe o mouse sobre um rosto para ver as similaridades.")
    habilitar_hover_interativo(fig, ax, pos_norm, node_half, sim, rostos, args.hover_min_similarity)
    plt.show()
else:
    plt.close(fig)
