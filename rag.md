# Region Adjacency Graph (RAG)

> *"Simplify, simplify."* — Henry David Thoreau

---

## Mục lục

1. [RAG là gì?](#1-rag-là-gì)
2. [Tại sao cần RAG?](#2-tại-sao-cần-rag)
3. [Cấu trúc của RAG](#3-cấu-trúc-của-rag)
4. [Thuật toán xây dựng RAG](#4-thuật-toán-xây-dựng-rag)
5. [Ứng dụng thực tế](#5-ứng-dụng-thực-tế)
6. [Hiệu quả và giới hạn](#6-hiệu-quả-và-giới-hạn)
7. [Implement Python — scikit-image](#7-implement-python--scikit-image)
8. [Demo tương tác — HTML / CSS / JS](#8-demo-tương-tác--html--css--js)
9. [Những điều dev cần biết thêm](#9-những-điều-dev-cần-biết-thêm)
10. [Giải thích thuật ngữ](#10-giải-thích-thuật-ngữ)
11. [Tài liệu tham khảo](#11-tài-liệu-tham-khảo)

---

## 1. RAG là gì?

**Region Adjacency Graph (RAG)** là một cấu trúc đồ thị dùng để biểu diễn **quan hệ không gian giữa các vùng** trong một ảnh đã qua phân đoạn (image segmentation).

Định nghĩa hình thức:

```
G = (V, E)
```

| Ký hiệu | Ý nghĩa |
|---------|---------|
| `V` (vertices/nodes) | Mỗi node đại diện cho một vùng (region) — một cụm pixel có đặc trưng tương đồng (màu sắc, texture, cường độ...) |
| `E` (edges) | Một cạnh nối hai node nếu hai vùng tương ứng **tiếp giáp nhau** về mặt không gian trong ảnh |
| `w(u, v)` | **Weight** của cạnh — thường là độ chênh lệch màu trung bình (mean color difference) giữa hai vùng |

RAG là đồ thị **vô hướng có trọng số** (undirected weighted graph) và kế thừa từ `networkx.Graph`, nên toàn bộ API của NetworkX đều dùng được.

### Ví dụ trực quan

Cho ảnh 4 vùng màu:

```
+-------+-------+
|       |       |
|  R1   |  R2   |
| (đỏ)  | (xanh)|
+-------+-------+
|       |       |
|  R3   |  R4   |
|(vàng) | (tím) |
+-------+-------+
```

RAG tương ứng:

```
R1 --[w=120]-- R2
|               |
[w=80]        [w=95]
|               |
R3 --[w=60]-- R4
```

Trong đó weight `w` là độ chênh lệch màu giữa hai vùng kề nhau. Weight **càng nhỏ** → hai vùng **càng giống nhau** → ứng viên tốt để gộp lại.

---

## 2. Tại sao cần RAG?

### Vấn đề của raw pixel processing

Xử lý ảnh trực tiếp trên pixel có chi phí tính toán rất cao:

```
Ảnh 800×600 = 480,000 pixels
Mỗi pixel cần so sánh với 8 neighbors = ~3,840,000 phép tính
```

Ngoài ra, pixel-level analysis dễ bị nhiễu (noise) và không nắm bắt được cấu trúc ngữ nghĩa của ảnh.

### RAG giải quyết bài toán đó như thế nào?

```
Bước 1: Segmentation (SLIC, Watershed, Felzenszwalb...)
         480,000 pixels → ~400 superpixels

Bước 2: Xây dựng RAG
         400 nodes, ~800 edges (thay vì 3.8M phép tính)

Bước 3: Graph operations (merge, threshold, cut...)
         Chỉ làm việc trên graph, không đụng pixel
```

**RAG là bước trừu tượng hóa**: biến bài toán pixel thành bài toán graph, giảm độ phức tạp từ `O(n_pixels)` xuống `O(n_regions)` — thường nhỏ hơn hàng trăm lần.

### So sánh trực tiếp

| Tiêu chí | Pixel processing | RAG |
|----------|-----------------|-----|
| Số phần tử xử lý | 480,000+ | 400–1000 |
| Độ nhạy với noise | Cao | Thấp (noise bị "nuốt" vào region) |
| Cấu trúc ngữ nghĩa | Không có | Có (vùng = đối tượng thực) |
| Khả năng merge | Khó | Tự nhiên qua graph |
| Tích hợp với Graph algorithms | Không | Toàn bộ NetworkX API |

---

## 3. Cấu trúc của RAG

### Node attributes

Mỗi node trong RAG lưu các thuộc tính về vùng nó đại diện:

```python
g.nodes[n] = {
    'labels':       [n],                      # danh sách label pixel thuộc vùng này
    'pixel count':  1523,                     # số pixel
    'total color':  [r_sum, g_sum, b_sum],    # tổng màu (dùng tính mean)
    'mean color':   [128.4, 97.2, 63.1],      # màu trung bình
    'centroid':     (240.5, 310.2)            # tâm vùng (từ regionprops)
}
```

### Edge attributes

```python
g[u][v] = {
    'weight': 45.7   # độ chênh lệch màu giữa vùng u và v (Euclidean distance trong RGB space)
}
```

### Công thức tính weight mặc định

```python
def weight(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    return np.linalg.norm(diff)
    # Euclidean distance trong không gian RGB 3D
    # = sqrt((ΔR)² + (ΔG)² + (ΔB)²)
    # Range: 0 (identical) → ~441 (black vs white)
```

### Class hierarchy

```
networkx.Graph
    └── skimage.future.graph.RAG
            ├── .add_edge(u, v, weight=...)
            ├── .merge_nodes(src, dst, weight_func=..., in_place=True)
            └── kế thừa toàn bộ nx.Graph API
```

---

## 4. Thuật toán xây dựng RAG

### 4.1 SLIC → RAG pipeline

```
Input Image
    │
    ▼
SLIC Superpixels          # Chia ảnh thành N vùng đồng nhất
    │  (compactness, n_segments)
    ▼
Label Map                 # Mảng 2D: mỗi pixel được gán label [0..N-1]
    │
    ▼
rag_mean_color(img, labels)
    │  # Duyệt qua tất cả cạnh pixel liền kề
    │  # Nếu hai pixel thuộc label khác nhau → thêm edge
    │  # Weight = ||mean_color[u] - mean_color[v]||₂
    ▼
RAG (networkx.Graph subclass)
```

### 4.2 Merge nodes

Khi merge hai node `src` và `dst`:

```
Trước merge:          Sau merge:
  A ---[w1]--- src      A ---[max(w1,w3)]--- merged
  B ---[w2]--- src      B ---[max(w2,w4)]--- merged
  C ---[w3]--- dst      (src và dst bị xóa/gộp)
  D ---[w4]--- dst      C và D kết nối vào merged
```

Có hai chiến lược xử lý conflict (khi neighbor `n` kề cả `src` lẫn `dst`):

```python
# Chiến lược 1: Giữ weight nhỏ hơn (default — conservative merge)
w = min(g[n][src]['weight'], g[n][dst]['weight'])

# Chiến lược 2: Giữ weight lớn hơn (aggressive — dùng khi muốn preserve boundaries)
def max_edge(g, src, dst, n):
    w1 = g[n].get(src, {'weight': -np.inf})['weight']
    w2 = g[n].get(dst, {'weight': -np.inf})['weight']
    return {'weight': max(w1, w2)}
```

### 4.3 Hierarchical merging

```
RAG ban đầu (N nodes)
    │
    ▼ merge cặp có weight thấp nhất
RAG (N-1 nodes)
    │
    ▼ lặp lại cho đến khi min_weight > threshold
    │
RAG cuối (M nodes, M << N)
```

Đây là **agglomerative clustering** trên graph — tương tự Ward's method nhưng có ràng buộc không gian (chỉ merge các vùng thực sự kề nhau).

---

## 5. Ứng dụng thực tế

### 5.1 Image segmentation refinement

Dùng RAG để "làm mịn" kết quả SLIC vốn over-segment:

```python
# SLIC tạo 400 superpixels → RAG merge xuống còn ~20 vùng ngữ nghĩa
labels2 = graph.cut_threshold(labels1, g, 29)
# Mọi cặp vùng có weight < 29 sẽ bị merge
```

**Use case**: Tách nền/foreground, detect object boundaries, preprocessing cho CNN.

### 5.2 Game level difficulty (use case của project này)

RAG rất phù hợp để tính độ khó của một level trong game puzzle/visual:

```python
g = graph.rag_mean_color(level_image, labels)

# Metrics từ RAG:
n_regions    = len(g.nodes)                      # số vùng → complexity
edge_weights = [g[u][v]['weight'] for u,v in g.edges()]
avg_contrast = np.mean(edge_weights)             # contrast trung bình
max_contrast = np.max(edge_weights)              # boundary khó nhất
connectivity = len(g.edges) / len(g.nodes)       # mật độ kết nối

# Difficulty score (heuristic):
difficulty = (n_regions * avg_contrast * connectivity) / normalization_factor
```

| Đặc điểm ảnh | Biểu hiện trong RAG | Độ khó |
|-------------|-------------------|--------|
| Ít vùng, màu tương đồng | Ít node, weight thấp | Dễ |
| Nhiều vùng, màu tương đồng | Nhiều node, weight thấp | Trung bình (confusing) |
| Nhiều vùng, màu tương phản | Nhiều node, weight cao | Khó |
| Vùng nhỏ lẫn nhau | High connectivity | Rất khó |

### 5.3 Object detection preprocessing

RAG giúp group các superpixel thành candidate objects trước khi đưa vào model:

```
SLIC → RAG → graph cut → candidate regions → CNN classifier
```

### 5.4 Medical image analysis

Phân tách mô tế bào, detect khối u — nơi mà ranh giới giữa các vùng sinh học rất quan trọng. RAG bảo toàn thông tin boundary tốt hơn global thresholding.

### 5.5 Texture-based retrieval

Dùng RAG structure (số node, distribution of weights, graph density) làm **feature vector** để so sánh ảnh — không cần pixel-level comparison.

```python
def rag_features(img):
    labels = segmentation.slic(img, n_segments=200)
    g = graph.rag_mean_color(img, labels)
    weights = [d['weight'] for _,_,d in g.edges(data=True)]
    return {
        'n_regions':   len(g.nodes),
        'n_edges':     len(g.edges),
        'mean_weight': np.mean(weights),
        'std_weight':  np.std(weights),
        'density':     nx.density(g)
    }
```

---

## 6. Hiệu quả và giới hạn

### Hiệu quả

**Giảm chiều dữ liệu mạnh mẽ:**

```
Ảnh 1080p (2M pixels) → ~500 superpixels → RAG với ~1000 edges
Tỉ lệ giảm: 2,000x về số phần tử, ~4,000x về số so sánh
```

**Robust với noise:** Noise pixel-level bị triệt tiêu trong quá trình tính mean color của region.

**Tích hợp tốt với graph algorithms:** Dijkstra, spectral clustering, minimum spanning tree — tất cả đều áp dụng được ngay vì RAG kế thừa từ NetworkX.

**Spatial constraint tự nhiên:** Chỉ merge các vùng thực sự kề nhau — tránh các lỗi ngữ nghĩa kiểu "merge bầu trời với mặt đất vì cùng màu xanh".

### Giới hạn

**Phụ thuộc vào bước segmentation ban đầu:**
- Nếu SLIC/Watershed cho kết quả kém → RAG kém theo
- Tham số `compactness` và `n_segments` ảnh hưởng lớn đến chất lượng

**Floating point precision:**

```python
# Bug thực tế ghi nhận trong tài liệu — cùng một cặp vùng:
# Tính với float64:  weight = 29.82  (< threshold 30 → không hiển thị)
# Tính với int:      weight = 30.23  (> threshold 30 → hiển thị)
# → Phải thống nhất dtype trước khi tính
```

**Không phân biệt được texture:** RAG với `rag_mean_color` chỉ dùng màu trung bình — hai vùng có cùng màu trung bình nhưng texture hoàn toàn khác sẽ bị merge nhầm.

**Graph không encode hình dạng:** Không biết vùng nào to, nhỏ, tròn, méo — chỉ biết "kề nhau" và "màu cách nhau bao nhiêu".

### Khi nào không nên dùng RAG

- Ảnh có cấu trúc hình học quan trọng hơn màu sắc → dùng Hough transform hoặc contour detection
- Cần real-time processing trên video → RAG xây dựng lại mỗi frame khá tốn kém
- Object detection với deep features → YOLO/Faster-RCNN cho kết quả tốt hơn nhiều

---

## 7. Implement Python — scikit-image

### Cài đặt

```bash
pip install scikit-image networkx matplotlib numpy
```

### 7.1 Basic RAG với merge nodes

```python
import skimage as ski
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np


def max_edge(g, src, dst, n):
    """
    Callback để chọn weight lớn hơn khi merge nodes.
    Dùng khi muốn bảo toàn boundaries mạnh nhất.
    """
    w1 = g[n].get(src, {'weight': -np.inf})['weight']
    w2 = g[n].get(dst, {'weight': -np.inf})['weight']
    return {'weight': max(w1, w2)}


# Tạo RAG thủ công (demo)
g = ski.graph.RAG()
g.add_edge(1, 2, weight=10)
g.add_edge(2, 3, weight=20)
g.add_edge(3, 4, weight=30)
g.add_edge(4, 1, weight=40)
g.add_edge(1, 3, weight=50)

for n in g.nodes():
    g.nodes[n]['labels'] = [n]

gc = g.copy()

# Merge với chiến lược min (default)
g.merge_nodes(1, 3)

# Merge với chiến lược max, không in-place
gc.merge_nodes(1, 3, weight_func=max_edge, in_place=False)
```

### 7.2 RAG từ ảnh thực — mean color

```python
from skimage import data, segmentation, color
from skimage.future import graph


img = data.coffee()  # Thay bằng ảnh của bạn: img = io.imread('your_image.png')

# Bước 1: Segmentation
labels1 = segmentation.slic(img, compactness=30, n_segments=400)

# Bước 2: Build RAG
g = graph.rag_mean_color(img, labels1)

# Bước 3: Threshold cut — merge các vùng màu gần nhau
labels2 = graph.cut_threshold(labels1, g, 29)

# Visualize
out1 = color.label2rgb(labels1, img, kind='avg')
out2 = color.label2rgb(labels2, img, kind='avg')
```

### 7.3 Hierarchical merging với custom callbacks

```python
def _weight_mean_color(graph, src, dst, n):
    """Recompute weight dựa trên mean color của dst sau merge."""
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    return np.linalg.norm(diff)


def merge_mean_color(graph, src, dst):
    """Update mean color của dst khi merge src vào."""
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (
        graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count']
    )


labels2 = graph.merge_hierarchical(
    labels, g,
    thresh=40,
    rag_copy=False,
    in_place_merge=True,
    merge_func=merge_mean_color,
    weight_func=_weight_mean_color
)
```

### 7.4 Visualize RAG trên ảnh

```python
from skimage.measure import regionprops
from skimage import draw


def display_edges(image, rag, threshold=30):
    """Vẽ edges của RAG lên ảnh. Edge xanh = weight < threshold."""
    image = image.copy()

    for region in regionprops(labels):
        rag.nodes[region['label']]['centroid'] = region['centroid']

    for n1, n2 in rag.edges():
        r1, c1 = map(int, rag.nodes[n1]['centroid'])
        r2, c2 = map(int, rag.nodes[n2]['centroid'])

        weight = rag[n1][n2]['weight']
        line   = draw.line(r1, c1, r2, c2)
        circle = draw.disk((r1, c1), 2)

        if weight < threshold:
            image[line]   = 0, 1, 0  # xanh lá = similar → merge candidate
        image[circle] = 1, 1, 0      # vàng = node centroid

    return image
```

### 7.5 Tính difficulty score cho game level

```python
def compute_level_difficulty(img, n_segments=200, compactness=30):
    """
    Tính difficulty score cho một level dựa trên RAG.
    Score cao hơn = level khó hơn.
    """
    labels = segmentation.slic(img, compactness=compactness, n_segments=n_segments)
    g      = graph.rag_mean_color(img, labels)

    weights    = [d['weight'] for _, _, d in g.edges(data=True)]
    n_regions  = len(g.nodes)
    avg_weight = np.mean(weights) if weights else 0
    density    = nx.density(g)

    # Heuristic: nhiều vùng + contrast cao + kết nối dày = khó
    raw_score  = (n_regions * avg_weight * density) / 1000
    score      = min(raw_score / 5, 1.0)

    if score < 0.33:
        difficulty = 'Easy'
    elif score < 0.66:
        difficulty = 'Medium'
    else:
        difficulty = 'Hard'

    return {
        'score':        round(score, 3),
        'difficulty':   difficulty,
        'n_regions':    n_regions,
        'n_edges':      len(g.edges),
        'avg_contrast': round(avg_weight, 2),
        'std_contrast': round(np.std(weights), 2) if weights else 0,
        'density':      round(density, 4)
    }


result = compute_level_difficulty(img)
print(result)
# {'score': 0.412, 'difficulty': 'Medium', 'n_regions': 187, ...}
```

---

## 8. Demo tương tác — HTML / CSS / JS

File demo dưới đây chạy hoàn toàn trong browser, không cần server hay thư viện ngoài. Lưu thành `rag-demo.html` và mở trực tiếp.

```html
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG — Region Adjacency Graph Demo</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #f8f7f4;
    --surface: #ffffff;
    --border:  rgba(0,0,0,0.1);
    --text:    #1a1a18;
    --muted:   #6b6b65;
    --green:   #2d7d46;
    --amber:   #92550a;
    --red:     #8b2020;
    --radius:  10px;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 2rem 1rem;
  }

  h1 { font-size: 1.5rem; font-weight: 500; margin-bottom: .25rem; }

  .subtitle { font-size: .875rem; color: var(--muted); margin-bottom: 2rem; }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    max-width: 900px;
    margin: 0 auto;
  }

  .panel {
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
  }

  .panel-title {
    font-size: .7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--muted);
    margin-bottom: .75rem;
  }

  canvas { display: block; width: 100%; border-radius: 6px; border: 0.5px solid var(--border); }

  .controls { margin-top: .875rem; display: flex; flex-direction: column; gap: .5rem; }

  .ctrl-row { display: flex; align-items: center; gap: .625rem; }

  .ctrl-row label { font-size: .8rem; color: var(--muted); width: 80px; flex-shrink: 0; }

  input[type=range] { flex: 1; accent-color: var(--text); cursor: pointer; }

  .val { font-size: .8rem; font-weight: 500; min-width: 28px; text-align: right; }

  .stats {
    max-width: 900px;
    margin: 1rem auto 0;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .75rem;
  }

  .stat {
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: var(--radius);
    padding: .875rem 1rem;
    text-align: center;
  }

  .stat-val  { font-size: 1.5rem; font-weight: 500; }
  .stat-lbl  { font-size: .7rem; color: var(--muted); margin-top: .3rem; }

  .difficulty-row {
    max-width: 900px;
    margin: 1rem auto 0;
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .diff-label { font-size: .875rem; color: var(--muted); }

  .badge { display: inline-block; padding: .3rem .875rem; border-radius: 5px; font-size: .8rem; font-weight: 500; }
  .badge-easy   { background: #eaf3de; color: #2d6614; }
  .badge-medium { background: #faeeda; color: #854f0b; }
  .badge-hard   { background: #faece7; color: #7d2e15; }

  .score-bar { flex: 1; min-width: 120px; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .score-fill { height: 100%; border-radius: 3px; background: var(--text); transition: width .4s ease; }

  .btn {
    padding: .45rem 1rem;
    border: 0.5px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    color: var(--text);
    font-size: .8rem;
    cursor: pointer;
    transition: background .15s;
  }
  .btn:hover { background: var(--bg); }

  .edge-section {
    max-width: 900px;
    margin: 1rem auto 0;
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
  }

  .edge-list { max-height: 140px; overflow-y: auto; font-size: .78rem; }

  .edge-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: .3rem 0;
    border-bottom: 0.5px solid var(--border);
  }
  .edge-row:last-child { border-bottom: none; }

  .edge-weight { font-weight: 500; }
  .w-low  { color: var(--green); }
  .w-mid  { color: var(--amber); }
  .w-high { color: var(--red); }

  @media (max-width: 600px) {
    .grid  { grid-template-columns: 1fr; }
    .stats { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>

<div style="max-width:900px;margin:0 auto 1.5rem">
  <h1>Region Adjacency Graph</h1>
  <p class="subtitle">
    Mỗi vùng màu → 1 node &nbsp;·&nbsp; Hai node kề nhau → 1 edge với weight = độ chênh lệch màu
    <br>Threshold lọc edges yếu (vùng giống nhau) — ứng viên để merge
  </p>
</div>

<div class="grid">
  <div class="panel">
    <div class="panel-title">Ảnh phân đoạn (Voronoi superpixels)</div>
    <canvas id="imgCanvas" width="380" height="320"></canvas>
    <div class="controls">
      <div class="ctrl-row">
        <label>Segments</label>
        <input type="range" id="segSlider" min="4" max="20" value="9" step="1">
        <span class="val" id="segVal">9</span>
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">RAG — đồ thị vùng kề nhau</div>
    <canvas id="ragCanvas" width="380" height="320"></canvas>
    <div class="controls">
      <div class="ctrl-row">
        <label>Threshold</label>
        <input type="range" id="threshSlider" min="5" max="150" value="60" step="5">
        <span class="val" id="threshVal">60</span>
      </div>
    </div>
  </div>
</div>

<div class="stats">
  <div class="stat"><div class="stat-val" id="sNodes">9</div><div class="stat-lbl">Regions (nodes)</div></div>
  <div class="stat"><div class="stat-val" id="sEdges">0</div><div class="stat-lbl">Adjacencies (edges)</div></div>
  <div class="stat"><div class="stat-val" id="sAvg">0</div><div class="stat-lbl">Avg color diff</div></div>
  <div class="stat"><div class="stat-val" id="sConn">0</div><div class="stat-lbl">Active edges (&lt; thresh)</div></div>
</div>

<div class="difficulty-row">
  <div>
    <span class="diff-label">Level difficulty: </span>
    <span id="diffBadge" class="badge badge-medium">Medium</span>
  </div>
  <div class="score-bar">
    <div class="score-fill" id="scoreFill" style="width:50%"></div>
  </div>
  <span class="val" id="scoreVal">0.50</span>
  <button class="btn" onclick="mergeSimilar()">Merge most similar ↻</button>
  <button class="btn" onclick="generate()">Reset ↺</button>
</div>

<div class="edge-section">
  <div class="panel-title" style="margin-bottom:.5rem">Edge list — sorted by weight (color difference)</div>
  <div class="edge-list" id="edgeList"></div>
</div>

<script>
const PALETTE = [
  [221,89,72],  [78,162,210], [72,188,118], [228,192,72],
  [168,108,192],[88,152,88],  [210,148,88], [148,192,212],
  [228,118,152],[108,192,168],[192,192,88], [160,88,172],
  [88,108,192], [212,168,128],[128,212,148],[192,108,108],
  [72,148,172], [212,132,92], [152,212,88], [108,88,192]
];

const W = 380, H = 320;
let regions = [], edges = [], segCount = 9, threshold = 60;

const imgCvs = document.getElementById('imgCanvas');
const ragCvs = document.getElementById('ragCanvas');
const imgCtx = imgCvs.getContext('2d');
const ragCtx = ragCvs.getContext('2d');

function colorDiff(a, b) {
  return Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2);
}

function generate() {
  const n = segCount;
  const cols = Math.ceil(Math.sqrt(n * W / H));
  const rows = Math.ceil(n / cols);
  const seeds = [];
  let idx = 0;
  for (let r = 0; r < rows && idx < n; r++) {
    for (let c = 0; c < cols && idx < n; c++) {
      seeds.push([
        (c + 0.3 + Math.random() * 0.4) * (W / cols),
        (r + 0.3 + Math.random() * 0.4) * (H / rows)
      ]);
      idx++;
    }
  }

  // Voronoi assignment
  const assign = new Int32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let best = 0, bd = 1e9;
      for (let i = 0; i < n; i++) {
        const d = (x - seeds[i][0]) ** 2 + (y - seeds[i][1]) ** 2;
        if (d < bd) { bd = d; best = i; }
      }
      assign[y * W + x] = best;
    }
  }

  // Draw segmented image
  const imgData = imgCtx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const c = PALETTE[assign[i] % PALETTE.length];
    imgData.data[i*4]   = c[0];
    imgData.data[i*4+1] = c[1];
    imgData.data[i*4+2] = c[2];
    imgData.data[i*4+3] = 255;
  }
  imgCtx.putImageData(imgData, 0, 0);

  // Boundaries
  for (let y = 0; y < H - 1; y++) {
    for (let x = 0; x < W - 1; x++) {
      if (assign[y*W+x] !== assign[y*W+x+1] || assign[y*W+x] !== assign[(y+1)*W+x]) {
        imgCtx.fillStyle = 'rgba(0,0,0,0.22)';
        imgCtx.fillRect(x, y, 1, 1);
      }
    }
  }

  // Region labels
  imgCtx.textAlign = 'center'; imgCtx.textBaseline = 'middle';
  for (let i = 0; i < n; i++) {
    imgCtx.font = 'bold 13px -apple-system, sans-serif';
    imgCtx.fillStyle = 'rgba(0,0,0,0.4)';
    imgCtx.fillText(i + 1, seeds[i][0] + 1, seeds[i][1] + 1);
    imgCtx.fillStyle = 'rgba(255,255,255,0.92)';
    imgCtx.fillText(i + 1, seeds[i][0], seeds[i][1]);
  }

  // Build adjacency
  const adjSet = new Set();
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W - 1; x++) {
      const a = assign[y*W+x], b = assign[y*W+x+1];
      if (a !== b) adjSet.add(Math.min(a,b) + '_' + Math.max(a,b));
    }
  for (let y = 0; y < H - 1; y++)
    for (let x = 0; x < W; x++) {
      const a = assign[y*W+x], b = assign[(y+1)*W+x];
      if (a !== b) adjSet.add(Math.min(a,b) + '_' + Math.max(a,b));
    }

  regions = seeds.map((s, i) => ({
    id: i, x: s[0], y: s[1],
    color: PALETTE[i % PALETTE.length],
    alive: true
  }));

  edges = [...adjSet].map(k => {
    const [a, b] = k.split('_').map(Number);
    return { a, b, w: Math.round(colorDiff(PALETTE[a % PALETTE.length], PALETTE[b % PALETTE.length])) };
  }).sort((x, y) => x.w - y.w);

  drawRAG();
  updateStats();
}

function drawRAG() {
  ragCtx.clearRect(0, 0, W, H);

  for (const e of edges) {
    const ra = regions[e.a], rb = regions[e.b];
    if (!ra?.alive || !rb?.alive) continue;

    const visible = e.w < threshold;
    ragCtx.beginPath();
    ragCtx.moveTo(ra.x, ra.y);
    ragCtx.lineTo(rb.x, rb.y);

    if (visible) {
      const t = e.w / threshold;
      ragCtx.strokeStyle = `rgba(${Math.round(t*200+30)},${Math.round((1-t)*160+60)},50,0.75)`;
      ragCtx.lineWidth = Math.max(0.8, 3.5 - e.w / 30);
    } else {
      ragCtx.strokeStyle = 'rgba(160,160,160,0.15)';
      ragCtx.lineWidth = 0.5;
    }
    ragCtx.stroke();

    if (visible && e.w < threshold * 0.55) {
      const mx = (ra.x + rb.x) / 2, my = (ra.y + rb.y) / 2;
      ragCtx.font = '10px -apple-system, sans-serif';
      ragCtx.textAlign = 'center'; ragCtx.textBaseline = 'middle';
      ragCtx.fillStyle = 'rgba(30,30,30,0.6)';
      ragCtx.fillText(e.w, mx, my - 2);
    }
  }

  for (const r of regions) {
    if (!r?.alive) continue;
    ragCtx.beginPath();
    ragCtx.arc(r.x, r.y, 11, 0, Math.PI * 2);
    ragCtx.fillStyle = `rgb(${r.color[0]},${r.color[1]},${r.color[2]})`;
    ragCtx.fill();
    ragCtx.strokeStyle = 'rgba(0,0,0,0.2)'; ragCtx.lineWidth = 1;
    ragCtx.stroke();

    ragCtx.font = 'bold 11px -apple-system, sans-serif';
    ragCtx.textAlign = 'center'; ragCtx.textBaseline = 'middle';
    ragCtx.fillStyle = 'rgba(0,0,0,0.65)';
    ragCtx.fillText(r.id + 1, r.x, r.y);
  }
}

function updateStats() {
  const alive       = regions.filter(r => r?.alive);
  const liveEdges   = edges.filter(e => regions[e.a]?.alive && regions[e.b]?.alive);
  const activeEdges = liveEdges.filter(e => e.w < threshold);
  const weights     = liveEdges.map(e => e.w);
  const avgW        = weights.length ? Math.round(weights.reduce((a,b)=>a+b,0) / weights.length) : 0;

  document.getElementById('sNodes').textContent = alive.length;
  document.getElementById('sEdges').textContent = liveEdges.length;
  document.getElementById('sAvg').textContent   = avgW;
  document.getElementById('sConn').textContent  = activeEdges.length;

  const density  = liveEdges.length / Math.max(alive.length, 1);
  const rawScore = (alive.length * avgW * density) / 8000;
  const score    = Math.min(rawScore, 1);

  document.getElementById('scoreFill').style.width = Math.round(score * 100) + '%';
  document.getElementById('scoreVal').textContent  = score.toFixed(2);

  const badge = document.getElementById('diffBadge');
  if (score < 0.33)      { badge.textContent = 'Easy';   badge.className = 'badge badge-easy'; }
  else if (score < 0.66) { badge.textContent = 'Medium'; badge.className = 'badge badge-medium'; }
  else                   { badge.textContent = 'Hard';   badge.className = 'badge badge-hard'; }

  document.getElementById('edgeList').innerHTML = liveEdges.slice(0, 15).map(e => {
    const cls = e.w < threshold * 0.4 ? 'w-low' : e.w < threshold ? 'w-mid' : 'w-high';
    return `<div class="edge-row">
      <span>Region ${e.a+1} ↔ Region ${e.b+1}</span>
      <span class="edge-weight ${cls}">Δ = ${e.w}</span>
    </div>`;
  }).join('');
}

function mergeSimilar() {
  const candidate = edges
    .filter(e => regions[e.a]?.alive && regions[e.b]?.alive)
    .sort((x, y) => x.w - y.w)[0];
  if (!candidate) return;

  const { a, b } = candidate;
  const ra = regions[a], rb = regions[b];

  ra.x = (ra.x + rb.x) / 2;
  ra.y = (ra.y + rb.y) / 2;
  ra.color = [
    Math.round((ra.color[0] + rb.color[0]) / 2),
    Math.round((ra.color[1] + rb.color[1]) / 2),
    Math.round((ra.color[2] + rb.color[2]) / 2)
  ];
  rb.alive = false;

  edges = edges
    .filter(e => !(e.a === a && e.b === b) && !(e.a === b && e.b === a))
    .map(e => { if (e.a === b) e.a = a; if (e.b === b) e.b = a; return e; })
    .filter(e => e.a !== e.b)
    .filter((e, i, arr) =>
      arr.findIndex(x => (x.a===e.a&&x.b===e.b)||(x.a===e.b&&x.b===e.a)) === i
    );

  edges.forEach(e => {
    if (regions[e.a]?.alive && regions[e.b]?.alive)
      e.w = Math.round(colorDiff(regions[e.a].color, regions[e.b].color));
  });

  drawRAG();
  updateStats();
}

document.getElementById('segSlider').addEventListener('input', function () {
  segCount = +this.value;
  document.getElementById('segVal').textContent = segCount;
  generate();
});

document.getElementById('threshSlider').addEventListener('input', function () {
  threshold = +this.value;
  document.getElementById('threshVal').textContent = threshold;
  drawRAG();
  updateStats();
});

generate();
</script>
</body>
</html>
```

**Cách dùng demo:**
- Kéo **Segments** để thay đổi số vùng trong ảnh
- Kéo **Threshold** để lọc edge — chỉ hiện edges giữa các vùng màu gần nhau (merge candidates)
- Nhấn **Merge most similar** để gộp hai vùng giống nhau nhất, y hệt `merge_hierarchical()` trong skimage
- Nhấn **Reset** để random lại ảnh mới

---

## 9. Những điều dev cần biết thêm

### 9.1 Dtype gotcha — nguồn gốc của silent bugs

Đây là lỗi thực tế được ghi nhận trong tài liệu gốc của scikit-image:

```python
# Hai vùng với mean color float64:
v1_float = np.array([205.07, 151.04, 101.55])
v2_float = np.array([199.41, 134.50,  77.40])
diff_float = np.linalg.norm(v1_float - v2_float)
# → 29.82  (< threshold 30 → edge KHÔNG được hiển thị)

# Cùng hai vùng nhưng cast sang int:
v1_int = v1_float.astype(int)   # [205, 151, 101]
v2_int = v2_float.astype(int)   # [199, 134,  77]
diff_int = np.linalg.norm(v1_int - v2_int)
# → 30.23  (> threshold 30 → edge ĐƯỢC hiển thị)

# Cùng một cặp vùng, hai kết quả khác nhau chỉ vì dtype!
```

**Fix:** Luôn thống nhất dtype trước khi tính:

```python
# Dùng float64 xuyên suốt pipeline
mean_color = region.mean_intensity.astype(np.float64)
```

### 9.2 `in_place` vs copy — khi nào dùng gì?

```python
# in_place=True (default): sửa trực tiếp g, src bị xóa
g.merge_nodes(1, 3)

# in_place=False: tạo node mới, giữ nguyên src và dst để debug/audit
g.merge_nodes(1, 3, weight_func=max_edge, in_place=False)
```

Trong production, dùng `in_place=False` khi cần audit trail (biết vùng nào merge vào đâu).

### 9.3 RAG không phải chỉ dùng cho màu sắc

`rag_mean_color` là function tiện lợi nhất, nhưng có thể build RAG với bất kỳ feature nào:

```python
# RAG dựa trên texture (Local Binary Pattern descriptor)
from skimage.feature import local_binary_pattern

def rag_texture(img, labels):
    lbp = local_binary_pattern(color.rgb2gray(img), P=8, R=1)
    g   = graph.RAG(labels)
    for region in regionprops(labels):
        mask = labels == region.label
        g.nodes[region.label]['lbp_hist'] = np.histogram(lbp[mask], bins=16)[0]
    for u, v in g.edges():
        hist_u = g.nodes[u]['lbp_hist']
        hist_v = g.nodes[v]['lbp_hist']
        g[u][v]['weight'] = np.sum(np.abs(hist_u - hist_v))  # L1 distance
    return g
```

### 9.4 Kết hợp RAG với minimum spanning tree

RAG + MST cho phép tìm "xương sống" cấu trúc ảnh — rất hữu ích cho shape analysis:

```python
import networkx as nx

g   = graph.rag_mean_color(img, labels)
mst = nx.minimum_spanning_tree(g)

# Edges trong MST = các ranh giới quan trọng nhất của ảnh
# Prune MST để detect objects chính
for u, v, d in sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:5]:
    print(f"Important boundary: region {u} ↔ region {v}, contrast={d['weight']:.1f}")
```

### 9.5 Complexity analysis

| Operation | Complexity | Ghi chú |
|-----------|-----------|---------|
| Build RAG từ label map | `O(H × W)` | Duyệt mọi pixel một lần |
| `cut_threshold` | `O(E log E)` | Sort edges rồi union-find |
| `merge_hierarchical` | `O(E log E + N²)` | Bottleneck ở cập nhật weights |
| NetworkX operations (shortest path...) | `O(V + E)` đến `O(V²)` | Phụ thuộc thuật toán |

Với ảnh 1080p và 500 segments: build RAG ~0.3s, merge ~0.1s — đủ cho preprocessing pipeline, không đủ cho real-time video.

### 9.6 Debug RAG bằng NetworkX

Vì RAG là subclass của `nx.Graph`, có thể dùng toàn bộ NetworkX toolbox để debug:

```python
import networkx as nx

g = graph.rag_mean_color(img, labels)

# Tìm vùng kết nối nhiều nhất (most "central" region)
centrality    = nx.degree_centrality(g)
most_central  = max(centrality, key=centrality.get)

# Kiểm tra graph có connected không
print(nx.is_connected(g))          # True/False

# Tìm communities (nhóm vùng màu tương đồng)
communities = nx.community.greedy_modularity_communities(g, weight='weight')

# Export để visualize bằng Gephi hoặc D3.js
nx.write_gexf(g, 'rag.gexf')
nx.write_graphml(g, 'rag.graphml')
```

### 9.7 Normalized Cut — thay thế threshold cut khi cần chính xác hơn

`cut_threshold` đơn giản nhưng nhạy cảm với giá trị threshold. `cut_normalized` (Shi & Malik 2000) tốt hơn cho ảnh phức tạp:

```python
# Normalized cut — tự tìm ranh giới tối ưu, không cần chọn threshold
labels2 = graph.cut_normalized(labels1, g)

# So sánh:
# cut_threshold  : O(E log E), cần tuning threshold thủ công, fast
# cut_normalized : chậm hơn ~5x, tự động, tốt hơn cho real-world images
```

### 9.8 Checklist khi implement RAG cho production

```
☐ Chuẩn hóa dtype về float64 trước khi tính color features
☐ Chọn n_segments dựa trên kích thước ảnh: n_segments ≈ W*H / 2000
☐ Test với ảnh grayscale riêng (convert sang RGB trước khi dùng rag_mean_color)
☐ Validate rằng không có isolated nodes (vùng không kề ai)
☐ Log số regions trước và sau merge để detect threshold quá aggressive
☐ Profile build time với ảnh lớn nhất trong production
☐ Nếu dùng custom weight_func: return dict {'weight': float}, không phải float
☐ Với game level: normalize difficulty score theo dataset thực (không dùng heuristic cứng)
```

---

## 10. Giải thích thuật ngữ

Bảng tra cứu nhanh — sắp xếp theo nhóm chủ đề để dễ tìm.

---

### Thuật ngữ đồ thị (Graph theory)

| Thuật ngữ | Tiếng Việt | Giải thích |
|-----------|-----------|-----------|
| **Graph** `G = (V, E)` | Đồ thị | Cấu trúc dữ liệu gồm tập các đỉnh (V) và tập các cạnh (E) kết nối chúng |
| **Vertex / Node** | Đỉnh / Nút | Một phần tử trong đồ thị — trong RAG, mỗi node đại diện cho một vùng ảnh |
| **Edge** | Cạnh | Đường kết nối giữa hai node — trong RAG, thể hiện hai vùng tiếp giáp nhau |
| **Weight** `w(u,v)` | Trọng số cạnh | Giá trị số gắn với một cạnh — trong RAG là độ chênh lệch màu giữa hai vùng |
| **Undirected graph** | Đồ thị vô hướng | Đồ thị mà cạnh `(u,v)` và `(v,u)` là một — RAG là vô hướng vì "kề nhau" là quan hệ đối xứng |
| **Weighted graph** | Đồ thị có trọng số | Đồ thị mà mỗi cạnh mang một giá trị số (weight) |
| **Adjacency** | Kề nhau | Hai node được gọi là kề nhau (adjacent) nếu có cạnh nối trực tiếp |
| **Degree** | Bậc của đỉnh | Số cạnh kết nối vào một node — node bậc cao = vùng tiếp giáp nhiều vùng khác |
| **Connected graph** | Đồ thị liên thông | Mọi cặp node đều có đường đi đến nhau — ảnh bình thường cho RAG liên thông |
| **Isolated node** | Đỉnh cô lập | Node không có cạnh nào — thường là lỗi trong quá trình build RAG |
| **Density** | Mật độ đồ thị | Tỉ lệ `E / E_max` — RAG có density cao nghĩa là các vùng kết nối dày đặc |
| **MST** | Cây khung nhỏ nhất | Minimum Spanning Tree — cây con của graph nối tất cả node với tổng weight nhỏ nhất |
| **Merge nodes** | Gộp đỉnh | Hợp hai node thành một, chuyển tất cả cạnh của node bị xóa sang node còn lại |
| **In-place** | Thay thế trực tiếp | Sửa đổi cấu trúc gốc thay vì tạo bản sao — `in_place=True` xóa node nguồn khỏi graph |
| **Subgraph** | Đồ thị con | Một phần của graph gốc, lấy ra tập node và cạnh con |
| **Degree centrality** | Độ trung tâm theo bậc | Chỉ số đo tầm quan trọng của node dựa trên số cạnh kết nối |

---

### Thuật ngữ xử lý ảnh (Image processing)

| Thuật ngữ | Tiếng Việt | Giải thích |
|-----------|-----------|-----------|
| **Image segmentation** | Phân đoạn ảnh | Quá trình chia ảnh thành các vùng có đặc trưng đồng nhất (màu, texture...) |
| **Superpixel** | Siêu điểm ảnh | Một cụm pixel liền kề có đặc trưng tương đồng — đơn vị xử lý thay cho pixel đơn lẻ |
| **SLIC** | — | *Simple Linear Iterative Clustering* — thuật toán tạo superpixel phổ biến nhất, nhanh và cho kết quả đều đặn |
| **Label map** | Bản đồ nhãn | Mảng 2D cùng kích thước ảnh, mỗi pixel được gán một số nguyên (label) chỉ vùng nó thuộc về |
| **Region** | Vùng | Tập hợp các pixel liền kề được gán cùng một label |
| **Boundary** | Ranh giới | Đường phân cách giữa hai vùng liền kề — edge trong RAG tương ứng với một boundary |
| **Compactness** | Độ chặt | Tham số SLIC — giá trị cao → superpixel hình vuông đều; giá trị thấp → superpixel bám theo màu sắc |
| **Over-segmentation** | Phân đoạn thừa | Kết quả SLIC/Watershed tạo ra quá nhiều vùng nhỏ — RAG được dùng để merge chúng lại |
| **Mean color** | Màu trung bình | Giá trị RGB trung bình của tất cả pixel trong một vùng — feature chính dùng trong `rag_mean_color` |
| **Centroid** | Tâm vùng | Tọa độ trung bình `(row, col)` của tất cả pixel trong một vùng — dùng để vẽ RAG lên ảnh |
| **Watershed** | Lưu vực | Thuật toán segmentation khác, dựa trên gradient ảnh — thay thế cho SLIC khi cần bám theo edges |
| **Felzenszwalb** | — | Thuật toán segmentation dựa trên graph, tốt cho ảnh có texture mạnh |
| **Noise** | Nhiễu | Pixel bất thường không thuộc về vùng nào — RAG tự nhiên lọc noise qua bước tính mean color |
| **Grayscale** | Ảnh xám | Ảnh một kênh màu (0–255) — khi dùng với `rag_mean_color` cần convert sang RGB trước |
| **RGB** | — | *Red-Green-Blue* — không gian màu 3 chiều phổ biến nhất; mỗi pixel là vector `[R, G, B]` |
| **Pixel intensity** | Cường độ pixel | Giá trị số của pixel — trong RGB là bộ ba `[R, G, B]` từ 0 đến 255 |

---

### Thuật ngữ toán học (Mathematics)

| Thuật ngữ | Ký hiệu | Giải thích |
|-----------|---------|-----------|
| **Euclidean distance** | `‖a − b‖₂` | Khoảng cách thẳng giữa hai điểm — dùng để tính color difference: `√(ΔR²+ΔG²+ΔB²)` |
| **L1 distance** | `‖a − b‖₁` | Tổng giá trị tuyệt đối của hiệu: `|ΔR| + |ΔG| + |ΔB|` — ít nhạy với outlier hơn L2 |
| **Norm** `‖·‖` | Chuẩn | Hàm đo "độ lớn" của một vector — `np.linalg.norm()` mặc định tính L2 norm |
| **float64** | Số thực 64-bit | Kiểu dữ liệu dấu phẩy động độ chính xác kép — mặc định của NumPy arrays |
| **dtype** | Kiểu dữ liệu | *Data type* — int vs float gây ra kết quả tính toán khác nhau (xem mục 9.1) |
| **Threshold** | Ngưỡng | Giá trị cắt — edge có weight nhỏ hơn threshold được coi là "tương đồng", ứng viên để merge |
| **Agglomerative clustering** | Phân cụm kết tụ | Thuật toán gộp dần từng cặp gần nhất — `merge_hierarchical` trong RAG là agglomerative có spatial constraint |
| **Complexity** `O(n)` | Độ phức tạp | Ký hiệu Big-O — đo tốc độ tăng thời gian chạy theo kích thước input |

---

### Thuật ngữ API (scikit-image / NetworkX)

| Tên | Loại | Ý nghĩa |
|-----|------|---------|
| `rag_mean_color(img, labels)` | Function | Xây dựng RAG từ ảnh và label map, dùng độ chênh lệch màu trung bình làm weight |
| `cut_threshold(labels, rag, thresh)` | Function | Merge tất cả cặp vùng có weight < thresh — nhanh, cần tuning thủ công |
| `cut_normalized(labels, rag)` | Function | Normalized cut — tự động tìm ranh giới tối ưu, chậm hơn nhưng chính xác hơn |
| `merge_hierarchical(...)` | Function | Merge lần lượt từng cặp có weight nhỏ nhất cho đến khi đạt threshold |
| `merge_nodes(src, dst)` | Method | Gộp node `src` vào `dst`, cập nhật tất cả cạnh liên quan |
| `weight_func` | Callback | Hàm do người dùng định nghĩa để tính weight mới sau khi merge — nhận `(g, src, dst, n)` |
| `merge_func` | Callback | Hàm do người dùng định nghĩa để cập nhật node attributes sau merge — nhận `(g, src, dst)` |
| `in_place` | Parameter | `True` = xóa node gốc; `False` = tạo node mới, giữ nguyên node cũ để debug |
| `rag_copy` | Parameter | Trong `merge_hierarchical` — nếu `False` thì sửa graph gốc, tiết kiệm memory |
| `regionprops(labels)` | Function | Tính các thuộc tính hình học của mỗi vùng (centroid, area, bbox...) từ label map |
| `label2rgb(labels, img)` | Function | Tô màu lại từng vùng theo màu trung bình — dùng để visualize kết quả segmentation |
| `mark_boundaries(img, labels)` | Function | Vẽ đường viền ranh giới giữa các vùng lên ảnh |
| `nx.density(g)` | Function | Tính mật độ đồ thị: `2E / (V × (V−1))` |
| `nx.minimum_spanning_tree(g)` | Function | Tìm cây khung nhỏ nhất — trả về graph mới chỉ chứa các cạnh "quan trọng" nhất |
| `nx.write_gexf(g, path)` | Function | Export graph ra file `.gexf` để mở bằng Gephi |
| `context-stroke` | SVG keyword | Từ khóa SVG — marker kế thừa màu stroke của đường chứa nó |

---

### Thuật ngữ dùng trong context game level

| Thuật ngữ | Ý nghĩa trong context RAG + game |
|-----------|----------------------------------|
| **Difficulty score** | Điểm độ khó tính từ RAG — thường là hàm của số vùng, avg contrast, và density |
| **Color contrast** | Độ tương phản màu giữa hai vùng kề nhau = weight của edge trong RAG |
| **Region complexity** | Số lượng vùng (`n_nodes`) — nhiều vùng = level phức tạp hơn về mặt thị giác |
| **Connectivity density** | Tỉ lệ edge/node — cao = các vùng kết nối dày, ảnh có nhiều ranh giới |
| **Merge candidate** | Cặp vùng có weight < threshold — hai vùng "giống nhau", có thể gộp lại mà không mất thông tin |
| **Spatial constraint** | Ràng buộc không gian — RAG chỉ cho phép merge các vùng thực sự kề nhau, tránh merge "nhảy cóc" |
| **Normalization** | Chuẩn hóa score về range [0,1] để so sánh được giữa các level có kích thước khác nhau |

---

## 11. Tài liệu tham khảo

1. **Vighnesh Birodkar** — "scikit-image RAG Introduction"
   https://vcansimplify.wordpress.com/2014/07/06/scikit-image-rag-introduction

2. **scikit-image Documentation** — Region Adjacency Graphs examples
   https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/plot_rag.html

3. **MLVN Blog** — Region Adjacency Graphs
   https://melvincabatuan.github.io/RAG/

4. **Shi, J. & Malik, J. (2000)** — "Normalized Cuts and Image Segmentation"
   IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(8), 888–905

5. **Achanta et al. (2012)** — "SLIC Superpixels Compared to State-of-the-art Superpixel Methods"
   IEEE TPAMI — paper gốc của thuật toán SLIC dùng trong pipeline

6. **NetworkX Documentation**
   https://networkx.org/documentation/stable/

---

*Tài liệu này dựa trên scikit-image 0.24.x. API có thể thay đổi ở phiên bản mới hơn — kiểm tra changelog tại https://scikit-image.org/docs/stable/release_notes.html*