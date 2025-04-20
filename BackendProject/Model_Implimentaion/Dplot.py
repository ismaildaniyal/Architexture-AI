import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, box
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from shapely.ops import unary_union
import networkx as nx
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------- Model Definition ----------------------
class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    def forward(self, X):
        X_unsq = X.unsqueeze(1)            # Shape: (N, 1, hidden_dim)
        attn_out, _ = self.attn(X_unsq, X_unsq, X_unsq)
        X_norm = self.norm1(X_unsq + attn_out)
        mlp_out = self.mlp(X_norm)
        X_out = self.norm2(X_norm + mlp_out)
        return X_out.squeeze(1)            # Back to shape: (N, hidden_dim)

class GraphTransformer(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_layers=2, num_heads=4):
        super(GraphTransformer, self).__init__()
        self.embedding = nn.Linear(in_features, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, out_features)
    def forward(self, X, A):
        X_cat = torch.cat([X, A], dim=-1)
        X_emb = self.embedding(X_cat)
        for layer in self.layers:
            X_emb = layer(X_emb)
        return self.output_layer(X_emb)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.ln = nn.LayerNorm(hidden_features)
    def forward(self, X):
        out = F.relu(self.ln(self.fc1(X)))
        return self.fc2(out)

class GraphTransformerMLPModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, num_heads, mlp_hidden_features):
        super(GraphTransformerMLPModel, self).__init__()
        self.graph_transformer = GraphTransformer(feature_dim, hidden_dim, feature_dim, num_layers, num_heads)
        self.mlp = MLP(feature_dim, mlp_hidden_features, 4)
        self.ln = nn.LayerNorm(feature_dim)
    def forward(self, X, A):
        X_cat = torch.cat([X, A], dim=-1)
        Y = self.graph_transformer(X, A)
        S = X_cat + self.ln(Y)
        return self.mlp(S)

# ---------------------- Preprocessing Functions ----------------------
def overlap_percentage(b1, b2):
    intersection_area = b1.intersection(b2).area
    min_area = min(b1.area, b2.area)
    if b1.touches(b2) or intersection_area > 0:
        return (intersection_area / min_area) * 100 if min_area > 0 else 0
    return 0

def move_box(bbox, shift_x, shift_y):
    xmin, ymin, xmax, ymax = bbox.bounds
    return box(xmin + shift_x, ymin + shift_y, xmax + shift_x, ymax + shift_y)

def build_graph(bboxes, min_overlap=5):
    G = nx.Graph()
    for i in range(len(bboxes)):
        G.add_node(i)
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if overlap_percentage(bboxes[i], bboxes[j]) >= min_overlap:
                G.add_edge(i, j)
    return G

def is_fully_connected(G):
    return nx.is_connected(G)

def find_nearest_box(isolated_idx, bboxes, connected_indices):
    min_distance = float("inf")
    nearest_idx = None
    isolated_box = bboxes[isolated_idx]
    for idx in connected_indices:
        other_box = bboxes[idx]
        distance = isolated_box.centroid.distance(other_box.centroid)
        if distance < min_distance:
            min_distance = distance
            nearest_idx = idx
    return nearest_idx

def attach_disconnected_rooms(bboxes, shift_amount=5, max_attempts=100):
    adjusted_boxes = bboxes.copy()
    for attempt in range(max_attempts):
        G = build_graph(adjusted_boxes)
        if is_fully_connected(G):
            print(f"All rooms connected after {attempt + 1} adjustments.")
            return adjusted_boxes
        connected_components = list(nx.connected_components(G))
        largest_component = max(connected_components, key=len)
        isolated_rooms = [idx for component in connected_components if component != largest_component for idx in component]
        for idx in isolated_rooms:
            nearest_idx = find_nearest_box(idx, adjusted_boxes, list(largest_component))
            if nearest_idx is not None:
                nearest_box = adjusted_boxes[nearest_idx]
                isolated_box = adjusted_boxes[idx]
                dx = np.sign(nearest_box.centroid.x - isolated_box.centroid.x) * shift_amount
                dy = np.sign(nearest_box.centroid.y - isolated_box.centroid.y) * shift_amount
                adjusted_boxes[idx] = move_box(isolated_box, dx, dy)
                new_G = build_graph(adjusted_boxes)
                if is_fully_connected(new_G):
                    print(f"Room {idx} connected by shifting ({dx}, {dy})")
                    return adjusted_boxes
    print("Max attempts reached! Some rooms might still be disconnected.")
    return adjusted_boxes

def are_parallel(line1, line2, angle_threshold=5):
    v1 = np.array([line1.coords[1][0] - line1.coords[0][0], 
                   line1.coords[1][1] - line1.coords[0][1]])
    v2 = np.array([line2.coords[1][0] - line2.coords[0][0],
                   line2.coords[1][1] - line2.coords[0][1]])
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return False
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return abs(cos_theta) > np.cos(np.deg2rad(angle_threshold))

def line_distance(line1, line2):
    from shapely.geometry import Point
    return min(line1.distance(Point(p)) for p in line2.coords)

def merge_close_parallel_walls(bboxes, distance_threshold=15, angle_threshold=15):
    boxes = [box(*b) for b in bboxes]
    remaining_lines = {}
    for i in range(len(boxes)):
        current_lines = get_box_lines(boxes[i].bounds)
        for j in range(len(boxes)):
            if i != j:
                current_lines = remove_overlapping_lines_from_lines(current_lines, boxes[j].bounds)
        remaining_lines[i] = current_lines
    all_lines = []
    for room_idx in remaining_lines:
        for line in remaining_lines[room_idx]:
            x1, y1 = line.coords[0]
            x2, y2 = line.coords[1]
            room_bbox = boxes[room_idx].bounds
            if abs(x1 - x2) < 1e-6:
                if abs(x1 - room_bbox[0]) < 1e-6:
                    wall_type = 'left'
                else:
                    wall_type = 'right'
            else:
                if abs(y1 - room_bbox[1]) < 1e-6:
                    wall_type = 'bottom'
                else:
                    wall_type = 'top'
            all_lines.append((room_idx, wall_type, line))
    merged_pairs = set()
    adjusted_bboxes = [list(b.bounds) for b in boxes]
    for i in range(len(all_lines)):
        room_i, wall_i, line_i = all_lines[i]
        for j in range(i+1, len(all_lines)):
            room_j, wall_j, line_j = all_lines[j]
            if (room_i, room_j) in merged_pairs or (room_j, room_i) in merged_pairs:
                continue
            if are_parallel(line_i, line_j, angle_threshold):
                dist = line_distance(line_i, line_j)
                if dist < distance_threshold:
                    if wall_i in ['left', 'right'] and wall_j in ['left', 'right']:
                        x_avg = (line_i.coords[0][0] + line_j.coords[0][0]) / 2
                        if wall_i == 'right':
                            adjusted_bboxes[room_i][2] = x_avg
                        else:
                            adjusted_bboxes[room_i][0] = x_avg
                        if wall_j == 'right':
                            adjusted_bboxes[room_j][2] = x_avg
                        else:
                            adjusted_bboxes[room_j][0] = x_avg
                    elif wall_i in ['top', 'bottom'] and wall_j in ['top', 'bottom']:
                        y_avg = (line_i.coords[0][1] + line_j.coords[0][1]) / 2
                        if wall_i == 'top':
                            adjusted_bboxes[room_i][3] = y_avg
                        else:
                            adjusted_bboxes[room_i][1] = y_avg
                        if wall_j == 'top':
                            adjusted_bboxes[room_j][3] = y_avg
                        else:
                            adjusted_bboxes[room_j][1] = y_avg
                    merged_pairs.add((room_i, room_j))
                    print(f"Merged walls between Room {room_i+1} ({wall_i}) and Room {room_j+1} ({wall_j}) at distance {dist:.2f}")
    return adjusted_bboxes

def postprocess_bounding_boxes(bboxes, shift_amount=5):
    print("Building initial graph...")
    G = build_graph(bboxes)
    if is_fully_connected(G):
        print("All rooms are already connected.")
        return bboxes
    print("Disconnected rooms detected! Fixing...")
    return attach_disconnected_rooms(bboxes, shift_amount)

def reduce_overlap(bboxes, max_overlap_threshold=35, shift_amount=10, max_attempts=50):
    adjusted_boxes = [box(*bbox) for bbox in bboxes]
    for attempt in range(max_attempts):
        max_overlap = 0
        for i in range(len(adjusted_boxes)):
            for j in range(i + 1, len(adjusted_boxes)):
                overlap = overlap_percentage(adjusted_boxes[i], adjusted_boxes[j])
                max_overlap = max(max_overlap, overlap)
        if max_overlap <= max_overlap_threshold:
            print(f"Overlap reduction complete. Maximum overlap is {max_overlap:.1f}% (≤ {max_overlap_threshold}%).")
            break
        moved = False
        for i in range(len(adjusted_boxes)):
            for j in range(i + 1, len(adjusted_boxes)):
                current_overlap = overlap_percentage(adjusted_boxes[i], adjusted_boxes[j])
                if current_overlap == 100.0:
                    box_i = adjusted_boxes[i]
                    box_j = adjusted_boxes[j]
                    if box_i.area < box_j.area:
                        smaller_box = box_i
                        larger_box = box_j
                        smaller_idx = i
                    else:
                        smaller_box = box_j
                        larger_box = box_i
                        smaller_idx = j
                    x_min_l, y_min_l, x_max_l, y_max_l = larger_box.bounds
                    corners = [
                        (x_min_l, y_min_l),
                        (x_max_l, y_min_l),
                        (x_max_l, y_max_l),
                        (x_min_l, y_max_l)
                    ]
                    w = smaller_box.bounds[2] - smaller_box.bounds[0]
                    h = smaller_box.bounds[3] - smaller_box.bounds[1]
                    possible_positions = []
                    for corner in corners:
                        cx, cy = corner
                        if corner == (x_min_l, y_min_l):
                            new_xmin = cx
                            new_ymin = cy
                            new_xmax = cx + w
                            new_ymax = cy + h
                        elif corner == (x_max_l, y_min_l):
                            new_xmin = cx - w
                            new_ymin = cy
                            new_xmax = cx
                            new_ymax = cy + h
                        elif corner == (x_max_l, y_max_l):
                            new_xmin = cx - w
                            new_ymin = cy - h
                            new_xmax = cx
                            new_ymax = cy
                        else:
                            new_xmin = cx
                            new_ymin = cy - h
                            new_xmax = cx + w
                            new_ymax = cy
                        if (new_xmin >= x_min_l and new_ymin >= y_min_l and
                            new_xmax <= x_max_l and new_ymax <= y_max_l):
                            possible_positions.append((new_xmin, new_ymin, new_xmax, new_ymax))
                    if possible_positions:
                        original_center_x = (smaller_box.bounds[0] + smaller_box.bounds[2]) / 2
                        original_center_y = (smaller_box.bounds[1] + smaller_box.bounds[3]) / 2
                        min_dist = float('inf')
                        best_pos = None
                        for pos in possible_positions:
                            new_center_x = (pos[0] + pos[2]) / 2
                            new_center_y = (pos[1] + pos[3]) / 2
                            dist = ((new_center_x - original_center_x)**2 + (new_center_y - original_center_y)**2)**0.5
                            if dist < min_dist:
                                min_dist = dist
                                best_pos = pos
                        adjusted_boxes[smaller_idx] = box(*best_pos)
                        moved = True
                        print(f"MOVED smaller Room {smaller_idx+1} to corner position {best_pos} maintaining 100% overlap.")
                elif current_overlap > max_overlap_threshold:
                    best_dir = None
                    min_ov = current_overlap
                    original = adjusted_boxes[i]
                    for dx in [-shift_amount, 0, shift_amount]:
                        for dy in [-shift_amount, 0, shift_amount]:
                            if dx == 0 and dy == 0:
                                continue
                            nb = move_box(original, dx, dy)
                            ov = overlap_percentage(nb, adjusted_boxes[j])
                            if ov < min_ov:
                                min_ov = ov
                                best_dir = (dx, dy)
                    if best_dir:
                        adjusted_boxes[i] = move_box(original, *best_dir)
                        moved = True
                        print(f"MOVED Room {i+1} by {best_dir} to reduce overlap from {current_overlap:.1f}% to {min_ov:.1f}%.")
        if not moved and max_overlap > max_overlap_threshold:
            print(f"No movement possible in attempt {attempt+1}, but max overlap {max_overlap:.1f}% > {max_overlap_threshold}%. Continuing...")
    if attempt == max_attempts - 1:
        print(f"Max attempts ({max_attempts}) reached. Final maximum overlap is {max_overlap:.1f}%.")
    return [list(bbox.bounds) for bbox in adjusted_boxes]

def get_box_lines(bbox):
    x_min, y_min, x_max, y_max = bbox
    return [
        LineString([(x_min, y_min), (x_max, y_min)]),  # Bottom
        LineString([(x_max, y_min), (x_max, y_max)]),  # Right
        LineString([(x_max, y_max), (x_min, y_max)]),  # Top
        LineString([(x_min, y_max), (x_min, y_min)])   # Left
    ]

def remove_overlapping_lines_from_lines(lines, smaller_box):
    small_poly = Polygon([
        (smaller_box[0], smaller_box[1]),
        (smaller_box[2], smaller_box[1]),
        (smaller_box[2], smaller_box[3]),
        (smaller_box[0], smaller_box[3])
    ])
    visible_lines = []
    for line in lines:
        diff = line.difference(small_poly)
        if diff and not diff.is_empty:
            if isinstance(diff, LineString):
                visible_lines.append(diff)
            else:
                visible_lines.extend([seg for seg in diff.geoms if isinstance(seg, LineString)])
    return visible_lines

# ---------------------- 3D Wall Extrusion Functions ----------------------
def extrude_vertical_wall(x, y_start, y_end, thickness, height, side):
    offset = -thickness if side == 'left' else thickness
    B1 = (x, y_start, 0)
    B2 = (x, y_end, 0)
    B3 = (x + offset, y_end, 0)
    B4 = (x + offset, y_start, 0)
    T1 = (x, y_start, height)
    T2 = (x, y_end, height)
    T3 = (x + offset, y_end, height)
    T4 = (x + offset, y_start, height)
    faces = [[B1, B2, B3, B4], [T1, T2, T3, T4], [B1, B2, T2, T1],
             [B2, B3, T3, T2], [B3, B4, T4, T3], [B4, B1, T1, T4]]
    return faces

def extrude_horizontal_wall(y, x_start, x_end, thickness, height, side):
    offset = -thickness if side == 'bottom' else thickness
    B1 = (x_start, y, 0)
    B2 = (x_end, y, 0)
    B3 = (x_end, y + offset, 0)
    B4 = (x_start, y + offset, 0)
    T1 = (x_start, y, height)
    T2 = (x_end, y, height)
    T3 = (x_end, y + offset, height)
    T4 = (x_start, y + offset, height)
    faces = [[B1, B2, B3, B4], [T1, T2, T3, T4], [B1, B2, T2, T1],
             [B2, B3, T3, T2], [B3, B4, T4, T3], [B4, B1, T1, T4]]
    return faces

# ---------------------- Main Function ----------------------
def plot_3d_house_plan():
    # ---------------------- Data Loading and Preprocessing ----------------------
    X_np = np.load(r'C:\Users\SMART TECH\Desktop\New folder (3)\Architexture-AI1\BackendProject\Model_Implimentaion\room_matrix.npy', allow_pickle=True).astype(np.float32)
    A_np = np.load(r'C:\Users\SMART TECH\Desktop\New folder (3)\Architexture-AI1\BackendProject\Model_Implimentaion\adjacency_matrix.npy', allow_pickle=True).astype(np.float32)
    X = torch.tensor(X_np, dtype=torch.float32)
    A = torch.tensor(A_np, dtype=torch.float32)
    A = A.reshape(A.shape[0], -1)
    print("Shape of X:", X.shape)
    print("Shape of A:", A.shape)

    # ---------------------- Model Loading ----------------------
    model = GraphTransformerMLPModel(
        feature_dim=33,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        mlp_hidden_features=512
    )
    model.load_state_dict(torch.load(r'C:\Users\SMART TECH\Desktop\New folder (3)\Architexture-AI1\BackendProject\Model_Implimentaion\GraphTransformer.pth'))
    print("Model state dictionary loaded successfully!")

    # ---------------------- Prediction ----------------------
    model.eval()
    with torch.no_grad():
        predictions = model(X, A)
    print("Predictions:", predictions)

    # ---------------------- Bounding Box Processing and Clipping ----------------------
    bboxes = predictions.tolist()
    filtered_bboxes = []
    valid_indices = []
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        if xmin < xmax and ymin < ymax:
            filtered_bboxes.append(box(xmin, ymin, xmax, ymax))
            valid_indices.append(i)
    room_categories = ['livingroom', 'kitchen', 'balcony', 'bedroom', 'washroom', 'studyroom', 'closet', 'storage', 'corridor']
    room_types = [room_categories[np.argmax(X_np[idx, :9])] for idx in valid_indices]
    room_sizes = [X_np[idx, 9] for idx in valid_indices]

    # Postprocessing
    adjusted_bboxes = postprocess_bounding_boxes(filtered_bboxes, shift_amount=2)
    adjusted_bboxes_list = [list(bbox.bounds) for bbox in adjusted_bboxes]
    adjusted_bboxes_list = reduce_overlap(adjusted_bboxes_list, max_overlap_threshold=35, shift_amount=2, max_attempts=50)
    adjusted_bboxes_list = merge_close_parallel_walls(adjusted_bboxes_list, 
                                                    distance_threshold=15, 
                                                    angle_threshold=15)
    room_data = list(zip(adjusted_bboxes_list, room_types, room_sizes))
    room_data.sort(key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]), reverse=True)
    bboxes, room_types, room_sizes = zip(*room_data)
    bboxes = list(bboxes)
    room_types = list(room_types)
    room_sizes = list(room_sizes)

    # Scale bounding boxes for 3D visualization
    scale_factor = 500
    bboxes = [[coord / scale_factor for coord in bbox] for bbox in bboxes]
    bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    remaining_lines = {}
    for i, big_box in enumerate(bboxes):
        current_lines = get_box_lines(big_box)
        for j in range(i + 1, len(bboxes)):
            current_lines = remove_overlapping_lines_from_lines(current_lines, bboxes[j])
        remaining_lines[i] = current_lines
    # ---------------------- Generate 3D Polygons ----------------------
# ---------------------- Generate 3D Polygons ----------------------
    extrude_height = 0.006
    wall_thickness = 0.005
    all_wall_polys = []
    ground_polys = []

    for i, lines in remaining_lines.items():
        bbox = bboxes[i]
        x_min, y_min, x_max, y_max = bbox
        for line in lines:
            x, y = line.xy
            if len(x) == 2:
                x_start, x_end = x[0], x[1]
                y_start, y_end = y[0], y[1]
                if abs(x_start - x_end) < 1e-6:
                    side = 'left' if abs(x_start - x_min) < 1e-6 else 'right'
                    all_wall_polys += extrude_vertical_wall(x_start, y_start, y_end, wall_thickness, extrude_height, side)
                else:
                    side = 'bottom' if abs(y_start - y_min) < 1e-6 else 'top'
                    all_wall_polys += extrude_horizontal_wall(y_start, x_start, x_end, wall_thickness, extrude_height, side)

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        ground_face = [
            (x_min, y_min, 0),
            (x_max, y_min, 0),
            (x_max, y_max, 0),
            (x_min, y_max, 0)
        ]
        ground_polys.append(ground_face)
            # Extrude walls
    all_wall_polys += extrude_vertical_wall(x_min, y_min, y_max, wall_thickness, extrude_height, 'left')
    all_wall_polys += extrude_vertical_wall(x_max, y_min, y_max, wall_thickness, extrude_height, 'right')
    all_wall_polys += extrude_horizontal_wall(y_min, x_min, x_max, wall_thickness, extrude_height, 'bottom')
    all_wall_polys += extrude_horizontal_wall(y_max, x_min, x_max, wall_thickness, extrude_height, 'top')
    
    wall_colors = np.full((len(all_wall_polys), 3), [240/255, 240/255, 240/255])  # Silver
    ground_colors = np.full((len(ground_polys), 3), [200/255, 170/255, 120/255])  # Wood

    # ---------------------- 3D Plotting ----------------------
    fig = plt.figure(figsize=(16, 12))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.add_collection3d(Poly3DCollection(all_wall_polys, facecolors=wall_colors, edgecolors='black', linewidths=0.5, alpha=0.8))
    ax3d.add_collection3d(Poly3DCollection(ground_polys, facecolors=ground_colors, edgecolors='black', linewidths=0.5, alpha=1.0))

    all_x = np.concatenate([np.array(bboxes)[:, [0, 2]].ravel()])
    all_y = np.concatenate([np.array(bboxes)[:, [1, 3]].ravel()])
    ax3d.set_xlim(all_x.min() - 0.01, all_x.max() + 0.01)
    ax3d.set_ylim(all_y.min() - 0.01, all_y.max() + 0.01)
    ax3d.set_zlim(0, extrude_height + 0.05)

    ax3d.grid(False)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor('white')
    ax3d.yaxis.pane.set_edgecolor('white')
    ax3d.zaxis.pane.set_edgecolor('white')
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.set_xlabel("")
    ax3d.set_ylabel("")
    ax3d.set_zlabel("")
    ax3d.set_axis_off()

    ax3d.view_init(elev=55, azim=240)







    # # ---------------------- 3D save ----------------------
    # # Combine all polygons (walls and ground) into a single list for simplicity
    # # Combine all polygons (walls and ground) into a single list for simplicity
    # all_polys = all_wall_polys + ground_polys
    # all_colors = np.concatenate([wall_colors, ground_colors], axis=0)

    # # Step 1: Flatten vertices and create indices
    # vertices = []
    # indices = []
    # vertex_offset = 0

    # for poly in all_polys:
    #     for vertex in poly:
    #         vertices.extend(vertex)  # Each vertex is (x, y, z)
    #     # Define triangle indices for this quad (two triangles per face)
    #     indices.extend([
    #         vertex_offset, vertex_offset + 1, vertex_offset + 2,  # First triangle
    #         vertex_offset, vertex_offset + 2, vertex_offset + 3   # Second triangle
    #     ])
    #     vertex_offset += 4

    # vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)  # Shape: (N, 3)
    # indices = np.array(indices, dtype=np.uint32)                    # Shape: (M,)

    # # Step 2: Prepare colors (one color per face, repeated for each vertex of the face)
    # colors = []
    # for i, poly in enumerate(all_polys):
    #     color = all_colors[i]
    #     for _ in range(4):  # Each face has 4 vertices
    #         colors.extend([color[0], color[1], color[2], 1.0])  # RGBA
    # colors = np.array(colors, dtype=np.float32).reshape(-1, 4)  # Shape: (N, 4)

    # # Step 3: Create binary data for buffers
    # vertex_data = vertices.tobytes()
    # index_data = indices.tobytes()
    # color_data = colors.tobytes()
    # binary_blob = vertex_data + index_data + color_data

    # # Step 4: Set up glTF structure
    # gltf = GLTF2()

    # # Buffer
    # buffer = Buffer()
    # buffer.byteLength = len(binary_blob)
    # gltf.buffers.append(buffer)

    # # Buffer Views
    # vertex_buffer_view = BufferView()
    # vertex_buffer_view.buffer = 0
    # vertex_buffer_view.byteOffset = 0
    # vertex_buffer_view.byteLength = len(vertex_data)
    # vertex_buffer_view.target = 34962  # ARRAY_BUFFER
    # gltf.bufferViews.append(vertex_buffer_view)

    # index_buffer_view = BufferView()
    # index_buffer_view.buffer = 0
    # index_buffer_view.byteOffset = len(vertex_data)
    # index_buffer_view.byteLength = len(index_data)
    # index_buffer_view.target = 34963  # ELEMENT_ARRAY_BUFFER
    # gltf.bufferViews.append(index_buffer_view)

    # color_buffer_view = BufferView()
    # color_buffer_view.buffer = 0
    # color_buffer_view.byteOffset = len(vertex_data) + len(index_data)
    # color_buffer_view.byteLength = len(color_data)
    # color_buffer_view.target = 34962  # ARRAY_BUFFER
    # gltf.bufferViews.append(color_buffer_view)

    # # Accessors
    # vertex_accessor = Accessor()
    # vertex_accessor.bufferView = 0
    # vertex_accessor.byteOffset = 0
    # vertex_accessor.componentType = 5126  # FLOAT
    # vertex_accessor.count = len(vertices)
    # vertex_accessor.type = "VEC3"
    # vertex_accessor.min = vertices.min(axis=0).tolist()
    # vertex_accessor.max = vertices.max(axis=0).tolist()
    # gltf.accessors.append(vertex_accessor)

    # index_accessor = Accessor()
    # index_accessor.bufferView = 1
    # index_accessor.byteOffset = 0
    # index_accessor.componentType = 5125  # UNSIGNED_INT
    # index_accessor.count = len(indices)
    # index_accessor.type = "SCALAR"
    # gltf.accessors.append(index_accessor)

    # color_accessor = Accessor()
    # color_accessor.bufferView = 2
    # color_accessor.byteOffset = 0
    # color_accessor.componentType = 5126  # FLOAT
    # color_accessor.count = len(colors)
    # color_accessor.type = "VEC4"
    # gltf.accessors.append(color_accessor)

    # # Material
    # material = Material()
    # material.pbrMetallicRoughness = {
    #     "baseColorFactor": [1.0, 1.0, 1.0, 1.0],  # Default white, overridden by vertex colors
    #     "metallicFactor": 0.0,
    #     "roughnessFactor": 1.0
    # }
    # material.doubleSided = True
    # gltf.materials.append(material)

    # # Primitive
    # primitive = Primitive()
    # primitive.attributes = {"POSITION": 0, "COLOR_0": 2}  # Vertex positions and colors
    # primitive.indices = 1  # Indices
    # primitive.material = 0  # Reference to material
    # primitive.mode = 4  # TRIANGLES

    # # Mesh
    # mesh = Mesh()
    # mesh.primitives = [primitive]
    # gltf.meshes.append(mesh)

    # # Node
    # node = Node()
    # node.mesh = 0
    # gltf.nodes.append(node)

    # # Scene
    # scene = Scene()
    # scene.nodes = [0]
    # gltf.scenes.append(scene)
    # gltf.scene = 0

    # # Step 5: Set binary data explicitly
    # gltf.set_binary_blob(binary_blob)

    # # Step 6: Save as .glb
    # gltf.save("house_plan_3d.glb")
    # print("Saved 3D model as 'house_plan_3d.glb'!")



    # ---------------------- 3D Plotting ----------------------

    plt.tight_layout()
    # plt.savefig("house_plan_3d_top.png", dpi=300, bbox_inches='tight')  # Save as PNG
    # plt.show()

    image_id = uuid.uuid4().hex
    image_filename = f"final_plot_{image_id}.png"
    image_dir = r"C:\Users\SMART TECH\Desktop\New folder (3)\Architexture-AI1\BackendProject\3D_plot"
    image_path = os.path.join(image_dir, image_filename)

    # Step 6: Save as .glb
    plt.savefig(image_path, dpi=300, bbox_inches='tight')  # Save as PNG
    image_id = uuid.uuid4().hex
    image_filename = f"final_plot_{image_id}.png"
    image_dir = r"C:\Users\SMART TECH\Desktop\New folder (3)\Architexture-AI1\BackendProject\images"
    image_path = os.path.join(image_dir, image_filename)

    # Step 6: Save as .glb
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print("Saved 3D model as 'house_plan_3d.png'!")
    return image_path