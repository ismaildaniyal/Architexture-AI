import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, box
from matplotlib.lines import Line2D
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from shapely.ops import unary_union
import networkx as nx

# ----------------------
# Model Definition
# ----------------------

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
        X_unsq = X.unsqueeze(1)
        attn_out, _ = self.attn(X_unsq, X_unsq, X_unsq)
        X_norm = self.norm1(X_unsq + attn_out)
        mlp_out = self.mlp(X_norm)
        X_out = self.norm2(X_norm + mlp_out)
        return X_out.squeeze(1)

class GraphTransformer(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_layers=2, num_heads=4):
        super(GraphTransformer, self).__init__()
        self.embedding = nn.Linear(in_features, hidden_dim)
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)])
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


# ----------------------
# PostProcessing Functions
# ----------------------

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
            print(f"✅ All rooms connected after {attempt + 1} adjustments.")
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
                    print(f"🔹 Room {idx} connected by shifting ({dx}, {dy})")
                    return adjusted_boxes
    print("⚠️ Max attempts reached! Some rooms might still be disconnected.")
    return adjusted_boxes

def postprocess_bounding_boxes(bboxes, shift_amount=5):
    print("🔄 Building initial graph...")
    G = build_graph(bboxes)
    if is_fully_connected(G):
        print("✅ All rooms are already connected.")
        return bboxes
    print("⚠️ Disconnected rooms detected! Fixing...")
    return attach_disconnected_rooms(bboxes, shift_amount)

def reduce_overlap(bboxes, max_overlap_threshold=35, shift_amount=10, max_attempts=50):
    """Reduce overlap between bounding boxes iteratively based on specified criteria."""
    adjusted_boxes = [box(*bbox) for bbox in bboxes]
    directions = [
        (shift_amount, 0), (-shift_amount, 0),  # Right, Left
        (0, shift_amount), (0, -shift_amount),  # Down, Up
        (shift_amount, shift_amount), (-shift_amount, -shift_amount),  # Diagonal Right-Down, Left-Up
        (shift_amount, -shift_amount), (-shift_amount, shift_amount)  # Diagonal Right-Up, Left-Down
    ]
    
    for attempt in range(max_attempts):
        # Check if any overlap exceeds the threshold
        max_overlap = 0
        for i in range(len(adjusted_boxes)):
            for j in range(i + 1, len(adjusted_boxes)):
                overlap = overlap_percentage(adjusted_boxes[i], adjusted_boxes[j])
                max_overlap = max(max_overlap, overlap)
        
        # Stop if all overlaps are at or below 35%
        if max_overlap <= max_overlap_threshold:
            print(f"✅ Overlap reduction complete. Maximum overlap is {max_overlap:.1f}% (≤ {max_overlap_threshold}%).")
            break
        
        # Process pairs with overlap > 35%
        moved = False
        for i in range(len(adjusted_boxes)):
            for j in range(i + 1, len(adjusted_boxes)):
                current_overlap = overlap_percentage(adjusted_boxes[i], adjusted_boxes[j])
                # Only adjust if overlap exceeds 35%
                if current_overlap > max_overlap_threshold:
                    best_direction = None
                    min_overlap = current_overlap
                    original_box = adjusted_boxes[i]
                    
                    for dx, dy in directions:
                        new_box = move_box(original_box, dx, dy)
                        new_overlap = overlap_percentage(new_box, adjusted_boxes[j])
                        if new_overlap < min_overlap:
                            min_overlap = new_overlap
                            best_direction = (dx, dy)
                    
                    if best_direction:
                        adjusted_boxes[i] = move_box(original_box, *best_direction)
                        print(f"🔹 Moved Room {i+1} by {best_direction} to reduce overlap from {current_overlap:.1f}% to {min_overlap:.1f}%.")
                        moved = True
        
        # If no moves were made but max_overlap > 35%, continue to the next iteration
        if not moved and max_overlap > max_overlap_threshold:
            print(f"⚠️ No movement possible in attempt {attempt+1}, but max overlap {max_overlap:.1f}% > {max_overlap_threshold}%. Continuing...")
    
    # If max_attempts is reached, report the final state
    if attempt == max_attempts - 1:
        print(f"⚠️ Max attempts ({max_attempts}) reached. Final maximum overlap is {max_overlap:.1f}%.")
    
    return [list(bbox.bounds) for bbox in adjusted_boxes]

def add_window_per_polygon(polygons, outer_walls, window_length=10, corner_buffer=2):
    windows = []
    for poly in polygons:
        coords = list(poly.exterior.coords)[:-1]
        valid_lines = []
        for i in range(len(coords)):
            line_start = coords[i]
            line_end = coords[(i + 1) % len(coords)]
            line = LineString([line_start, line_end])
            if line.intersects(outer_walls) and not line.disjoint(outer_walls):
                intersection = line.intersection(outer_walls)
                if intersection.length >= 0.9 * line.length:
                    valid_lines.append((line, line.length))
        if valid_lines:
            valid_lines.sort(key=lambda x: x[1], reverse=True)
            line, length = valid_lines[0]
            effective_length = length - 2 * corner_buffer
            if effective_length > window_length:
                start_pos = corner_buffer + (effective_length - window_length) / 2
                end_pos = start_pos + window_length
            else:
                start_pos = corner_buffer
                end_pos = length - corner_buffer
            start = line.interpolate(start_pos)
            end = line.interpolate(end_pos)
            windows.append(LineString([start, end]))
        else:
            windows.append(None)
    return windows

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


def main():
    # ----------------------
    # Data Loading and Preprocessing
    # ----------------------

    X_np = np.load(r'C:\Users\SMART TECH\Desktop\New folder (2)\Architexture-AI\BackendProject\Model_Implimentaion\room_matrix.npy', allow_pickle=True).astype(np.float32)
    A_np = np.load(r'C:\Users\SMART TECH\Desktop\New folder (2)\Architexture-AI\BackendProject\Model_Implimentaion\adjacency_matrix.npy', allow_pickle=True).astype(np.float32)
    X = torch.tensor(X_np, dtype=torch.float32)
    A = torch.tensor(A_np, dtype=torch.float32)
    A = A.reshape(A.shape[0], -1)
    print("Shape of X:", X.shape)
    print("Shape of A:", A.shape)

    # ----------------------
    # Model Loading and Prediction
    # ----------------------

    model = GraphTransformerMLPModel(feature_dim=33, hidden_dim=64, num_layers=3, num_heads=4, mlp_hidden_features=512)
    model.load_state_dict(torch.load(r'C:\Users\SMART TECH\Desktop\New folder (2)\Architexture-AI\BackendProject\Model_Implimentaion\GraphTransformer.pth'))
    print("Model state dictionary loaded successfully!")

    model.eval()
    with torch.no_grad():
        predictions = model(X, A)
    print("Predictions:", predictions)

    # ----------------------
    # Bounding Box Processing with Room Info
    # ----------------------

    bboxes = predictions.tolist()

    # Filter invalid boxes and track indices
    filtered_bboxes = []
    valid_indices = []
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        if xmin < xmax and ymin < ymax:
            filtered_bboxes.append(box(xmin, ymin, xmax, ymax))
            valid_indices.append(i)

    # Extract room types and sizes
    room_categories = ['livingroom', 'kitchen', 'balcony', 'bedroom', 'washroom', 'studyroom', 'closet', 'storage', 'corridor']
    room_types = [room_categories[np.argmax(X_np[idx, :9])] for idx in valid_indices]
    room_sizes = [X_np[idx, 9] for idx in valid_indices]

    # Adjust bounding boxes to ensure connectivity
    adjusted_bboxes = postprocess_bounding_boxes(filtered_bboxes, shift_amount=2)
    adjusted_bboxes_list = [list(bbox.bounds) for bbox in adjusted_bboxes]

    # Reduce overlap with new criteria
    adjusted_bboxes_list = reduce_overlap(adjusted_bboxes_list, max_overlap_threshold=35, shift_amount=2, max_attempts=50)

    # Sort bboxes, room_types, and room_sizes by area (descending)
    room_data = list(zip(adjusted_bboxes_list, room_types, room_sizes))
    room_data.sort(key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]), reverse=True)
    bboxes, room_types, room_sizes = zip(*room_data)
    bboxes = list(bboxes)
    room_types = list(room_types)
    room_sizes = list(room_sizes)

    # # ----------------------
    # # First Plot (Original Boxes)
    # # ----------------------

    # plt.figure(figsize=(6, 6))
    # ax = plt.gca()

    # for i, bbox in enumerate(bboxes):
    #     box_lines = get_box_lines(bbox)
    #     print(f"Bounding Box {i+1}: {bbox}")
    #     for line in box_lines:
    #         x, y = line.xy
    #         ax.plot(x, y, 'b', linewidth=2)
    #     x_center = (bbox[0] + bbox[2]) / 2
    #     y_center = (bbox[1] + bbox[3]) / 2
    #     if 0 <= x_center <= 500 and 0 <= y_center <= 500:
    #         ax.text(x_center, y_center, f"Room {i+1}", fontsize=10, color='black', ha='center', va='center',
    #                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # # Dynamically set plot limits based on bounding box coordinates
    # xmin = min(bbox[0] for bbox in bboxes)
    # ymin = min(bbox[1] for bbox in bboxes)
    # xmax = max(bbox[2] for bbox in bboxes)
    # ymax = max(bbox[3] for bbox in bboxes)
    # padding = 10
    # plt.xlim(xmin - padding, xmax + padding)
    # plt.ylim(ymin - padding, ymax + padding)

    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.gca().invert_yaxis()
    # plt.title("Original Bounding Boxes (Before Overlap Removal)")
    # plt.show()

    # ----------------------
    # Second Plot (Processed Boxes with Legend)
    # ----------------------

    remaining_lines = {}
    for i, big_box in enumerate(bboxes):
        current_lines = get_box_lines(big_box)
        for j in range(i + 1, len(bboxes)):
            current_lines = remove_overlapping_lines_from_lines(current_lines, bboxes[j])
        remaining_lines[i] = current_lines

    plt.figure(figsize=(8, 6))  # Increased width for legend
    ax = plt.gca()

    # Create polygons and outer walls
    room_polygons = [Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]),
                            (bbox[2], bbox[3]), (bbox[0], bbox[3])]) for bbox in bboxes]
    outer_walls = unary_union(room_polygons).boundary
    windows = add_window_per_polygon(room_polygons, outer_walls, window_length=20)

    # Process windows and outer walls
    valid_windows = [w for w in windows if w is not None]
    if valid_windows:
        buffered_windows = [w.buffer(0.1) for w in valid_windows]
        window_union = unary_union(buffered_windows)
        modified_outer_walls = outer_walls.difference(window_union)
    else:
        modified_outer_walls = outer_walls

    # Plot lines, room numbers, outer walls, and windows
    for i, (bbox, lines) in enumerate(zip(bboxes, remaining_lines.values())):
        for line in lines:
            x, y = line.xy
            ax.plot(x, y, 'black', linewidth=1.5)
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        ax.text(x_center, y_center, f"R{i+1}", ha='center', va='center', fontsize=8, color='black')

    if modified_outer_walls.geom_type == "LineString":
        x_outer, y_outer = modified_outer_walls.xy
        ax.plot(x_outer, y_outer, 'black', linewidth=4)
    elif modified_outer_walls.geom_type == "MultiLineString":
        for line in modified_outer_walls.geoms:
            x_outer, y_outer = line.xy
            ax.plot(x_outer, y_outer, 'black', linewidth=4)

    for window in windows:
        if window is not None:
            x_w, y_w = window.xy
            ax.plot(x_w, y_w, color='grey', linewidth=4)

    # Dynamically set plot limits based on bounding box coordinates
    xmin = min(bbox[0] for bbox in bboxes)
    ymin = min(bbox[1] for bbox in bboxes)
    xmax = max(bbox[2] for bbox in bboxes)
    ymax = max(bbox[3] for bbox in bboxes)
    padding = 10
    plt.xlim(xmin - padding, xmax + padding)
    plt.ylim(ymin - padding, ymax + padding)

    # Add legend with room types and sizes
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Room {i+1}: {room_types[i]}, Size: {room_sizes[i]:.1f}', markersize=0)
        for i in range(len(bboxes))
    ]
    legend_elements += [
        Line2D([0], [0], color='grey', linewidth=2, label='Windows')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=6, ncol=2)

    # Your plotting code here...
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    plt.gca().invert_yaxis()
    # plt.title("Filtered Bounding Boxes with Outer Walls and Windows")
    plt.tight_layout()

    # Remove axis scales (ticks and numbers)
    plt.xticks([])  # Hide x-axis ticks
    plt.yticks([])  # Hide y-axis ticks
    plt.gca().set_frame_on(True)  # Optional: Hide border if needed
    image_id = uuid.uuid4().hex
    image_filename = f"final_plot_{image_id}.png"
    image_dir = r"C:\Users\SMART TECH\Desktop\New folder (2)\Architexture-AI\BackendProject\images"
    image_path = os.path.join(image_dir, image_filename)
    plt.savefig(image_path)  # Save before displaying
    # plt.show()

    print("Plot saved as 'final_plot.png'")
    return image_path
