import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# KITTI 클래스 이름
CLASS_NAMES = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
    'Cyclist', 'Tram', 'Misc', 'DontCare'
]

def parse_kitti_3d_pred(pred_file):
    results = []
    with open(pred_file, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith("//"):
                continue
            parts = line.strip().split()
            class_name = parts[0]
            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            ry = float(parts[14])
            class_id = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else len(CLASS_NAMES)-1
            results.append({'class_id': class_id, 'h': h, 'w': w, 'l': l, 'x': x, 'y': y, 'z': z, 'ry': ry})
    return results

def read_calib_p2(calib_path):
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                vals = [float(x) for x in line.strip().split()[1:]]
                return np.array(vals).reshape(3, 4)
    raise ValueError("P2 matrix not found in calib file.")

def get_3d_box_corners(obj):
    l, h, w = obj['l'], obj['h'], obj['w']
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, -h, -h, 0, 0, -h, -h, 0]
    z_corners = [w/2, w/2, w/2, w/2, -w/2, -w/2, -w/2, -w/2]
    corners = np.array([x_corners, y_corners, z_corners])
    R = np.array([
        [np.cos(obj['ry']), 0, np.sin(obj['ry'])],
        [0, 1, 0],
        [-np.sin(obj['ry']), 0, np.cos(obj['ry'])]
    ])
    corners_rot = R @ corners
    corners_rot[0, :] += obj['x']
    corners_rot[1, :] += obj['y']
    corners_rot[2, :] += obj['z']
    return corners_rot

def project_corners_to_img(corners_3d, P):
    n = corners_3d.shape[1]
    homo = np.vstack((corners_3d, np.ones((1, n))))
    proj = P @ homo
    proj[:2] /= proj[2]
    return proj[:2].T.astype(np.int32)

def draw_3d_box(img, pts, class_name=None, score=None, color=(0,255,0), thickness=2):
    # 3D box의 12개 edge를 연결 (선 두께 2로 변경)
    edges = [
        (0,1), (1,2), (2,3), (3,0), # 아래
        (4,5), (5,6), (6,7), (7,4), # 위
        (0,4), (1,5), (2,6), (3,7)  # 옆
    ]
    for s, e in edges:
        cv2.line(img, tuple(pts[s]), tuple(pts[e]), color, thickness)
    # 클래스명만 텍스트로 표시 (글씨 두께 2로 더 두껍게)
    if class_name is not None:
        top_indices = [4, 5, 6, 7]
        x_mean = int(np.mean(pts[top_indices, 0]))
        y_min = int(np.min(pts[top_indices, 1]))
        cv2.putText(img, class_name, (x_mean, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return img

def visualize_3d_boxes_on_image(image_path, calib_path, pred_file):
    if not (os.path.exists(image_path) and os.path.exists(calib_path) and os.path.exists(pred_file)):
        print("파일 경로를 확인하세요.")
        return
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    P2 = read_calib_p2(calib_path)
    preds = parse_kitti_3d_pred(pred_file)
    for obj in preds:
        corners_3d = get_3d_box_corners(obj)
        pts2d = project_corners_to_img(corners_3d, P2)
        class_name = CLASS_NAMES[obj['class_id']]
        score = 1.0  # 예측 파일에 score가 없으므로 1.0 고정
        img = draw_3d_box(img, pts2d, class_name=class_name, score=score, color=(0, 255, 0))
    plt.figure(figsize=(12,8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def visualize_and_save_all(pred_dir, image_dir, calib_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]
    for pred_file in pred_files:
        base = os.path.splitext(pred_file)[0]
        image_path = os.path.join(image_dir, f"{base}.png")
        calib_path = os.path.join(calib_dir, f"{base}.txt")
        pred_path = os.path.join(pred_dir, pred_file)
        if not (os.path.exists(image_path) and os.path.exists(calib_path) and os.path.exists(pred_path)):
            print(f"파일 없음: {base}")
            continue
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        P2 = read_calib_p2(calib_path)
        preds = parse_kitti_3d_pred(pred_path)
        for obj in preds:
            corners_3d = get_3d_box_corners(obj)
            pts2d = project_corners_to_img(corners_3d, P2)
            class_name = CLASS_NAMES[obj['class_id']]
            img = draw_3d_box(img, pts2d, class_name=class_name, color=(0, 255, 0))
        save_path = os.path.join(save_dir, f"{base}_vis.png")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"저장 완료: {save_path}")

# 사용 예시
if __name__ == "__main__":
    #multi_mse_all(clean)
    pred_dir = os.path.join("clean", "pred(multi_mse_all)")
    image_dir = os.path.join("clean", "image_2")
    calib_dir = os.path.join("clean", "calib")
    save_dir = "./result/multi_mse_all(clean)"
    visualize_and_save_all(pred_dir, image_dir, calib_dir, save_dir)

    #multi_ mse_all(foggy)
    pred_dir = os.path.join("foggy", "pred(multi_mse_all)")
    image_dir = os.path.join("foggy", "image_2")
    calib_dir = os.path.join("foggy", "calib")
    save_dir = "./result/multi_mse_all(foggy)"
    visualize_and_save_all(pred_dir, image_dir, calib_dir, save_dir)