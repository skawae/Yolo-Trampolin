import cv2
import torch
import numpy as np
import csv
from ultralytics import YOLO
from collections import deque
import os
import time

# ==========================================================
# 調整パラメータ設定エリア
# ==========================================================
VIDEO_PATH = "../MediapipePose/video-export/kawae0428-7-2.mp4"  

# 使用する台の実際のサイズ（メートル単位）
BLUE_REAL_WIDTH = 5.05   
BLUE_REAL_LENGTH = 2.91  
RED_REAL_WIDTH = 2.15   
RED_REAL_LENGTH = 1.08  

# 鳥瞰図の位置ズレ修正パラメータ（メートル単位）
OFFSET_X = 0.0  
OFFSET_Y = -0.5  

# 骨格全体のブレ抑制（1〜5で調整）
SMOOTHING_FRAMES = 4       
DETECTION_CONFIDENCE = 0.4     

# ダブルクリック誤検知防止用のインターバル（秒単位）
CLICK_INTERVAL = 0.4 

# [追加パラメーター] ピクセルからメートルへの簡易変換係数 (1ピクセル辺り何メートルか)
# 本来はキャリブレーションから厳密に算出しますが、高さや速度計算の基準値として定義します。
PIXEL_TO_METER = 0.005  # 例: 1ピクセル = 5mm 
# ==========================================================

video_base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_VIDEO_PATH = f"{video_base_name}_skeleton_output.mp4"

clicked_points = []
all_landing_points = []  
x_history = deque(maxlen=SMOOTHING_FRAMES)
y_history = deque(maxlen=SMOOTHING_FRAMES)

last_click_time = 0.0

GUIDE_TEXTS = [
    "1/8: Click BLUE [Top-Left]", "2/8: Click BLUE [Top-Right]",
    "3/8: Click BLUE [Bottom-Right]", "4/8: Click BLUE [Bottom-Left]",
    "5/8: Click RED  [Top-Left]", "6/8: Click RED  [Top-Right]",
    "7/8: Click RED  [Bottom-Right]", "8/8: Click RED  [Bottom-Left]"
]

def mouse_callback(event, x, y, flags, param):
    global clicked_points, last_click_time
    if event == cv2.EVENT_LBUTTONDOWN:
        current_time = time.time()
        if current_time - last_click_time >= CLICK_INTERVAL:
            if len(clicked_points) < 8:
                clicked_points.append([x, y])
                print(f"座標登録 [{len(clicked_points)}/8]: X={x}, Y={y}")
                last_click_time = current_time
        else:
            print("警告: 連続クリックを検知したため無視しました。")

# 2つのベクトル間の角度(0〜180度)を計算するヘルパー関数
def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# 基準線(ベクトル)に対する特定のベクトルの傾き(符号付き角度)を計算するヘルパー関数
def calculate_line_angle(p1, p2, reference_vector=[0, -1]):
    v = np.array(p2) - np.array(p1)
    norm_v = np.linalg.norm(v)
    norm_ref = np.linalg.norm(reference_vector)
    if norm_v == 0 or norm_ref == 0:
        return 0.0
    dot_product = np.dot(v, reference_vector)
    cos_angle = dot_product / (norm_v * norm_ref)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    # ベクトルの外積方向で前傾・後傾を判定
    if np.cross(reference_vector, v) > 0:
        return angle
    return -angle

def main():
    global clicked_points, all_landing_points, x_history, y_history
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8s-pose.pt").to(device)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: {VIDEO_PATH} を開けません。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): 
        fps = 30.0
    frame_duration = 1.0 / fps  # 1フレームあたりの秒数

    video_writer = None
    landing_images_dir = "landing_snapshots"
    os.makedirs(landing_images_dir, exist_ok=True)

    # フェーズ1: フレーム選択
    win_select = "Select Visible Frame"
    cv2.namedWindow(win_select, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_select, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        display_select = frame.copy()
        cv2.putText(display_select, "PRESS 'SPACE' TO SELECT THIS FRAME", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(win_select, display_select)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:
            init_frame = frame.copy()
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow(win_select)

    # フェーズ2: マウスクリック入力（キャリブレーション）
    win_calib = "Calibration Phase"
    cv2.namedWindow(win_calib, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_calib, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win_calib, mouse_callback)
    
    while len(clicked_points) < 8:
        display_frame = init_frame.copy()
        step = len(clicked_points)
        current_color = (255, 0, 0) if step < 4 else (0, 0, 255)
        cv2.putText(display_frame, GUIDE_TEXTS[step], (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_color, 2)

        blue_pts = clicked_points[0:4]
        for pt in blue_pts: cv2.circle(display_frame, (pt[0], pt[1]), 6, (255, 0, 0), -1)
        if len(blue_pts) > 1: cv2.polylines(display_frame, [np.array(blue_pts, np.int32)], False, (255, 0, 0), 2)
        if len(blue_pts) == 4: cv2.line(display_frame, tuple(blue_pts[3]), tuple(blue_pts[0]), (255, 0, 0), 2)

        red_pts = clicked_points[4:8]
        for pt in red_pts: cv2.circle(display_frame, (pt[0], pt[1]), 6, (0, 0, 255), -1)
        if len(red_pts) > 1: cv2.polylines(display_frame, [np.array(red_pts, np.int32)], False, (0, 0, 255), 2)
        if len(red_pts) == 4: cv2.line(display_frame, tuple(red_pts[3]), tuple(red_pts[0]), (0, 0, 255), 2)

        cv2.imshow(win_calib, display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow(win_calib)

    blue_image_pts = np.array(clicked_points[0:4], dtype=np.float32)
    red_image_pts = np.array(clicked_points[4:8], dtype=np.float32)

    red_real_pts = np.array([
        [-RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2], [ RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2],
        [ RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2], [-RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(red_image_pts, red_real_pts)

    # ------------------------------------------------------
    # 時系列・ダイナミクス計算用の履歴バッファ
    # ------------------------------------------------------
    frame_buffer = []  # 各フレームの全解析データを一時格納するバッファ
    bed_rest_y = np.mean(red_image_pts[:, 1])  # 赤枠の平均Y座標をベッドの静止表面とする

    print("\n--- [手順3] 20項目拡張解析を開始します ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    win_main = "Universal Trampoline System v4"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_main, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    current_frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, verbose=False, conf=DETECTION_CONFIDENCE)
        
        cv2.polylines(frame, [blue_image_pts.astype(np.int32)], True, (255, 0, 0), 3)
        cv2.polylines(frame, [red_image_pts.astype(np.int32)], True, (0, 0, 255), 3)

        frame_data = {"frame_idx": current_frame_idx, "has_pose": False, "raw_frame": frame.copy()}

        if results and results[0].keypoints is not None and len(results[0].keypoints.xy.cpu().numpy()) > 0:
            keypoints_xy = results[0].keypoints.xy.cpu().numpy()[0]
            keypoints_conf = results[0].keypoints.conf.cpu().numpy()[0]
            frame = results[0].plot()
            
            cv2.polylines(frame, [blue_image_pts.astype(np.int32)], True, (255, 0, 0), 3)
            cv2.polylines(frame, [red_image_pts.astype(np.int32)], True, (0, 0, 255), 3)

            # 使用側（半身）の自動判定
            left_score = sum([keypoints_conf[i] for i in [5, 9, 11, 13, 15]])
            right_score = sum([keypoints_conf[i] for i in [6, 10, 12, 14, 16]])
            
            if left_score >= right_score:
                idx_ear, idx_shoulder, idx_elbow, idx_wrist, idx_hip, idx_knee, idx_ankle, idx_toe = 3, 5, 7, 9, 11, 13, 15, 17
                side_used = "LEFT SIDE"
            else:
                idx_ear, idx_shoulder, idx_elbow, idx_wrist, idx_hip, idx_knee, idx_ankle, idx_toe = 4, 6, 8, 10, 12, 14, 16, 17
                side_used = "RIGHT SIDE"

            # 各関節座標の抽出
            pt_nose = keypoints_xy[0]
            pt_ear = keypoints_xy[idx_ear]
            pt_shoulder = keypoints_xy[idx_shoulder]
            pt_elbow = keypoints_xy[idx_elbow]
            pt_wrist = keypoints_xy[idx_wrist]
            pt_hip = keypoints_xy[idx_hip]
            pt_knee = keypoints_xy[idx_knee]
            pt_ankle = keypoints_xy[idx_ankle]
            
            # つま先（YOLOv8で検出できない場合は足首前方にダミー生成）
            pt_toe = keypoints_xy[idx_toe] if len(keypoints_xy) > 17 and keypoints_xy[idx_toe][0] > 0 else pt_ankle + np.array([20, 0])

            # 足首の移動平均（ブレ抑制）
            if pt_ankle[1] > 0:
                x_history.append(pt_ankle[0])
                y_history.append(pt_ankle[1])
                smooth_ankle_x = sum(x_history) / len(x_history)
                smooth_ankle_y = sum(y_history) / len(y_history)
                
                # 鳥瞰図用リアル座標への変換
                img_coord = np.array([[[smooth_ankle_x, smooth_ankle_y]]], dtype=np.float32)
                real_coord = cv2.perspectiveTransform(img_coord, H)
                real_x = real_coord[0][0][0] + OFFSET_X
                real_y = real_coord[0][0][1] + OFFSET_Y

                # --------------------------------------------------
                # カテゴリ I. 角度・アライメント指標（10項目）の計算
                # --------------------------------------------------
                # 1. 股関節角度
                m1_hip_angle = calculate_angle(pt_shoulder, pt_hip, pt_knee)
                # 2. 膝関節角度
                m2_knee_angle = calculate_angle(pt_hip, pt_knee, pt_ankle)
                # 3. 足関節角度
                m3_ankle_angle = calculate_angle(pt_knee, pt_ankle, pt_toe)
                # 4. 体幹前傾・後傾角 (垂直線 [0, -1] に対する肩-腰ベクトルの傾き)
                m4_trunk_lean = calculate_line_angle(pt_hip, pt_shoulder, [0, -1])
                # 5. 頭部前傾角 (首-耳のラインの傾き)
                pt_neck = (keypoints_xy[5] + keypoints_xy[6]) / 2.0
                m5_head_tilt = calculate_line_angle(pt_neck, pt_ear, [0, -1])
                # 6. 肩関節屈曲角度 (体幹軸に対する上腕の角度)
                m6_shoulder_flex = calculate_angle(pt_hip, pt_shoulder, pt_elbow)
                # 7. 肘関節角度
                m7_elbow_angle = calculate_angle(pt_shoulder, pt_elbow, pt_wrist)
                # 8. 骨盤傾斜角 (水平線 [1, 0] に対する腰関節回りの仮想的な傾き)
                m8_pelvic_tilt = calculate_line_angle(pt_hip, pt_shoulder, [1, 0]) - 90.0
                # 9. 大腿部の対地角度 (水平線 [1, 0] に対する腰-膝の角度)
                m9_thigh_ground_angle = abs(calculate_line_angle(pt_hip, pt_knee, [1, 0]))
                # 10. 胸椎・腰椎の屈曲度 (首-肩-腰の直線からのズレを擬似計算)
                m10_spine_curvature = calculate_angle(pt_neck, pt_shoulder, pt_hip)

                # 簡易重心計算 (頭・肩・腰・膝・足首の平均値から簡易推定)
                pt_com = (pt_nose + pt_shoulder + pt_hip + pt_knee + pt_ankle) / 5.0

                # データをフレームバッファへ格納
                frame_data.update({
                    "has_pose": True,
                    "side_used": side_used,
                    "real_pos": (real_x, real_y),
                    "pt_ankle": (smooth_ankle_x, smooth_ankle_y),
                    "pt_com": pt_com,
                    "pt_shoulder": pt_shoulder,
                    "pt_hip": pt_hip,
                    "pt_ear": pt_ear,
                    "pt_wrist": pt_wrist,
                    "pt_knee": pt_knee,
                    "angles": [m1_hip_angle, m2_knee_angle, m3_ankle_angle, m4_trunk_lean, m5_head_tilt, 
                               m6_shoulder_flex, m7_elbow_angle, m8_pelvic_tilt, m9_thigh_ground_angle, m10_spine_curvature],
                    "raw_frame_with_skeleton": frame.copy()
                })

        frame_buffer.append(frame_data)

        if video_writer is None:
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

        video_writer.write(frame)
        cv2.imshow(win_main, frame)
        current_frame_idx += 1
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()

    # ------------------------------------------------------
    # 後処理フェーズ: 時系列バッファから 20項目 を完全抽出・判定
    # ------------------------------------------------------
    print("\n--- [手順4] 着地イベントの検知と時系列指標の二次計算 ---")
    
    # 骨格が取れている有効なフレームインデックスの抽出
    valid_indices = [i for i, f in enumerate(frame_buffer) if f["has_pose"]]

    for idx in range(2, len(valid_indices) - 2):
        f_idx = valid_indices[idx]
        
        # 3フレーム前後の足首Y座標（ピクセル）を取得し最下点（ボトムアウト）を検知
        y_curr = frame_buffer[f_idx]["pt_ankle"][1]
        y_prev = frame_buffer[valid_indices[idx-1]]["pt_ankle"][1]
        y_pprev = frame_buffer[valid_indices[idx-2]]["pt_ankle"][1]
        y_next = frame_buffer[valid_indices[idx+1]]["pt_ankle"][1]
        y_nnext = frame_buffer[valid_indices[idx+2]]["pt_ankle"][1]

        # 映像座標系では下方向がプラスなので、値が最大＝最下点となる
        if y_prev < y_curr and y_pprev < y_curr and y_curr > y_next and y_curr > y_nnext:
            
            # --- 着地シークエンス（接地から離陸まで）の範囲探索 ---
            # 接地した瞬間のインデックスを探す（足裏が静止ベッド表面を越えた、またはY軸下行が止まった位置）
            touchdown_idx = f_idx
            for k in range(f_idx, 0, -1):
                if frame_buffer[k]["has_pose"] and frame_buffer[k]["pt_ankle"][1] <= bed_rest_y:
                    touchdown_idx = k
                    break
            
            # 離陸した瞬間（テイクオフ）のインデックスを探す
            takeoff_idx = f_idx
            for k in range(f_idx, len(frame_buffer)):
                if frame_buffer[k]["has_pose"] and frame_buffer[k]["pt_ankle"][1] <= bed_rest_y:
                    takeoff_idx = k
                    break
            
            # 最高到達点（直前のジャンプ頂点：Y座標が最小になるフレーム）
            apex_idx = touchdown_idx
            for k in range(max(0, touchdown_idx - int(fps * 2)), touchdown_idx):
                if frame_buffer[k]["has_pose"] and frame_buffer[k]["pt_com"][1] < frame_buffer[apex_idx]["pt_com"][1]:
                    apex_idx = k

            # 有効なシークエンス情報が取れている場合、20項目をビルド
            if frame_buffer[touchdown_idx]["has_pose"] and frame_buffer[takeoff_idx]["has_pose"]:
                
                # 最下点フレームのデータ
                bot_data = frame_buffer[f_idx]
                td_data = frame_buffer[touchdown_idx]
                to_data = frame_buffer[takeoff_idx]
                ap_data = frame_buffer[apex_idx]

                # --------------------------------------------------
                # カテゴリ II. 位置・変位（移動量）指標（5項目）の計算
                # --------------------------------------------------
                # 11. 重心(CoM)の前後変位 (接地からボトムアウトまでのX軸差分)
                m11_com_sway = (bot_data["pt_com"][0] - td_data["pt_com"][0]) * PIXEL_TO_METER
                # 12. 頭部の前方突出量 (肩のX座標基準に対する耳の突き出し)
                m12_head_forward = (bot_data["pt_ear"][0] - bot_data["pt_shoulder"][0]) * PIXEL_TO_METER
                # 13. 臀部の後方突き出し量 (足首X座標に対する腰X座標の遅れ)
                m13_hip_backward = (bot_data["pt_ankle"][0] - bot_data["pt_hip"][0]) * PIXEL_TO_METER
                # 14. ベッドの最大沈み込み深さ (静止ベッド面と最下点足首の差分)
                m14_bed_depression = (bot_data["pt_ankle"][1] - bed_rest_y) * PIXEL_TO_METER
                # 15. 最高到達点からの垂直落下距離 (ジャンプ頂点CoMと接地時CoMの垂直メートル距離)
                m15_drop_distance = (td_data["pt_com"][1] - ap_data["pt_com"][1]) * PIXEL_TO_METER

                # --------------------------------------------------
                # カテゴリ III. 速度・加速度・時間指標（5項目）の計算
                # --------------------------------------------------
                # 16. 進入・離脱速度ベクトル (接地直前2フレームと離陸直後2フレームから計算)
                td_prev = frame_buffer[max(0, touchdown_idx - 1)]
                v_entry_x = (td_data["pt_com"][0] - td_prev["pt_com"][0]) * PIXEL_TO_METER / frame_duration if td_prev["has_pose"] else 0
                v_entry_y = (td_data["pt_com"][1] - td_prev["pt_com"][1]) * PIXEL_TO_METER / frame_duration if td_prev["has_pose"] else 0
                m16_entry_vel = np.sqrt(v_entry_x**2 + v_entry_y**2)
                m16_entry_angle = np.degrees(np.arctan2(v_entry_y, v_entry_x))

                to_next = frame_buffer[min(len(frame_buffer)-1, takeoff_idx + 1)]
                v_exit_x = (to_next["pt_com"][0] - to_data["pt_com"][0]) * PIXEL_TO_METER / frame_duration if to_next["has_pose"] else 0
                v_exit_y = (to_next["pt_com"][1] - to_data["pt_com"][1]) * PIXEL_TO_METER / frame_duration if to_next["has_pose"] else 0
                m16_exit_vel = np.sqrt(v_exit_x**2 + v_exit_y**2)
                m16_exit_angle = np.degrees(np.arctan2(v_exit_y, v_exit_x))

                # 17. 接地時間
                m17_contact_time = (takeoff_idx - touchdown_idx) * frame_duration * 1000  # ミリ秒変換
                # 18. 減速期・加速期の比率
                decel_phase = (f_idx - touchdown_idx) * frame_duration
                accel_phase = (takeoff_idx - f_idx) * frame_duration
                m18_phase_ratio = decel_phase / accel_phase if accel_phase > 0 else 1.0
                # 19. 腕振りの最大角速度 (着地シークエンス中の肩関節角度の変化率の最大値)
                max_ang_vel = 0.0
                for k in range(touchdown_idx, takeoff_idx):
                    if frame_buffer[k]["has_pose"] and frame_buffer[k+1]["has_pose"]:
                        omega = abs(frame_buffer[k+1]["angles"][5] - frame_buffer[k]["angles"][5]) / frame_duration
                        if omega > max_ang_vel: max_ang_vel = omega
                m19_arm_angular_vel = max_ang_vel
                # 20. 腕と下半身の連動時間差 (膝が最も曲がったフレームと、腕が最大に上がったフレームのラグ)
                max_knee_flex_idx = touchdown_idx
                max_arm_flex_idx = touchdown_idx
                for k in range(touchdown_idx, takeoff_idx):
                    if frame_buffer[k]["has_pose"]:
                        if frame_buffer[k]["angles"][1] < frame_buffer[max_knee_flex_idx]["angles"][1]: max_knee_flex_idx = k
                        if frame_buffer[k]["angles"][5] > frame_buffer[max_arm_flex_idx]["angles"][5]: max_arm_flex_idx = k
                m20_coordination_lag = abs(max_knee_flex_idx - max_arm_flex_idx) * frame_duration * 1000  # ミリ秒変換

                # コンソールへの即時フィードバック
                print(f"\n====== 着地解析レコード (20項目拡張版) ======")
                print(f" 着地位置 Real X: {bot_data['real_pos'][0]:.2f}m, Y: {bot_data['real_pos'][1]:.2f}m")
                print(f" [I.角度] 股関節:{bot_data['angles'][0]:.1f}° | 膝関節:{bot_data['angles'][1]:.1f}° | 足首:{bot_data['angles'][2]:.1f}°")
                print(f" [I.角度] 体幹傾き:{bot_data['angles'][3]:.1f}° | 頭部傾き:{bot_data['angles'][4]:.1f}° | 肩屈曲:{bot_data['angles'][5]:.1f}°")
                print(f" [II.変位] 重心ブレ:{m11_com_sway:.3f}m | 頭突出:{m12_head_forward:.3f}m | お尻出し:{m13_hip_backward:.3f}m | 沈み込み:{m14_bed_depression:.2f}m")
                print(f" [III.力学] 接地時間:{m17_contact_time:.1f}ms | 減速/加速比:{m18_phase_ratio:.2f} | 腕最大角速度:{m19_arm_angular_vel:.1f}deg/s")
                print(f"============================================\n")

                # 結果を保存リストへ格納
                all_landing_points.append({
                    "real_pos": bot_data["real_pos"],
                    "color": (0, 255, 0) if (-RED_REAL_WIDTH/2 <= bot_data["real_pos"][0] <= RED_REAL_WIDTH/2) else (0, 165, 255),
                    "side_used": bot_data["side_used"],
                    "delta_x": 0.0, "delta_y": 0.0, "total_dist": 0.0,
                    "metrics": {
                        "m1": bot_data['angles'][0], "m2": bot_data['angles'][1], "m3": bot_data['angles'][2],
                        "m4": bot_data['angles'][3], "m5": bot_data['angles'][4], "m6": bot_data['angles'][5],
                        "m7": bot_data['angles'][6], "m8": bot_data['angles'][7], "m9": bot_data['angles'][8],
                        "m10": bot_data['angles'][9], "m11": m11_com_sway, "m12": m12_head_forward,
                        "m13": m13_hip_backward, "m14": m14_bed_depression, "m15": m15_drop_distance,
                        "m16_en_v": m16_entry_vel, "m16_en_a": m16_entry_angle, "m16_ex_v": m16_exit_vel, "m16_ex_a": m16_exit_angle,
                        "m17": m17_contact_time, "m18": m18_phase_ratio, "m19": m19_arm_angular_vel, "m20": m20_coordination_lag
                    }
                })

                # スナップショット画像を保存
                landing_count = len(all_landing_points)
                img_filename = f"{landing_images_dir}/landing_{landing_count}.png"
                cv2.imwrite(img_filename, bot_data["raw_frame_with_skeleton"])

    cv2.destroyAllWindows()

    # フェーズ5: 鳥瞰サマリー画像の生成
    if len(all_landing_points) > 0:
        SCALE = 150  
        PADDING = 60  
        canvas_width = int(BLUE_REAL_WIDTH * SCALE) + (PADDING * 2)
        canvas_height = int(BLUE_REAL_LENGTH * SCALE) + (PADDING * 2)
        birds_eye_img = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        def meter_to_pixel(mx, my):
            return (int(mx * SCALE + canvas_width / 2), int(my * SCALE + canvas_height / 2))

        cv2.rectangle(birds_eye_img, meter_to_pixel(-BLUE_REAL_WIDTH/2, -BLUE_REAL_LENGTH/2), meter_to_pixel(BLUE_REAL_WIDTH/2, BLUE_REAL_LENGTH/2), (255, 0, 0), 3)
        cv2.rectangle(birds_eye_img, meter_to_pixel(-RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2), meter_to_pixel(RED_REAL_WIDTH/2, RED_REAL_LENGTH/2), (0, 0, 255), 3)

        for index, landing in enumerate(all_landing_points):
            mx, my = landing["real_pos"]
            px, py = meter_to_pixel(mx, my)
            
            if index > 0:
                prev_mx, prev_my = all_landing_points[index - 1]["real_pos"]
                prev_px, prev_py = meter_to_pixel(prev_mx, prev_my)
                diff_x = mx - prev_mx
                diff_y = my - prev_my
                total_dist = np.sqrt(diff_x**2 + diff_y**2)
                landing["delta_x"] = diff_x
                landing["delta_y"] = diff_y
                landing["total_dist"] = total_dist
                
                cv2.arrowedLine(birds_eye_img, (prev_px, prev_py), (px, py), (160, 160, 160), 2, tipLength=0.2)

            cv2.circle(birds_eye_img, (px, py), 12, landing["color"], -1)
            cv2.circle(birds_eye_img, (px, py), 12, (0, 0, 0), 2)  
            cv2.putText(birds_eye_img, str(index + 1), (px + 16, py + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imwrite("birds_eye_landing_summary.png", birds_eye_img)

        # フェーズ6: CSVファイルへのデータレポート出力 (20項目完全版仕様)
        csv_filename = "landing_comprehensive_biometrics_report.csv"
        try:
            with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                # ヘッダーに20の指標すべての名前をフルマッピング
                writer.writerow([
                    "着地番号", "使用半身", "位置_X(m)", "位置_Y(m)", "前回からの移動距離(m)",
                    "1.股関節角度(deg)", "2.膝関節角度(deg)", "3.足関節角度(deg)", "4.体幹前傾後傾角(deg)", "5.頭部前傾角(deg)",
                    "6.肩関節屈曲角(deg)", "7.肘関節角度(deg)", "8.骨盤傾斜角(deg)", "9.大腿対地角(deg)", "10.脊椎屈曲度(deg)",
                    "11.重心前後変位(m)", "12.頭部前方突出量(m)", "13.臀部後方突き出し(m)", "14.ベッド最大沈み込み(m)", "15.最高点垂直落下距離(m)",
                    "16.進入速度(m/s)", "16.進入角度(deg)", "16.離脱速度(m/s)", "16.離脱角度(deg)",
                    "17.総接地時間(ms)", "18.減速加速比率", "19.腕振り最大角速度(deg/s)", "20.腕下半身連動時間差(ms)"
                ])
                
                for index, landing in enumerate(all_landing_points):
                    mx, my = landing["real_pos"]
                    m = landing["metrics"]
                    writer.writerow([
                        index + 1, landing["side_used"], f"{mx:.3f}", f"{my:.3f}",
                        f"{landing['total_dist']:.3f}" if index > 0 else "-",
                        f"{m['m1']:.1f}", f"{m['m2']:.1f}", f"{m['m3']:.1f}", f"{m['m4']:.1f}", f"{m['m5']:.1f}",
                        f"{m['m6']:.1f}", f"{m['m7']:.1f}", f"{m['m8']:.1f}", f"{m['m9']:.1f}", f"{m['m10']:.1f}",
                        f"{m['m1']:.3f}", f"{m['m12']:.3f}", f"{m['m13']:.3f}", f"{m['m14']:.3f}", f"{m['m15']:.3f}",
                        f"{m['m16_en_v']:.2f}", f"{m['m16_en_a']:.1f}", f"{m['m16_ex_v']:.2f}", f"{m['m16_ex_a']:.1f}",
                        f"{m['m17']:.1f}", f"{m['m18']:.2f}", f"{m['m19']:.1f}", f"{m['m20']:.1f}"
                    ])
            print(f"【20項目レポート出力完了】CSVファイル：『{csv_filename}』")
        except Exception as e:
            print(f"CSV保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()