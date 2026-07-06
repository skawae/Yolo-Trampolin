import cv2
import torch
import numpy as np
import csv
from ultralytics import YOLO
import os

# ==========================================================
# 調整パラメータ設定エリア
# ==========================================================
VIDEO_PATH = "../MediapipePose/video-export/kawae0428-7-2.mp4"  

# 使用する台の実際のサイズ（メートル単位）
BLUE_REAL_WIDTH = 5.05   
BLUE_REAL_LENGTH = 2.91  
RED_REAL_WIDTH = 2.15   
RED_REAL_LENGTH = 1.08  

# 鳥瞰図の位置ズレ修正パラメータ
OFFSET_X = 0.0  
OFFSET_Y = -0.5  

DETECTION_CONFIDENCE = 0.4     
PIXEL_TO_METER = 0.005  

# ベッドの変形（赤線の移動）を検知するための感度しきい値
BED_DEFORMATION_THRESHOLD = 80000  
# ==========================================================

video_base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_VIDEO_PATH = f"{video_base_name}_skeleton_output.mp4"

clicked_points = []
historical_sequence_data = []  

GUIDE_TEXTS = [
    "1/8: Click BLUE [Top-Left]", "2/8: Click BLUE [Top-Right]",
    "3/8: Click BLUE [Bottom-Right]", "4/8: Click BLUE [Bottom-Left]",
    "5/8: Click RED  [Top-Left]", "6/8: Click RED  [Top-Right]",
    "7/8: Click RED  [Bottom-Right]", "8/8: Click RED  [Bottom-Left]"
]

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 8:
            clicked_points.append([x, y])
            print(f"座標登録 [{len(clicked_points)}/8]: X={x}, Y={y}")

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
    if np.cross(reference_vector, v) > 0:
        return angle
    return -angle

def main():
    global clicked_points, historical_sequence_data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8s-pose.pt").to(device)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: {VIDEO_PATH} を開けません。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    frame_duration = 1.0 / fps  

    video_writer = None
    sequence_images_dir = "landing_sequence_images"
    os.makedirs(sequence_images_dir, exist_ok=True)

    # フェーズ1: フレーム選択とキャリブレーション
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

    # ★ここが抜けている、または位置がズレているためエラーが起きた.
    red_real_pts = np.array([
        [-RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2], [ RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2],
        [ RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2], [-RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2]
    ], dtype=np.float32)

    # 2D DLT行列の生成
    H, _ = cv2.findHomography(red_image_pts, red_real_pts)

    bed_rest_y = np.mean(red_image_pts[:, 1])  

    roi_y1 = int(min(red_image_pts[2][1], red_image_pts[3][1])) - 10  
    roi_y2 = int(max(red_image_pts[2][1], red_image_pts[3][1])) + 40  
    roi_x1 = int(min(red_image_pts[2][0], red_image_pts[3][0]))
    roi_x2 = int(max(red_image_pts[2][0], red_image_pts[3][0]))

    # ==================================================================
    # ★修正：フェーズ1：動画の事前高速スキャン（着地区間 ＆ 宙返りフレームの全抽出）
    # ==================================================================
    print("\n--- [Phase 1/2] 動画を高速走査して『宙返りを挟む着地間』を特定中... ---")
    raw_landings = []           # すべての着地区間 [[開始, 終了], [開始, 終了], ...]
    flip_frames = set()         # 宙返り（逆さま）が発生したフレーム番号の集合
    
    prev_roi_gray = None
    is_bed_moving_raw = False
    current_start = None
    scan_idx = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 1. ベッドの変形（着地区間）の抽出
        roi_current = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_gray = cv2.cvtColor(roi_current, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (21, 21), 0)
        
        if prev_roi_gray is not None:
            roi_delta = cv2.absdiff(prev_roi_gray, roi_gray)
            _, roi_thresh = cv2.threshold(roi_delta, 25, 255, cv2.THRESH_BINARY)
            if np.sum(roi_thresh) > BED_DEFORMATION_THRESHOLD:
                if not is_bed_moving_raw:
                    is_bed_moving_raw = True
                    current_start = scan_idx
            else:
                if is_bed_moving_raw:
                    is_bed_moving_raw = False
                    raw_landings.append([current_start, scan_idx])
        prev_roi_gray = roi_gray.copy()
        
        # 2. 宙返り（逆さま）の抽出（3フレームに1回間引き）
        if scan_idx % 3 == 0:
            results = model.predict(frame, verbose=False, conf=DETECTION_CONFIDENCE)
            if results and results[0].keypoints is not None and len(results[0].keypoints.xy.cpu().numpy()) > 0:
                keypoints_xy = results[0].keypoints.xy.cpu().numpy()[0]
                pt_nose = keypoints_xy[0]
                pt_ankle_y = keypoints_xy[15][1] if keypoints_xy[15][1] > 0 else keypoints_xy[16][1]
                if pt_nose[1] > 0 and pt_ankle_y > 0 and pt_nose[1] > pt_ankle_y:
                    flip_frames.add(scan_idx)
                    
        scan_idx += 1

    # 3. 【最重要ロジック】着地と着地の間に宙返りがある区間を特定し、解析ターゲット区間を生成
    target_analysis_segments = []
    print(f"検出された全着地区間数: {len(raw_landings)}")
    
    for i in range(len(raw_landings) - 1):
        landing_A_end = raw_landings[i][0]       # 着地Aの開始（技の始まり着地のタッチダウン）
        landing_B_end = raw_landings[i+1][1]     # 着地Bの終了（技の終わり着地のテイクオフ）
        
        # 着地Aの開始から着地Bの終了までの間に、宙返りフレームが1コマでも存在するかチェック
        has_flip_between = any(landing_A_end <= f <= landing_B_end for f in flip_frames)
        
        if has_flip_between:
            # 条件を満たした場合、着地Aの開始〜着地Bの終了までを「一連の姿勢推定ターゲット」として登録
            target_analysis_segments.append([landing_A_end, landing_B_end])
            print(f"  ➔ ターゲット区間特定: フレーム {landing_A_end} 〜 {landing_B_end} (着地間の空中内に宙返りを確認)")

    # ==================================================================
    # ★フェーズ2：本解析動画再生ループ（特定区間のみYOLOを完全駆動）
    # ==================================================================
    print("\n--- [Phase 2/2] ターゲット着地間のみを狙い撃ちして本解析中... ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 動画を先頭に巻き戻し
    
    trial_count = 0            
    seq_frame_count = 0        
    touchdown_com_y = 0.0      
    touchdown_ankle_x = 0.0    
    prev_com_pos = None  
    current_frame_idx = 0

    win_main = "Universal Trampoline System v9 (Between Landings)"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_main, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        display_frame = frame.copy()
        
        # 基本枠線とROIの描画
        cv2.polylines(display_frame, [blue_image_pts.astype(np.int32)], True, (255, 0, 0), 3)
        cv2.polylines(display_frame, [red_image_pts.astype(np.int32)], True, (0, 0, 255), 3)
        cv2.rectangle(display_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

        # 💡 現在のフレームが、特定された「宙返りを挟む着地間セグメント」の中に入っているか判定
        in_target_segment = False
        active_segment_num = -1
        for idx, seg in enumerate(target_analysis_segments):
            if seg[0] <= current_frame_idx <= seg[1]:
                in_target_segment = True
                active_segment_num = idx + 1
                break

        # --- 条件に応じた姿勢推定とスキップの切り替え ---
        if in_target_segment:
            # 技始まりの着地〜空中（宙返り）〜技終わりの着地までの間、YOLOを毎フレーム完全駆動
            if seq_frame_count == 0 or current_frame_idx == target_analysis_segments[active_segment_num-1][0]:
                trial_count = active_segment_num
                seq_frame_count = 0
                print(f"【解析セグメント進入】試技 #{trial_count} (フレーム:{current_frame_idx}) の毎フレーム追跡を開始。")

            results = model.track(display_frame, persist=True, verbose=False, conf=DETECTION_CONFIDENCE)
            
            if results and results[0].keypoints is not None and len(results[0].keypoints.xy.cpu().numpy()) > 0:
                keypoints_xy = results[0].keypoints.xy.cpu().numpy()[0]
                keypoints_conf = results[0].keypoints.conf.cpu().numpy()[0]
                display_frame = results[0].plot()
                
                # スケルトン描画後の枠線再描画
                cv2.polylines(display_frame, [blue_image_pts.astype(np.int32)], True, (255, 0, 0), 3)
                cv2.polylines(display_frame, [red_image_pts.astype(np.int32)], True, (0, 0, 255), 3)

                # 左右半身判定
                left_score = sum([keypoints_conf[i] for i in [5, 9, 11, 13, 15]])
                right_score = sum([keypoints_conf[i] for i in [6, 10, 12, 14, 16]])
                if left_score >= right_score:
                    idx_ear, idx_shoulder, idx_elbow, idx_wrist, idx_hip, idx_knee, idx_ankle, idx_toe = 3, 5, 7, 9, 11, 13, 15, 17
                    side_used = "LEFT SIDE"
                else:
                    idx_ear, idx_shoulder, idx_elbow, idx_wrist, idx_hip, idx_knee, idx_ankle, idx_toe = 4, 6, 8, 10, 12, 14, 16, 17
                    side_used = "RIGHT SIDE"

                pt_nose = keypoints_xy[0]
                pt_ear = keypoints_xy[idx_ear]
                pt_shoulder = keypoints_xy[idx_shoulder]
                pt_elbow = keypoints_xy[idx_elbow]
                pt_wrist = keypoints_xy[idx_wrist]
                pt_hip = keypoints_xy[idx_hip]
                pt_knee = keypoints_xy[idx_knee]
                pt_ankle = keypoints_xy[idx_ankle]
                pt_toe = keypoints_xy[idx_toe] if len(keypoints_xy) > 17 and keypoints_xy[idx_toe][0] > 0 else pt_ankle + np.array([20, 0])

                if pt_ankle[1] > 0:
                    seq_frame_count += 1
                    elapsed_time_ms = seq_frame_count * frame_duration * 1000

                    img_coord = np.array([[[pt_ankle[0], pt_ankle[1]]]], dtype=np.float32)
                    real_coord = cv2.perspectiveTransform(img_coord, H)
                    real_x = real_coord[0][0][0] + OFFSET_X
                    real_y = real_coord[0][0][1] + OFFSET_Y

                    pt_com = (pt_nose + pt_shoulder + pt_hip + pt_knee + pt_ankle) / 5.0

                    if seq_frame_count == 1:
                        touchdown_com_y = pt_com[1]
                        touchdown_ankle_x = pt_com[0]

                    # 角度・アライメント計算（1〜10項目）
                    m1_hip = calculate_angle(pt_shoulder, pt_hip, pt_knee)
                    m2_knee = calculate_angle(pt_hip, pt_knee, pt_ankle)
                    m3_ankle = calculate_angle(pt_knee, pt_ankle, pt_toe)
                    m4_trunk = calculate_line_angle(pt_hip, pt_shoulder, [0, -1])
                    pt_neck = (keypoints_xy[5] + keypoints_xy[6]) / 2.0
                    m5_head = calculate_line_angle(pt_neck, pt_ear, [0, -1])
                    m6_shoulder = calculate_angle(pt_hip, pt_shoulder, pt_elbow)
                    m7_elbow = calculate_angle(pt_shoulder, pt_elbow, pt_wrist)
                    m8_pelvic = calculate_line_angle(pt_hip, pt_shoulder, [1, 0]) - 90.0
                    m9_thigh = abs(calculate_line_angle(pt_hip, pt_knee, [1, 0]))
                    m10_spine = calculate_angle(pt_neck, pt_shoulder, pt_hip)

                    # 位置・変位計算（5項目）
                    m11_sway = (pt_com[0] - touchdown_ankle_x) * PIXEL_TO_METER
                    m12_head_fwd = (pt_ear[0] - pt_shoulder[0]) * PIXEL_TO_METER
                    m13_hip_bwd = (pt_ankle[0] - pt_hip[0]) * PIXEL_TO_METER
                    m14_dep = (pt_ankle[1] - bed_rest_y) * PIXEL_TO_METER
                    m15_drop = abs(pt_com[1] - touchdown_com_y) * PIXEL_TO_METER

                    # 速度ベクトル計算
                    if prev_com_pos is not None:
                        vx = (pt_com[0] - prev_com_pos[0]) * PIXEL_TO_METER / frame_duration
                        vy = (pt_com[1] - prev_com_pos[1]) * PIXEL_TO_METER / frame_duration
                        m16_vel = np.sqrt(vx**2 + vy**2)
                        m16_ang = np.degrees(np.arctan2(vy, vx))
                    else:
                        m16_vel, m16_ang = 0.0, 0.0
                    prev_com_pos = pt_com

                    # 【画像データの工夫】画像の物理保存
                    img_filename = f"between_landings_t{trial_count}_f{seq_frame_count:03d}.png"
                    img_full_path = os.path.join(sequence_images_dir, img_filename)
                    cv2.imwrite(img_full_path, display_frame)

                    # レコードへ追加
                    historical_sequence_data.append([
                        trial_count, seq_frame_count, f"{elapsed_time_ms:.1f}", side_used, f"{real_x:.3f}", f"{real_y:.3f}",
                        f"{m1_hip:.1f}", f"{m2_knee:.1f}", f"{m3_ankle:.1f}", f"{m4_trunk:.1f}", f"{m5_head:.1f}",
                        f"{m6_shoulder:.1f}", f"{m7_elbow:.1f}", f"{m8_pelvic:.1f}", f"{m9_thigh:.1f}", f"{m10_spine:.1f}",
                        f"{m11_sway:.3f}", f"{m12_head_fwd:.3f}", f"{m13_hip_bwd:.3f}", f"{m14_dep:.3f}", f"{m15_drop:.3f}",
                        f"{m16_vel:.2f}", f"{m16_ang:.1f}",
                        img_filename
                    ])

            cv2.putText(display_frame, f"FLIP-SEQUENCE ANALYZING... TRIAL #{trial_count} (Frame:{seq_frame_count})", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            # 登録された区間外（無関係な空中や、技を伴わない跳躍）はYOLOを完全スキップ
            cv2.putText(display_frame, "OUT OF TARGET SEGMENT / SKIP ACTIVE (YOLO OFF)", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            prev_com_pos = None
            seq_frame_count = 0 # リセット

        if video_writer is None:
            height, width, _ = display_frame.shape
            video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        video_writer.write(display_frame)

        cv2.imshow(win_main, display_frame)
        current_frame_idx += 1
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if video_writer is not None: video_writer.release()
    cv2.destroyAllWindows()

    # レポート出力
    if len(historical_sequence_data) > 0:
        csv_filename = "landing_between_flips_linked_report.csv"
        try:
            with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "技試技番号", "区間内フレーム連番", "経過時間(ms)", "使用半身", "足首位置_X(m)", "足首位置_Y(m)",
                    "1.股関節角度(deg)", "2.膝関節角度(deg)", "3.足関節角度(deg)", "4.体幹前傾後傾角(deg)", "5.頭部前傾角(deg)",
                    "6.肩関節屈曲角(deg)", "7.肘関節角度(deg)", "8.骨盤傾斜角(deg)", "9.大腿対地角(deg)", "10.脊椎屈曲度(deg)",
                    "11.重心前後変位(m)", "12.頭部前方突出量(m)", "13.臀部後方突き出し(m)", "14.ベッド沈み込み深さ(m)", "15.垂直移動距離(m)",
                    "16.現在の移動速度(m/s)", "16.現在の移動角度(deg)",
                    "対応姿勢画像ファイル名(LINK_IMAGE_NAME)"
                ])
                writer.writerows(historical_sequence_data)
                
            print(f"\n========================================================")
            print(f"【シークエンス自動処理完了】")
            print(f"  1. 着地間（技開始〜空中宙返り〜技終わり）数値レポートCSV: 『{csv_filename}』")
            print(f"  2. 連動スケルトン画像フォルダ: 『{sequence_images_dir}/』")
            print(f"  ※ 宙返りが行われた『着地から着地の間』だけが完璧にデータ化されました。")
            print(f"========================================================")
        except Exception as e:
            print(f"CSV保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()