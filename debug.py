import cv2
import torch
import numpy as np
import csv
from ultralytics import YOLO
import os

# ==========================================================
# 調整パラメータ設定エリア
# ==========================================================
VIDEO_PATH = "../MediapipePose/video-export/kawae0428-2-1.mp4"  

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

# ★ベッドの変形を検知するための感度しきい値
# 準備ジャンプを無視し、技の強い着地・沈み込みだけに反応させるため、しきい値を少し高め（120000）に調整しています。
BED_DEFORMATION_THRESHOLD = 120000  
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

    # フェーズ1: キャリブレーション
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
    red_real_pts = np.array([
        [-RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2], [ RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2],
        [ RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2], [-RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(red_image_pts, red_real_pts)
    bed_rest_y = np.mean(red_image_pts[:, 1])  

    roi_y1 = int(min(red_image_pts[2][1], red_image_pts[3][1])) - 10  
    roi_y2 = int(max(red_image_pts[2][1], red_image_pts[3][1])) + 40  
    roi_x1 = int(min(red_image_pts[2][0], red_image_pts[3][0]))
    roi_x2 = int(max(red_image_pts[2][0], red_image_pts[3][0]))

    # ==================================================================
    # フェーズ1：全フレーム事前精密スキャン（アルゴリズム全面改訂）
    # ==================================================================
    print("\n--- [Phase 1/2] 演技区間の検出バグを修正した精密スキャンを実行中... ---")
    bed_moving_frames = []      # ベッドの揺れがしきい値を超えたフレームの全リスト
    flip_frames = []            # 空中反転（逆さま）を検知したフレームの全リスト
    
    prev_roi_gray = None
    scan_idx = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 1. ベッドの変形（揺れ強度）の全コマ記録
        roi_current = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_gray = cv2.cvtColor(roi_current, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (21, 21), 0)
        
        if prev_roi_gray is not None:
            roi_delta = cv2.absdiff(prev_roi_gray, roi_gray)
            _, roi_thresh = cv2.threshold(roi_delta, 25, 255, cv2.THRESH_BINARY)
            if np.sum(roi_thresh) > BED_DEFORMATION_THRESHOLD:
                bed_moving_frames.append(scan_idx)
        prev_roi_gray = roi_gray.copy()
        
        # 2. 空中反転（逆さま状態）の全コマ記録
        results = model.predict(frame, verbose=False, conf=DETECTION_CONFIDENCE)
        if results and results[0].keypoints is not None and len(results[0].keypoints.xy.cpu().numpy()) > 0:
            keypoints_xy = results[0].keypoints.xy.cpu().numpy()[0]
            pt_nose = keypoints_xy[0]
            pt_hip_y = keypoints_xy[11][1] if keypoints_xy[11][1] > 0 else keypoints_xy[12][1]
            pt_ankle_y = keypoints_xy[15][1] if keypoints_xy[15][1] > 0 else keypoints_xy[16][1]
            
            if pt_nose[1] > 0 and pt_hip_y > 0 and pt_ankle_y > 0:
                # 完全に空中反転（頭が腰より下、かつ足首が頭より上）しているコマのみを記録
                if pt_nose[1] > pt_hip_y and pt_ankle_y < pt_nose[1]:
                    flip_frames.append(scan_idx)
                        
        scan_idx += 1

    # ==================================================================
    # 🎯【バグ修正の核心】空中宙返りの中心点から前後の「正しい着地」を一意に特定する
    # ==================================================================
    target_analysis_segments = []
    
    if len(flip_frames) > 0 and len(bed_moving_frames) > 0:
        # 空中宙返り期の中央値（中心点）を計算
        flip_center = int(np.median(flip_frames))
        
        # ① 宙返り中心から【過去に遡って】、準備ジャンプを飛び越えた「技はじめの強い沈み込み」を特定
        # 宙返り中心より前にあるベッド運動フレームのうち、最も宙返りに近い連続した塊の始点を探す
        prior_motions = [f for f in bed_moving_frames if f < flip_center]
        analysis_start = 0
        if len(prior_motions) > 0:
            # 宙返り直前のベッド運動の終点を基準にする
            last_prior_motion = prior_motions[-1]
            # そこから連続して遡れる限界（ベッドが動き始めた瞬間）を技の始点とする
            start_candidate = last_prior_motion
            for f in reversed(prior_motions):
                if start_candidate - f <= 3: # 3フレーム以内の連続性を許容
                    start_candidate = f
                else:
                    break
            analysis_start = max(0, start_candidate - 5) # 踏み切り前のタメを5フレーム余分に確保
            
        # ② 宙返り中心から【未来に進んで】、途切れず発生する「最終着地（静止するまで）」を特定
        posterior_motions = [f for f in bed_moving_frames if f > flip_center]
        analysis_end = scan_idx - 1
        if len(posterior_motions) > 0:
            # 宙返り後に最初に発生したベッド接触（着地の瞬間）
            first_posterior_motion = posterior_motions[0]
            # そこからベッドの揺れが完全に収まる（静止する）までの連続した終点を探す
            end_candidate = first_posterior_motion
            for f in posterior_motions:
                if f - end_candidate <= 5: # 衝撃の跳ね返りノイズを考慮して5フレームの連続性を許容
                    end_candidate = f
                else:
                    break
            analysis_end = min(scan_idx - 1, end_candidate + 15) # 着地後に完全に静止するまで15フレーム余分にホールド
            
        if analysis_start < flip_center < analysis_end:
            target_analysis_segments.append([analysis_start, analysis_end])
            print(f"🟢 [判定成功] 正しい技のシークエンスを固定しました: フレーム {analysis_start} 〜 {analysis_end}")
            print(f"   (技はじめの沈み込み開始: {analysis_start} / 宙返り中心空中心: {flip_center} / 技おわりの静止終了: {analysis_end})")
        else:
            # 安全用のフォールバック
            target_analysis_segments.append([max(0, flip_frames[0] - 30), min(scan_idx - 1, flip_frames[-1] + 45)])
            print("⚠️ タイムウィンドウ展開によりフォールバック区間を設定しました。")

    # ==================================================================
    # フェーズ2：本解析動画再生ループ（ピンポイント姿勢推定）
    # ==================================================================
    print("\n--- [Phase 2/2] 修正された正しい演技区間のみの本解析（姿勢推定）を実行中... ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    trial_count = 0            
    seq_frame_count = 0        
    touchdown_com_y = 0.0      
    touchdown_ankle_x_meter = None  
    prev_com_pos = None  
    current_frame_idx = 0

    win_main = "Universal Trampoline System v12 (Fixed Sequence)"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_main, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        display_frame = frame.copy()
        cv2.polylines(display_frame, [blue_image_pts.astype(np.int32)], True, (255, 0, 0), 3)
        cv2.polylines(display_frame, [red_image_pts.astype(np.int32)], True, (0, 0, 255), 3)

        in_target_segment = False
        if len(target_analysis_segments) > 0:
            if target_analysis_segments[0][0] <= current_frame_idx <= target_analysis_segments[0][1]:
                in_target_segment = True

        if in_target_segment:
            if seq_frame_count == 0:
                trial_count = 1
                seq_frame_count = 0

            results = model.track(display_frame, persist=True, verbose=False, conf=DETECTION_CONFIDENCE)
            
            if results and results[0].keypoints is not None and len(results[0].keypoints.xy.cpu().numpy()) > 0:
                keypoints_xy = results[0].keypoints.xy.cpu().numpy()[0]
                keypoints_conf = results[0].keypoints.conf.cpu().numpy()[0]
                display_frame = results[0].plot()
                
                cv2.polylines(display_frame, [blue_image_pts.astype(np.int32)], True, (255, 0, 0), 3)
                cv2.polylines(display_frame, [red_image_pts.astype(np.int32)], True, (0, 0, 255), 3)

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

                    if touchdown_ankle_x_meter is None:
                        touchdown_com_y = pt_com[1]
                        touchdown_ankle_x_meter = real_x  

                    # 1〜10項目（角度計算）
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

                    # 11項目：技はじめ着地からの「絶対移動距離(m)」
                    m11_total_sway = real_x - touchdown_ankle_x_meter

                    m12_head_fwd = (pt_ear[0] - pt_shoulder[0]) * PIXEL_TO_METER
                    m13_hip_bwd = (pt_ankle[0] - pt_hip[0]) * PIXEL_TO_METER
                    m14_dep = (pt_ankle[1] - bed_rest_y) * PIXEL_TO_METER
                    m15_drop = abs(pt_com[1] - touchdown_com_y) * PIXEL_TO_METER

                    if prev_com_pos is not None:
                        vx = (pt_com[0] - prev_com_pos[0]) * PIXEL_TO_METER / frame_duration
                        vy = (pt_com[1] - prev_com_pos[1]) * PIXEL_TO_METER / frame_duration
                        m16_vel = np.sqrt(vx**2 + vy**2)
                        m16_ang = np.degrees(np.arctan2(vy, vx))
                    else:
                        m16_vel, m16_ang = 0.0, 0.0
                    prev_com_pos = pt_com

                    # 連番画像の物理保存
                    img_filename = f"fixed_seq_t{trial_count}_f{seq_frame_count:03d}.png"
                    img_full_path = os.path.join(sequence_images_dir, img_filename)
                    cv2.imwrite(img_full_path, display_frame)

                    historical_sequence_data.append([
                        trial_count, seq_frame_count, f"{elapsed_time_ms:.1f}", side_used, f"{real_x:.3f}", f"{real_y:.3f}",
                        f"{m1_hip:.1f}", f"{m2_knee:.1f}", f"{m3_ankle:.1f}", f"{m4_trunk:.1f}", f"{m5_head:.1f}",
                        f"{m6_shoulder:.1f}", f"{m7_elbow:.1f}", f"{m8_pelvic:.1f}", f"{m9_thigh:.1f}", f"{m10_spine:.1f}",
                        f"{m11_total_sway:.3f}", f"{m12_head_fwd:.3f}", f"{m13_hip_bwd:.3f}", f"{m14_dep:.3f}", f"{m15_drop:.3f}",
                        f"{m16_vel:.2f}", f"{m16_ang:.1f}",
                        img_filename
                    ])

            cv2.putText(display_frame, f"FIXED SEQUENCE ANALYZING... (Frame:{seq_frame_count})", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(display_frame, "AIRBORNE / SKIP ACTIVE (YOLO OFF)", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            prev_com_pos = None

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
        csv_filename = "landing_perfect_sequence_report.csv"
        try:
            with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "試技番号", "区間内フレーム連番", "経過時間(ms)", "使用半身", "実空間位置_X(m)", "実空間位置_Y(m)",
                    "1.股関節角度(deg)", "2.膝関節角度(deg)", "3.足関節角度(deg)", "4.体幹前傾後傾角(deg)", "5.頭部前傾角(deg)",
                    "6.肩関節屈曲角(deg)", "7.肘関節角度(deg)", "8.骨盤傾斜角(deg)", "9.大腿対地角(deg)", "10.脊椎屈曲度(deg)",
                    "11.絶対移動距離_X(m)", "12.頭部前方突出量(m)", "13.臀部後方突き出し(m)", "14.ベッド沈み込み深さ(m)", "15.垂直移動距離(m)",
                    "16.現在の移動速度(m/s)", "16.現在の移動角度(deg)",
                    "対応姿勢画像ファイル名(LINK_IMAGE_NAME)"
                ])
                writer.writerows(historical_sequence_data)
                
            print(f"\n========================================================")
            print(f"【区間修正版の解析が完了しました】")
            print(f"  ➔ CSVレポート: 『{csv_filename}』")
            print(f"  ➔ 画像フォルダ: 『{sequence_images_dir}/』")
            print(f"  準備ジャンプを綺麗にスキップし、技はじめの強い沈み込みから")
            print(f"  技おわりの最終着地で完全に静止するまでを正確にホールドしました。")
            print(f"========================================================")
        except Exception as e:
            print(f"CSV保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()