import cv2
import torch
import numpy as np
import csv
from ultralytics import YOLO
from collections import deque
import os

# ==========================================================
# 調整パラメータ設定エリア
# ==========================================================
VIDEO_PATH = "../MediapipePose/video-export/kawae0428-1-1.mp4"  # 解析対象の動画ファイルパスを指定してください

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
# ==========================================================

# 【修正ポイント】入力動画名から拡張子を除いたベース名を取得し、出力動画名を自動生成
video_base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_VIDEO_PATH = f"{video_base_name}_skeleton_output.mp4"

clicked_points = []
all_landing_points = []  
x_history = deque(maxlen=SMOOTHING_FRAMES)
y_history = deque(maxlen=SMOOTHING_FRAMES)

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

# 2つのベクトル間の角度(0〜180度)を計算するヘルパー関数
def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2) # 点2から点1へのベクトル
    v2 = np.array(p3) - np.array(p2) # 点2から点3へのベクトル
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def main():
    global clicked_points, all_landing_points, x_history, y_history
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8s-pose.pt").to(device)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: {VIDEO_PATH} を開けません。")
        return

    # 動画書き込み用（VideoWriter）の初期化用変数
    video_writer = None

    # 着地画像を保存するディレクトリの作成
    landing_images_dir = "landing_snapshots"
    os.makedirs(landing_images_dir, exist_ok=True)

    # ------------------------------------------------------
    # フェーズ1: フレーム選択
    # ------------------------------------------------------
    win_select = "Select Visible Frame"
    cv2.namedWindow(win_select, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_select, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # フルスクリーン化
    
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

    # ------------------------------------------------------
    # フェーズ2: マウスクリック入力（キャリブレーション）
    # ------------------------------------------------------
    win_calib = "Calibration Phase"
    cv2.namedWindow(win_calib, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_calib, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # フルスクリーン化
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

    # フェーズ3: 2D DLT行列の生成
    blue_image_pts = np.array(clicked_points[0:4], dtype=np.float32)
    red_image_pts = np.array(clicked_points[4:8], dtype=np.float32)

    red_real_pts = np.array([
        [-RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2], [ RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2],
        [ RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2], [-RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(red_image_pts, red_real_pts)

    pre_prev_real_y = None
    prev_real_y = None

    print("\n--- [手順3] 解析を開始します ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ------------------------------------------------------
    # フェーズ4: 動画解析ループ
    # ------------------------------------------------------
    win_main = "Universal Trampoline System"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_main, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # メイン表示もフルスクリーン化

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, verbose=False, conf=DETECTION_CONFIDENCE)
        
        # オリジナルの描画ロジック：枠線の追加
        cv2.polylines(frame, [blue_image_pts.astype(np.int32)], True, (255, 0, 0), 3)
        cv2.polylines(frame, [red_image_pts.astype(np.int32)], True, (0, 0, 255), 3)

        if results and results[0].keypoints is not None:
            keypoints_xy = results[0].keypoints.xy.cpu().numpy()
            keypoints_conf = results[0].keypoints.conf.cpu().numpy()
            
            if len(keypoints_xy) > 0:
                person_xy = keypoints_xy[0]
                person_conf = keypoints_conf[0]
                
                # results[0].plot() でスケルトンが上書きされたフレームを取得
                frame = results[0].plot()
                
                # スケルトン描画後のフレームにも枠線を再描画（plotで消えるのを防ぐため）
                cv2.polylines(frame, [blue_image_pts.astype(np.int32)], True, (255, 0, 0), 3)
                cv2.polylines(frame, [red_image_pts.astype(np.int32)], True, (0, 0, 255), 3)

                left_score = person_conf[5] + person_conf[11] + person_conf[13] + person_conf[15]
                right_score = person_conf[6] + person_conf[12] + person_conf[14] + person_conf[16]
                
                if left_score >= right_score:
                    idx_shoulder, idx_wrist, idx_hip, idx_knee, idx_ankle = 5, 9, 11, 13, 15
                    side_used = "LEFT SIDE"
                else:
                    idx_shoulder, idx_wrist, idx_hip, idx_knee, idx_ankle = 6, 10, 12, 14, 16
                    side_used = "RIGHT SIDE"

                pt_shoulder = person_xy[idx_shoulder]
                pt_wrist = person_xy[idx_wrist]
                pt_hip = person_xy[idx_hip]
                pt_knee = person_xy[idx_knee]
                pt_ankle = person_xy[idx_ankle]
                pt_nose = person_xy[0]

                if pt_ankle[1] > 0:
                    x_history.append(pt_ankle[0])
                    y_history.append(pt_ankle[1])
                    img_x = sum(x_history) / len(x_history)
                    img_y = sum(y_history) / len(y_history)

                    img_coord = np.array([[[img_x, img_y]]], dtype=np.float32)
                    real_coord = cv2.perspectiveTransform(img_coord, H)
                    real_x = real_coord[0][0][0] + OFFSET_X
                    real_y = real_coord[0][0][1] + OFFSET_Y

                    is_inside_red = (-RED_REAL_WIDTH/2 <= real_x <= RED_REAL_WIDTH/2 and -RED_REAL_LENGTH/2 <= real_y <= RED_REAL_LENGTH/2)
                    is_inside_blue = (-BLUE_REAL_WIDTH/2 <= real_x <= BLUE_REAL_WIDTH/2 and -BLUE_REAL_LENGTH/2 <= real_y <= BLUE_REAL_LENGTH/2)

                    angle_knee, angle_arm, angle_head, angle_waist = 0.0, 0.0, 0.0, 0.0
                    
                    if pt_hip[1] > 0 and pt_knee[1] > 0 and pt_ankle[1] > 0:
                        angle_knee = calculate_angle(pt_hip, pt_knee, pt_ankle)
                        
                    if pt_hip[1] > 0 and pt_shoulder[1] > 0 and pt_wrist[1] > 0:
                        angle_arm = calculate_angle(pt_hip, pt_shoulder, pt_wrist)
                        
                    if person_xy[5][1] > 0 and person_xy[6][1] > 0 and pt_nose[1] > 0:
                        pt_neck = (person_xy[5] + person_xy[6]) / 2.0
                        pt_vertical_up = np.array([pt_neck[0], pt_neck[1] - 100])
                        angle_head = calculate_angle(pt_nose, pt_neck, pt_vertical_up)
                        
                    if pt_shoulder[1] > 0 and pt_hip[1] > 0 and pt_knee[1] > 0:
                        angle_waist = calculate_angle(pt_shoulder, pt_hip, pt_knee)

                    # 最下点判定
                    if pre_prev_real_y is not None and prev_real_y is not None:
                        if pre_prev_real_y < prev_real_y and prev_real_y > real_y:
                            if is_inside_blue:  
                                print(f"\n====== 着地姿勢解析レコード ======")
                                print(f" 使用半身: {side_used}")
                                print(f" 着地位置: X: {real_x:.2f}m, Y: {real_y:.2f}m")
                                print(f" 1. 膝関節の角度     : {angle_knee:.1f}°")
                                print(f" 2. 腕上げの角度     : {angle_arm:.1f}°")
                                print(f" 3. 頭の傾き角度     : {angle_head:.1f}°")
                                print(f" 4. 腰と上半身の角度 : {angle_waist:.1f}°")
                                print(f"================================\n")
                                
                                all_landing_points.append({
                                    "real_pos": (real_x, real_y),
                                    "color": (0, 255, 0) if is_inside_red else (0, 165, 255),
                                    "side_used": side_used,
                                    "angles": (angle_knee, angle_arm, angle_head, angle_waist),
                                    "delta_x": 0.0,
                                    "delta_y": 0.0,
                                    "total_dist": 0.0
                                })

                                # 着地した瞬間の骨格付きフレーム画像を切り抜いて保存
                                landing_count = len(all_landing_points)
                                img_filename = f"{landing_images_dir}/landing_{landing_count}.png"
                                cv2.imwrite(img_filename, frame)
                                print(f"【画像保存】着地 {landing_count} 回目の瞬間を保存しました: {img_filename}")

                    pre_prev_real_y = prev_real_y
                    prev_real_y = real_y

        # 動画保存用の初期化処理（最初のフレームサイズを元に設定）
        if video_writer is None:
            height, width, _ = frame.shape
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps): 
                fps = 30.0  # 万が一FPSが取得できない場合のフォールバック
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

        # 動画の各フレームを書き込み
        video_writer.write(frame)

        cv2.imshow(win_main, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"【保存完了】スケルトン重ね合わせ動画：『{OUTPUT_VIDEO_PATH}』")
    cv2.destroyAllWindows()

    # フェーズ5: 鳥瞰サマリー画像の生成と移動ベクトルの計算
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
                
                dir_x = "R" if diff_x >= 0 else "L"
                dir_y = "Front" if diff_y >= 0 else "Back"
                
                text_vector = f"[{dir_x}:{abs(diff_x):.2f}m, {dir_y}:{abs(diff_y):.2f}m]"
                text_dist = f"Dist:{total_dist:.2f}m"
                
                cv2.putText(birds_eye_img, text_vector, (px - 55, py - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
                cv2.putText(birds_eye_img, text_dist, (px - 35, py - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.circle(birds_eye_img, (px, py), 12, landing["color"], -1)
            cv2.circle(birds_eye_img, (px, py), 12, (0, 0, 0), 2)  
            cv2.putText(birds_eye_img, str(index + 1), (px + 16, py + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imwrite("birds_eye_landing_summary.png", birds_eye_img)
        print("【保存完了】手動補正適用済みの鳥瞰図：『birds_eye_landing_summary.png』")

        # フェーズ6: CSVファイルへのデータレポート出力
        csv_filename = "landing_analysis_report.csv"
        try:
            with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["着地番号", "使用半身", "位置_X(m)", "位置_Y(m)", "前回からの移動_X(m)", "前回からの移動_Y(m)", "総移動距離(m)", "膝関節角度", "腕上げ角度", "頭部傾き角度", "腰上半身角度"])
                
                for index, landing in enumerate(all_landing_points):
                    mx, my = landing["real_pos"]
                    ak, aa, ah, aw = landing["angles"]
                    writer.writerow([
                        index + 1,
                        landing["side_used"],
                        f"{mx:.3f}",
                        f"{my:.3f}",
                        f"{landing['delta_x']:.3f}" if index > 0 else "-",
                        f"{landing['delta_y']:.3f}" if index > 0 else "-",
                        f"{landing['total_dist']:.3f}" if index > 0 else "-",
                        f"{ak:.1f}",
                        f"{aa:.1f}",
                        f"{ah:.1f}",
                        f"{aw:.1f}"
                    ])
            print(f"【保存完了】数値データレポート：『{csv_filename}』")
        except Exception as e:
            print(f"CSV保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()