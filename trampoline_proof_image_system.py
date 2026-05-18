import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==========================================================
# 調整パラメータ設定エリア
# ==========================================================
VIDEO_PATH = "../MediapipePose/video-export/kawae0428-1-3.mp4"  # 解析対象の動画ファイルパス

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

def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def main():
    global clicked_points, all_landing_points, x_history, y_history
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8s-pose.pt").to(device)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: {VIDEO_PATH} を開けません。")
        return

    # フェーズ1: フレーム選択
    cv2.namedWindow("Select Visible Frame")
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        display_select = frame.copy()
        cv2.putText(display_select, "PRESS 'SPACE' TO SELECT THIS FRAME", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Select Visible Frame", display_select)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:
            init_frame = frame.copy()
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow("Select Visible Frame")

    # フェーズ2: マウスクリック入力
    cv2.namedWindow("Calibration Phase")
    cv2.setMouseCallback("Calibration Phase", mouse_callback)
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

        cv2.imshow("Calibration Phase", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow("Calibration Phase")

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

    print("\n--- [手順3] 解析を開始します。着地時に写真を自動保存します ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # フェーズ4: 動画解析ループ
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, verbose=False, conf=DETECTION_CONFIDENCE)
        cv2.polylines(frame, [blue_image_pts.astype(np.int32)], True, (255, 0, 0), 3)
        cv2.polylines(frame, [red_image_pts.astype(np.int32)], True, (0, 0, 255), 3)

        if results and results[0].keypoints is not None:
            keypoints_xy = results[0].keypoints.xy.cpu().numpy()
            keypoints_conf = results[0].keypoints.conf.cpu().numpy()
            
            if len(keypoints_xy) > 0:
                person_xy = keypoints_xy[0]
                person_conf = keypoints_conf[0]
                frame = results[0].plot()

                # 自動サイド選択（映りの良い方を採用）
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

                    # リアルタイムの角度計算
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

                    # 最下点判定（着地ボトムアウト）
                    if pre_prev_real_y is not None and prev_real_y is not None:
                        if pre_prev_real_y < prev_real_y and prev_real_y > real_y:
                            if is_inside_blue:  
                                # 鳥瞰図プロット用にデータ保存
                                area_name = "CENTER (RED)" if is_inside_red else "SAFETY (BLUE)"
                                text_color = (0, 255, 0) if is_inside_red else (0, 165, 255)
                                all_landing_points.append({"real_pos": (real_x, real_y), "color": text_color})
                                
                                # 現在の着地回数を取得
                                landing_count = len(all_landing_points)
                                print(f"[着地検出] 回数: {landing_count} | 証拠画像を保存します...")

                                # 【新機能】着地瞬間の画像切り出しと加工
                                moment_img = frame.copy() # 骨格ワイヤーフレーム入りの現在のコマをコピー
                                
                                # 1. 左上：着地回数を黄色で大きく描画
                                cv2.putText(moment_img, f"Landing No.{landing_count}", (30, 60), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4, cv2.LINE_AA)
                                
                                # 2. 右側：計算されたポーズ推定の数値を焼き付ける
                                img_h, img_w, _ = moment_img.shape
                                text_x = img_w - 480 if img_w > 500 else 30 # 画面サイズに応じて配置を自動調整
                                text_y_start = 60
                                
                                info_texts = [
                                    f"--- Pose Angle Data ({side_used}) ---",
                                    f"1. Knee Angle : {angle_knee:.1f} deg",
                                    f"2. Arm Raise  : {angle_arm:.1f} deg",
                                    f"3. Head Tilt  : {angle_head:.1f} deg",
                                    f"4. Waist Angle : {angle_waist:.1f} deg",
                                    f"Position     : X={real_x:.2f}m, Y={real_y:.2f}m",
                                    f"Area         : {area_name}"
                                ]
                                
                                # 各行を黒いフチ付きの白文字で描画（動画背景が見づらくても読めるようにします）
                                for i, t_str in enumerate(info_texts):
                                    target_y = text_y_start + (i * 40)
                                    # 黒い影（フチ）
                                    cv2.putText(moment_img, t_str, (text_x+2, target_y+2), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                                    # 白いメイン文字
                                    cv2.putText(moment_img, t_str, (text_x, target_y), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                                # 3. 画像ファイルとして保存
                                file_name = f"landing_moment_{landing_count}.png"
                                cv2.imwrite(file_name, moment_img)
                                print(f"➔ 画像保存完了: {file_name}")

                    pre_prev_real_y = prev_real_y
                    prev_real_y = real_y

        cv2.imshow("Universal Trampoline System", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
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
            cv2.circle(birds_eye_img, (px, py), 12, landing["color"], -1)
            cv2.circle(birds_eye_img, (px, py), 12, (0, 0, 0), 2)  
            cv2.putText(birds_eye_img, str(index + 1), (px + 16, py + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imwrite("birds_eye_landing_summary.png", birds_eye_img)
        print("\n【全工程完了】鳥瞰図『birds_eye_landing_summary.png』も保存されました。")

if __name__ == "__main__":
    main()