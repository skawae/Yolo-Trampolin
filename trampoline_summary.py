import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==========================================================
# 調整パラメータ設定エリア
# ==========================================================
VIDEO_PATH = "../MediapipePose/video-export/kawae0428-1-2.mp4"  # 解析する動画ファイルのパス

# 使用する台の実際のサイズ（メートル単位）
BLUE_REAL_WIDTH = 5.05   
BLUE_REAL_LENGTH = 2.91  
RED_REAL_WIDTH = 2.15   
RED_REAL_LENGTH = 1.08  

# --- 【最重要】鳥瞰図の位置ズレを修正するパラメータ ---
# 鳥瞰図のドットを直接メートル単位でズラすことができます。
# 例: 鳥瞰図の点が実際より「左に20cm」ずれている場合、右に直すために +0.2 を設定します。
OFFSET_X = 0.0  # 横方向の補正（プラスで右へ、マイナスで左へ移動 / 単位: メートル）
OFFSET_Y = -0.5  # 縦方向の補正（プラスで下へ、マイナスで上へ移動 / 単位: メートル）

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

def main():
    global clicked_points, all_landing_points, x_history, y_history
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n-pose.pt").to(device)

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

    # フェーズ3: 2D DLT行列の生成（赤枠基準）
    blue_image_pts = np.array(clicked_points[0:4], dtype=np.float32)
    red_image_pts = np.array(clicked_points[4:8], dtype=np.float32)

    # 実世界座標の定義（左上、右上、右下、左下）
    red_real_pts = np.array([
        [-RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2], 
        [ RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2],
        [ RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2], 
        [-RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(red_image_pts, red_real_pts)

    pre_prev_real_y = None
    prev_real_y = None

    print("\n--- [手順3] 解析を開始します ---")
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
            keypoints_data = results[0].keypoints.xy.cpu().numpy()
            
            if len(keypoints_data) > 0:
                person = keypoints_data[0]
                frame = results[0].plot()

                left_ankle = person[15] if len(person) > 15 else [0, 0]
                right_ankle = person[16] if len(person) > 16 else [0, 0]
                raw_x, raw_y = None, None
                if left_ankle[1] > 0 and right_ankle[1] > 0:
                    raw_x, raw_y = (left_ankle[0] + right_ankle[0])/2, (left_ankle[1] + right_ankle[1])/2
                elif left_ankle[1] > 0:
                    raw_x, raw_y = left_ankle[0], left_ankle[1]
                elif right_ankle[1] > 0:
                    raw_x, raw_y = right_ankle[0], right_ankle[1]

                if raw_x is not None and raw_y is not None:
                    x_history.append(raw_x)
                    y_history.append(raw_y)
                    img_x = sum(x_history) / len(x_history)
                    img_y = sum(y_history) / len(y_history)

                    # 2D DLTによる純粋な幾何変換の実行
                    img_coord = np.array([[[img_x, img_y]]], dtype=np.float32)
                    real_coord = cv2.perspectiveTransform(img_coord, H)
                    
                    # 【新処理】手動設定したオフセット値をここで適用して位置を補正
                    real_x = real_coord[0][0][0] + OFFSET_X
                    real_y = real_coord[0][0][1] + OFFSET_Y

                    is_inside_red = (-RED_REAL_WIDTH/2 <= real_x <= RED_REAL_WIDTH/2 and -RED_REAL_LENGTH/2 <= real_y <= RED_REAL_LENGTH/2)
                    is_inside_blue = (-BLUE_REAL_WIDTH/2 <= real_x <= BLUE_REAL_WIDTH/2 and -BLUE_REAL_LENGTH/2 <= real_y <= BLUE_REAL_LENGTH/2)

                    if is_inside_red:
                        cv2.circle(frame, (int(img_x), int(img_y)), 10, (0, 255, 0), -1)
                        area_name = "CENTER (RED)"
                        text_color = (0, 255, 0)
                    elif is_inside_blue:
                        cv2.circle(frame, (int(img_x), int(img_y)), 10, (0, 165, 255), -1)
                        area_name = "SAFETY (BLUE)"
                        text_color = (0, 165, 255)
                    else:
                        cv2.circle(frame, (int(img_x), int(img_y)), 10, (0, 0, 255), -1)
                        area_name = "OUT OF BOUNDS"
                        text_color = (0, 0, 255)

                    # 最下点判定
                    if pre_prev_real_y is not None and prev_real_y is not None:
                        if pre_prev_real_y < prev_real_y and prev_real_y > real_y:
                            if is_inside_blue:  
                                cv2.putText(frame, "TOUCH!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 4)
                                print(f"[着地] {area_name} | 補正後座標 X: {real_x:.2f}m, Y: {real_y:.2f}m")
                                
                                all_landing_points.append({
                                    "real_pos": (real_x, real_y),
                                    "area": area_name,
                                    "color": text_color
                                })

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
            px = int(mx * SCALE + canvas_width / 2)
            py = int(my * SCALE + canvas_height / 2)
            return (px, py)

        cv2.rectangle(birds_eye_img, meter_to_pixel(-BLUE_REAL_WIDTH/2, -BLUE_REAL_LENGTH/2), meter_to_pixel(BLUE_REAL_WIDTH/2, BLUE_REAL_LENGTH/2), (255, 0, 0), 3)
        cv2.rectangle(birds_eye_img, meter_to_pixel(-RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2), meter_to_pixel(RED_REAL_WIDTH/2, RED_REAL_LENGTH/2), (0, 0, 255), 3)
        cv2.drawMarker(birds_eye_img, meter_to_pixel(0, 0), (200, 200, 200), cv2.MARKER_CROSS, 20, 2)

        for index, landing in enumerate(all_landing_points):
            mx, my = landing["real_pos"]
            color = landing["color"]
            px, py = meter_to_pixel(mx, my)
            cv2.circle(birds_eye_img, (px, py), 12, color, -1)
            cv2.circle(birds_eye_img, (px, py), 12, (0, 0, 0), 2)  
            cv2.putText(birds_eye_img, str(index + 1), (px + 16, py + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        output_filename = "birds_eye_landing_summary.png"
        cv2.imwrite(output_filename, birds_eye_img)
        print(f"【保存完了】手動補正適用済みの鳥瞰図：『{output_filename}』")

if __name__ == "__main__":
    main()