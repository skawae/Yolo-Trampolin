import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ==========================================================
# 調整パラメータ設定エリア（台の形式に合わせた実際の寸法を入力）
# ==========================================================
VIDEO_PATH = "../MediapipePose/video-export/kawae0428-1-2.mp4"  # 動画ファイルのパス

# 使用する台の実際のサイズ（メートル単位）
BLUE_REAL_WIDTH = 5.05   # 青枠（外枠フレーム）の横幅
BLUE_REAL_LENGTH = 2.91  # 青枠（外枠フレーム）の縦幅

RED_REAL_WIDTH = 2.15   # 赤枠（ベッド内寸）の横幅
RED_REAL_LENGTH = 1.08  # 赤枠（ベッド内寸）の縦幅
# ==========================================================

# グローバル変数（クリック座標とガイド文）
clicked_points = []
GUIDE_TEXTS = [
    "1/8: Click BLUE [Top-Left]",
    "2/8: Click BLUE [Top-Right]",
    "3/8: Click BLUE [Bottom-Right]",
    "4/8: Click BLUE [Bottom-Left]",
    "5/8: Click RED  [Top-Left]",
    "6/8: Click RED  [Top-Right]",
    "7/8: Click RED  [Bottom-Right]",
    "8/8: Click RED  [Bottom-Left]"
]

# マウスクリックイベントを処理する関数
def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 8:
            clicked_points.append([x, y])
            print(f"座標登録 [{len(clicked_points)}/8]: X={x}, Y={y}")

def main():
    global clicked_points
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n-pose.pt").to(device)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: {VIDEO_PATH} を開けません。")
        return

    init_frame = None
    print("\n--- [手順1] 動画再生中... 映像が映ったら『スペースキー』を押して固定してください ---")

    # --- フェーズ1: 映像が映るまで動画を再生・フレーム選択 ---
    cv2.namedWindow("Select Visible Frame")
    
    while True:
        success, frame = cap.read()
        if not success:
            # 動画の最後までいってしまった場合は先頭に戻す
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        display_select = frame.copy()
        # ユーザーへの指示を画面に表示
        cv2.putText(display_select, "PRESS 'SPACE' TO SELECT THIS FRAME", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_select, "Press 'Esc' to Quit", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow("Select Visible Frame", display_select)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 32:  # スペースキーが押されたらそのフレームで確定
            init_frame = frame.copy()
            break
        elif key == 27:  # Escキーで終了
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select Visible Frame")

    # --- フェーズ2: 固定したフレームでマウスクリック入力 ---
    cv2.namedWindow("Calibration Phase")
    cv2.setMouseCallback("Calibration Phase", mouse_callback)

    print("\n--- [手順2] 画面の指示に従って、青枠と赤枠の4角をクリックしてください ---")

    while len(clicked_points) < 8:
        display_frame = init_frame.copy()
        step = len(clicked_points)
        
        # 画面上部へのナビゲーション表示
        current_color = (255, 0, 0) if step < 4 else (0, 0, 255)
        cv2.putText(display_frame, GUIDE_TEXTS[step], (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_color, 2)

        # 青枠のリアルタイム描画
        blue_pts = clicked_points[0:4]
        for pt in blue_pts:
            cv2.circle(display_frame, (pt[0], pt[1]), 6, (255, 0, 0), -1)
        if len(blue_pts) > 1:
            cv2.polylines(display_frame, [np.array(blue_pts, np.int32)], False, (255, 0, 0), 2)
        if len(blue_pts) == 4:
            cv2.line(display_frame, tuple(blue_pts[3]), tuple(blue_pts[0]), (255, 0, 0), 2)

        # 赤枠のリアルタイム描画
        red_pts = clicked_points[4:8]
        for pt in red_pts:
            cv2.circle(display_frame, (pt[0], pt[1]), 6, (0, 0, 255), -1)
        if len(red_pts) > 1:
            cv2.polylines(display_frame, [np.array(red_pts, np.int32)], False, (0, 0, 255), 2)
        if len(red_pts) == 4:
            cv2.line(display_frame, tuple(red_pts[3]), tuple(red_pts[0]), (0, 0, 255), 2)

        cv2.imshow("Calibration Phase", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Calibration Phase")

    # 座標の確定
    blue_image_pts = np.array(clicked_points[0:4], dtype=np.float32)
    red_image_pts = np.array(clicked_points[4:8], dtype=np.float32)

    # --- フェーズ3: 2D DLTの実世界座標マッピング ---
    red_real_pts = np.array([
        [-RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2],
        [ RED_REAL_WIDTH/2, -RED_REAL_LENGTH/2],
        [ RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2],
        [-RED_REAL_WIDTH/2,  RED_REAL_LENGTH/2]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(red_image_pts, red_real_pts)
    prev_real_y = None

    print("\n--- [手順3] 設定完了。トラッキングを開始します ---")

    # 動画の読み込み位置を最初に戻して解析スタート
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- フェーズ4: メイン解析ループ ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, verbose=False, conf=0.4)

        # 登録された2つの枠線を常時表示
        cv2.polylines(frame, [blue_image_pts.astype(np.int32)], True, (255, 0, 0), 3)
        cv2.polylines(frame, [red_image_pts.astype(np.int32)], True, (0, 0, 255), 3)

        if results and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.xy.cpu().numpy()
            
            if len(keypoints_data) > 0:
                person = keypoints_data[0]
                frame = results[0].plot()

                # 足首座標の取得
                left_ankle = person[15] if len(person) > 15 else [0, 0]
                right_ankle = person[16] if len(person) > 16 else [0, 0]
                
                img_x, img_y = None, None
                if left_ankle[1] > 0 and right_ankle[1] > 0:
                    img_x, img_y = (left_ankle[0] + right_ankle[0])/2, (left_ankle[1] + right_ankle[1])/2
                elif left_ankle[1] > 0:
                    img_x, img_y = left_ankle[0], left_ankle[1]
                elif right_ankle[1] > 0:
                    img_x, img_y = right_ankle[0], right_ankle[1]

                # 座標変換とエリア判定
                if img_x is not None and img_y is not None:
                    img_coord = np.array([[[img_x, img_y]]], dtype=np.float32)
                    real_coord = cv2.perspectiveTransform(img_coord, H)
                    real_x, real_y = real_coord[0][0][0], real_coord[0][0][1]

                    is_inside_red = (-RED_REAL_WIDTH/2 <= real_x <= RED_REAL_WIDTH/2 and 
                                     -RED_REAL_LENGTH/2 <= real_y <= RED_REAL_LENGTH/2)
                    is_inside_blue = (-BLUE_REAL_WIDTH/2 <= real_x <= BLUE_REAL_WIDTH/2 and 
                                      -BLUE_REAL_LENGTH/2 <= real_y <= BLUE_REAL_LENGTH/2)

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

                    # 着地判定
                    if prev_real_y is not None:
                        if abs(real_y - prev_real_y) < 0.04: 
                            cv2.putText(frame, f"TOUCH: {area_name}", (50, 80), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
                            print(f"[着地] エリア: {area_name} | 中心から X: {real_x:.2f}m, Y: {real_y:.2f}m")

                    prev_real_y = real_y

        cv2.imshow("Universal Trampoline System", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()