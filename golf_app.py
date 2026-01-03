import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- 1. 基本設定と関数 ---
st.set_page_config(layout="wide", page_title="K's Golf AI Coach")

# 角度計算ロジック
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

# --- 2. 映像処理クラス（ここがスマホ対応のキモ！） ---
# このクラスが、スマホから送られてきた映像を1枚ずつ加工して送り返す
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        # クラスが作られたときにAI（MediaPipe）を準備する
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        # 1. スマホから映像フレームを受け取る
        img = frame.to_ndarray(format="bgr24")

        # 2. 画像処理（いつものやつ！）
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = self.pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 3. 骨格検出と描画
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 座標取得
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # 角度計算
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # 判定ロジック
            if angle > 160:
                color = (0, 255, 0) # 緑
                stage = "Good!"
            else:
                color = (0, 0, 255) # 赤
                stage = "Bad"

            # 骨格描画
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
            )

            # 判定表示
            # 画面上部に帯をつける
            cv2.rectangle(image, (0,0), (image.shape[1], 50), color, -1)
            cv2.putText(image, f'{stage} Angle: {int(angle)}', (10,35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # 4. 加工した映像をスマホ送り返す
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- 3. アプリの見た目（UI） ---
st.title("⛳️ K's Golf AI Coach")

st.sidebar.header("設定")
club_list = ["ドライバー (1W)", "アイアン (7I)", "ウェッジ", "パター"] # 長くなるので省略したが、全リスト入れてOK
club_select = st.sidebar.selectbox("クラブを選択", club_list)

col1, col2 = st.columns(2)

with col1:
    st.header(f"プロのお手本")
    st.image("https://via.placeholder.com/360x640.png?text=Pro+Swing", use_container_width=True)

with col2:
    st.header("あなたのスイング")
    st.write("下のボタンを押すとカメラが起動します")
    
    # ★ここが変更点！ WebRTCコンポーネント
    # rtc_configurationは、スマホがサーバーと通信するための設定（Googleの無料サーバーを使用）
    webrtc_streamer(
        key="golf-pose",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PoseProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )