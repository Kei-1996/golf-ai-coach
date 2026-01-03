import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- 1. 基本設定と関数 ---
st.set_page_config(layout="wide", page_title="K's Golf AI Coach")

# スタイル調整（スマホで見たときに余白を減らす）
st.markdown("""
    <style>
    .main > div {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)

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

# --- 2. 映像処理クラス（リアルタイム用） ---
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 画像処理
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 骨格検出と描画
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle = calculate_angle(shoulder, elbow, wrist)
            
            if angle > 160:
                color = (0, 255, 0)
                stage = "Good!"
            else:
                color = (0, 0, 255)
                stage = "Bad"

            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
            )

            cv2.rectangle(image, (0,0), (image.shape[1], 50), color, -1)
            cv2.putText(image, f'{stage} Angle: {int(angle)}', (10,35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- 3. アプリのメイン構造 ---
st.title("⛳️ K's Golf AI Coach")

# ★サイドバーでモード切替★
st.sidebar.header("メニュー")
app_mode = st.sidebar.selectbox("モードを選択", ["リアルタイム判定 (Real-time)", "動画アップロード分析 (Upload)"])

st.sidebar.divider()

# 共通設定（クラブ選択）
club_list = ["ドライバー (1W)", "アイアン (7I)", "ウェッジ", "パター"]
club_select = st.sidebar.selectbox("使用クラブ", club_list)


# --- モードA: リアルタイム判定（今までの機能） ---
if app_mode == "リアルタイム判定 (Real-time)":
    st.header("⚡️ リアルタイム・コーチ")
    st.write("友達に撮ってもらいながら、フォームをチェックしよう！")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("👈 プロのお手本動画 (ここに表示)")
        st.image("https://via.placeholder.com/360x640.png?text=Pro+Swing", use_container_width=True)
    
    with col2:
        st.success("📸 カメラ映像")
        webrtc_streamer(
            key="golf-pose-realtime",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=PoseProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

# --- モードB: 動画アップロード分析（これからの機能） ---
elif app_mode == "動画アップロード分析 (Upload)":
    st.header("📂 動画分析ラボ")
    st.write("撮影したスイング動画をアップロードして、AIが詳細に分析します。")

    col1, col2 = st.columns(2)
    
    # 左側：プロの動画（アップロード機能）
    with col1:
        st.subheader("1. プロ/お手本の動画")
        pro_video = st.file_uploader("プロの動画をアップロード", type=['mp4', 'mov'], key="pro_video")
        if pro_video is not None:
            st.video(pro_video)
        else:
            st.info("比較したいお手本動画があればアップロードしてください")

    # 右側：自分の動画（アップロード機能）
    with col2:
        st.subheader("2. あなたのスイング動画")
        my_video = st.file_uploader("自分の動画をアップロード", type=['mp4', 'mov'], key="my_video")
        if my_video is not None:
            st.video(my_video) # とりあえず再生するだけ
            
            # ここに後で「分析スタートボタン」を作る！
            if st.button("🚀 AI分析を開始する"):
                st.warning("⚠️ 分析機能は現在開発中です！")
        else:
            st.info("スマホで撮ったスイング動画をアップロードしてください")
