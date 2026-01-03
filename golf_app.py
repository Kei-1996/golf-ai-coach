import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import tempfile
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- 1. 基本設定 ---
st.set_page_config(layout="wide", page_title="K's Golf AI Coach")

st.markdown("""
    <style>
    .main > div {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)

# --- Session State (記憶領域) の初期化 ---
if 'pro_processed_video' not in st.session_state:
    st.session_state['pro_processed_video'] = None
if 'pro_df' not in st.session_state:
    st.session_state['pro_df'] = None
if 'my_processed_video' not in st.session_state:
    st.session_state['my_processed_video'] = None
if 'my_df' not in st.session_state:
    st.session_state['my_df'] = None

# --- 2. 計算用関数 ---

def calculate_angle(a, b, c):
    """3点の座標から角度を計算"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

def analyze_video(input_path, output_path):
    """動画解析＆データ抽出（トップ位置検出用に手首Y座標も保存）"""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose_data = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        bar = st.progress(0)
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 座標取得
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow    = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist    = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                l_hip      = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                
                # ★データを保存 (トップ検出のためにWrist_Yを追加)
                pose_data.append({
                    "Frame": i,
                    "Time_Sec": i / fps if fps > 0 else 0,
                    "Arm_Angle": angle,
                    "L_Shoulder_X": l_shoulder[0],
                    "L_Shoulder_Y": l_shoulder[1],
                    "L_Wrist_Y": l_wrist[1],  # Y座標が小さいほど高い位置
                    "L_Hip_X": l_hip[0],
                    "L_Hip_Y": l_hip[1]
                })

                # 描画用判定
                if angle > 160:
                    color = (0, 255, 0)
                    stage = "Good!"
                else:
                    color = (0, 0, 255)
                    stage = "Bad"

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )

                cv2.rectangle(image, (0,0), (image.shape[1], 50), color, -1)
                cv2.putText(image, f'{stage} Angle: {int(angle)}', (10,35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

            out.write(image)
            if frame_count > 0:
                bar.progress((i + 1) / frame_count)

    cap.release()
    out.release()
    df = pd.DataFrame(pose_data)
    return output_path, df

# --- 3. 映像処理クラス（リアルタイム用） ---
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

# --- 4. アプリのメイン構造 ---
st.title("⛳️ K's Golf AI Coach")

st.sidebar.header("メニュー")
app_mode = st.sidebar.selectbox("モードを選択", ["リアルタイム判定 (Real-time)", "動画アップロード分析 (Upload)"])
st.sidebar.divider()
club_list = ["ドライバー (1W)", "アイアン (7I)", "ウェッジ", "パター"]
club_select = st.sidebar.selectbox("使用クラブ", club_list)


# --- モードA: リアルタイム判定 ---
if app_mode == "リアルタイム判定 (Real-time)":
    st.header("⚡️ リアルタイム・コーチ")
    col1, col2 = st.columns(2)
    with col1:
        st.info("👈 プロのお手本動画")
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

# --- モードB: 動画アップロード分析 ---
elif app_mode == "動画アップロード分析 (Upload)":
    st.header("📂 動画分析ラボ")
    
    # ★注意書きの追加 (重要！)
    st.warning("⚠️ **重要:** 正確な比較のため、**プロの動画と「同じアングル（正面/後方）」** で撮影された動画を使用してください。アングルが異なるとAIが正しく判定できません。")

    col1, col2 = st.columns(2)
    
    # --- 左カラム: プロ ---
    with col1:
        st.subheader("1. プロ/お手本の動画")
        pro_video = st.file_uploader("プロの動画", type=['mp4', 'mov'], key="pro_video")
        
        if pro_video is not None:
            st.video(pro_video)
            if st.button("🔍 プロ動画を解析"):
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(pro_video.read())
                output_pro = tfile.name + "_pro_processed.mp4"
                
                with st.spinner("プロ解析中..."):
                    path, df = analyze_video(tfile.name, output_pro)
                    st.session_state['pro_processed_video'] = path
                    st.session_state['pro_df'] = df
                    st.success("解析完了！")

            if st.session_state['pro_processed_video']:
                st.write("---")
                st.video(st.session_state['pro_processed_video'])
                csv = st.session_state['pro_df'].to_csv(index=False).encode('utf-8')
                st.download_button("📥 プロのCSV DL", csv, 'pro_data.csv', 'text/csv')

    # --- 右カラム: 自分 ---
    with col2:
        st.subheader("2. あなたのスイング動画")
        my_video = st.file_uploader("自分の動画", type=['mp4', 'mov'], key="my_video")
        
        if my_video is not None:
            if st.button("🚀 自分の動画を解析"):
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(my_video.read())
                output_my = tfile.name + "_my_processed.mp4"
                
                with st.spinner("自分解析中..."):
                    path, df = analyze_video(tfile.name, output_my)
                    st.session_state['my_processed_video'] = path
                    st.session_state['my_df'] = df
                    st.success("解析完了！")

            if st.session_state['my_processed_video']:
                st.write("---")
                st.video(st.session_state['my_processed_video'])
                csv_my = st.session_state['my_df'].to_csv(index=False).encode('utf-8')
                st.download_button("📥 自分のCSV DL", csv_my, 'my_data.csv', 'text/csv')

    # --- ★比較診断セクション (両方のデータが揃ったら表示) ---
    if st.session_state['pro_df'] is not None and st.session_state['my_df'] is not None:
        st.divider()
        st.header("🤖 AIコーチの診断レポート")
        
        # 1. トップ位置（手首が一番高い位置）を探す
        # L_Wrist_Y は画面上が0、下が1なので、最小値が一番高い位置
        pro_df = st.session_state['pro_df']
        my_df = st.session_state['my_df']
        
        # プロのトップ
        pro_top_idx = pro_df['L_Wrist_Y'].idxmin()
        pro_top_angle = pro_df.iloc[pro_top_idx]['Arm_Angle']
        
        # 自分のトップ
        my_top_idx = my_df['L_Wrist_Y'].idxmin()
        my_top_angle = my_df.iloc[my_top_idx]['Arm_Angle']
        
        # 2. 比較結果の表示
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("プロのトップ時 左肘角度", f"{int(pro_top_angle)}°")
        col_res2.metric("あなたのトップ時 左肘角度", f"{int(my_top_angle)}°")
        
        diff = my_top_angle - pro_top_angle
        col_res3.metric("差分", f"{int(diff)}°", delta=-diff) # 差が大きいと赤くなるように設定

        # 3. アドバイス生成
        st.subheader("💡 ワンポイント・アドバイス")
        if abs(diff) < 15:
            st.success("素晴らしい！プロとほぼ同じ肘の伸び具合です。この調子でキープしましょう！")
        elif diff > 15:
            st.error("肘が曲がりすぎています（チキンウィング気味）。トップでもう少し腕を伸ばす意識を持ちましょう。")
        else:
            st.warning("肘が伸びすぎて硬くなっている可能性があります。もう少しリラックスしても良いかもしれません。")
