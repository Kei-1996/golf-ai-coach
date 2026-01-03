import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import tempfile
import pandas as pd  # ★追加: データ分析用ライブラリ
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- 1. 基本設定 ---
st.set_page_config(layout="wide", page_title="K's Golf AI Coach")

st.markdown("""
    <style>
    .main > div {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. 計算用関数 ---

def calculate_angle(a, b, c):
    """3点の座標から角度を計算する関数"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def analyze_video(input_path, output_path):
    """
    動画を解析し、加工済み動画のパスと、抽出した骨格データ(DataFrame)を返す
    """
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # ★データ保存用のリスト
    pose_data = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        bar = st.progress(0)
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 画像処理
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 骨格描画 & データ抽出
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 必要な部位の座標を取得 (今回は左腕を中心に取得)
                # ※必要に応じて右腕や腰なども追加できる
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow    = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist    = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                l_hip      = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                # 角度計算
                angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                
                # ★データをリストに記録 (これがCSVの元になる)
                pose_data.append({
                    "Frame": i,
                    "Time_Sec": i / fps if fps > 0 else 0,
                    "Arm_Angle": angle,
                    "L_Shoulder_X": l_shoulder[0],
                    "L_Shoulder_Y": l_shoulder[1],
                    "L_Hip_X": l_hip[0],
                    "L_Hip_Y": l_hip[1]
                })

                # 色判定
                if angle > 160:
                    color = (0, 255, 0)
                    stage = "Good!"
                else:
                    color = (0, 0, 255)
                    stage = "Bad"

                # 描画
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
    
    # ★リストをDataFrame(表)に変換して返す
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

# --- モードB: 動画アップロード分析 ---
elif app_mode == "動画アップロード分析 (Upload)":
    st.header("📂 動画分析ラボ")
    st.write("プロの動画から「正解データ」を作り、自分の動画と比較しよう。")

    col1, col2 = st.columns(2)
    
    # --- 左カラム: プロの動画 ---
    with col1:
        st.subheader("1. プロ/お手本の動画")
        pro_video = st.file_uploader("プロの動画をアップロード", type=['mp4', 'mov'], key="pro_video")
        
        if pro_video is not None:
            # プレビュー
            st.video(pro_video)
            
            # ★プロ動画解析ボタン
            if st.button("🔍 プロ動画を解析してデータ化 (CSV)"):
                tfile_pro = tempfile.NamedTemporaryFile(delete=False) 
                tfile_pro.write(pro_video.read())
                output_pro = tfile_pro.name + "_pro_processed.mp4"
                
                with st.spinner("プロの動きを数値化しています..."):
                    processed_path, df_pro = analyze_video(tfile_pro.name, output_pro)
                
                st.success("解析完了！骨格データ抽出成功。")
                
                # 加工済み動画を表示
                st.video(processed_path)
                
                # ★抽出データ(CSV)のダウンロード
                csv = df_pro.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 プロの骨格データ(CSV)をダウンロード",
                    data=csv,
                    file_name='pro_swing_data.csv',
                    mime='text/csv',
                )
                
                # データのチラ見せ
                with st.expander("抽出されたデータの確認"):
                    st.dataframe(df_pro)

    # --- 右カラム: 自分の動画 ---
    with col2:
        st.subheader("2. あなたのスイング動画")
        my_video = st.file_uploader("自分の動画をアップロード", type=['mp4', 'mov'], key="my_video")
        
        if my_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(my_video.read())
            
            if st.button("🚀 自分のスイングを分析"):
                st.info("分析中...")
                output_file_path = tfile.name + "_processed.mp4"
                
                # 自分の動画もデータ化はするが、今は表示がメイン
                processed_path, df_my = analyze_video(tfile.name, output_file_path)
                
                st.success("分析完了！")
                st.video(output_file_path)
