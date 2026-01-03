import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- 1. 基本設定 (必ず一番最初に記述) ---
st.set_page_config(layout="wide", page_title="K's Golf AI Coach")

# スタイル調整（スマホで見たときに余白を減らす）
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
    """アップロードされた動画を解析して保存する関数"""
    cap = cv2.VideoCapture(input_path)
    
    # 動画の情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 保存用の設定（mp4v形式）
    # 注意: ブラウザによっては再生できない場合があります。その場合はH264変換などが必要です。
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # MediaPipeの準備
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        bar = st.progress(0) # 進捗バーを表示
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. 色変換
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 2. 推論
            results = pose.process(image)
            
            # 3. 描画準備
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 4. 骨格描画ロジック
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 座標取得
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # 角度計算
                angle = calculate_angle(shoulder, elbow, wrist)
                
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

                # テキスト表示
                cv2.rectangle(image, (0,0), (image.shape[1], 50), color, -1)
                cv2.putText(image, f'{stage} Angle: {int(angle)}', (10,35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

            # 書き出し
            out.write(image)
            
            # 進捗バー更新 (0除算回避のため frame_count チェック推奨だが簡易的に実装)
            if frame_count > 0:
                bar.progress((i + 1) / frame_count)

    cap.release()
    out.release()
    return True

# --- 3. 映像処理クラス（リアルタイム用） ---
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

# --- 4. アプリのメイン構造 ---
st.title("⛳️ K's Golf AI Coach")

# ★サイドバーでモード切替★
st.sidebar.header("メニュー")
app_mode = st.sidebar.selectbox("モードを選択", ["リアルタイム判定 (Real-time)", "動画アップロード分析 (Upload)"])

st.sidebar.divider()

# 共通設定（クラブ選択）
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
    st.write("撮影したスイング動画をアップロードして、AIが詳細に分析します。")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. プロ/お手本の動画")
        pro_video = st.file_uploader("プロの動画をアップロード", type=['mp4', 'mov'], key="pro_video")
        if pro_video is not None:
            st.video(pro_video)

    with col2:
        st.subheader("2. あなたのスイング動画")
        my_video = st.file_uploader("自分の動画をアップロード", type=['mp4', 'mov'], key="my_video")
        
        if my_video is not None:
            # アップロードされたファイルを一時ファイルとして保存
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(my_video.read())
            
            # 分析ボタン
            if st.button("🚀 AI分析を開始する"):
                st.info("分析中... しばらくお待ちください（動画の長さによって数分かかります）")
                
                # 出力用の一時ファイル名を作成
                output_file_path = tfile.name + "_processed.mp4"
                
                # ★分析実行！
                try:
                    analyze_video(tfile.name, output_file_path)
                    st.success("分析完了！")
                    
                    # 分析結果を表示
                    st.video(output_file_path)
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
