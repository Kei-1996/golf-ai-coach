import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# --- 1. 基本設定 & デザインシステム ---
st.set_page_config(layout="wide", page_title="K's Golf AI Coach")

# オーガニック・モダンデザイン (クリームベース、角丸、目に優しい配色)
st.markdown("""
    <style>
    /* 全体の背景色とフォント設定 */
    .stApp {
        background-color: #FAF9F6; /* オフホワイト/クリーム */
    }
    
    html, body, [class*="css"] {
        font-family: "Hiragino Kaku Gothic ProN", "Hiragino Sans", "Meiryo", sans-serif;
        color: #4A4A4A; /* 真っ黒ではなくダークグレーで目に優しく */
    }
    
    /* ヘッダー周り */
    h1, h2, h3 {
        color: #2C3E50;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    
    .main > div { padding-top: 2rem; }

    /* ビデオ表示 (角を大きく丸める) */
    video { 
        width: 100% !important; 
        height: auto !important; 
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(149, 157, 165, 0.1); /* 柔らかい影 */
    }
    
    /* --- カードデザイン (浮き上がるような柔らかい白) --- */
    .soft-card {
        background-color: #FFFFFF;
        border: none;
        border-radius: 24px; /* かなり丸く */
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03); /* 浮遊感 */
        text-align: center;
    }
    
    .score-label {
        font-size: 0.8rem;
        color: #8D99AE;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    
    .total-score-val {
        font-size: 4rem;
        font-weight: 500;
        color: #2F3E46; /* 深い緑がかったグレー */
        line-height: 1.0;
        margin-bottom: 12px;
    }
    
    .metric-val {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2F3E46;
    }
    
    .advice-text {
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 8px;
        line-height: 1.6;
    }

    /* --- メッセージボックス (線ではなく背景色で優しく) --- */
    .info-box {
        background-color: #E8F4F8; /* 薄い空色 */
        padding: 16px 20px;
        border-radius: 16px;
        margin: 16px 0;
        color: #4A6FA5;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .warning-box {
        background-color: #FFF0F0; /* 薄いピンク */
        padding: 16px 20px;
        border-radius: 16px;
        margin: 16px 0;
        color: #C0392B;
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* --- ボタンのカスタマイズ (丸みを帯びたピル型) --- */
    div.stButton > button {
        border-radius: 50px; /* ピル型 */
        border: none;
        background-color: #FFFFFF;
        color: #555;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #6B705C; /* 落ち着いたオリーブグリーン */
        color: #FFFFFF;
        box-shadow: 0 6px 15px rgba(107, 112, 92, 0.3);
        transform: translateY(-2px);
    }
    
    /* サイドバーの背景も合わせる */
    section[data-testid="stSidebar"] {
        background-color: #F4F3F0; /* 本体より少し濃いクリーム */
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State ---
if 'club_data' not in st.session_state: st.session_state['club_data'] = {}
if 'my_processed_video' not in st.session_state: st.session_state['my_processed_video'] = None
if 'my_df' not in st.session_state: st.session_state['my_df'] = None
if 'my_metrics' not in st.session_state: st.session_state['my_metrics'] = None
if 'sync_video_path' not in st.session_state: st.session_state['sync_video_path'] = None

# --- 2. 計算・解析用関数 ---

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def get_vertical_angle(a, b):
    a = np.array(a)
    b = np.array(b)
    radians = np.arctan2(abs(a[0]-b[0]), abs(a[1]-b[1]))
    angle = np.abs(radians*180.0/np.pi)
    return angle

def analyze_video_advanced(input_path, output_path, rotate_mode="なし"):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if rotate_mode in ["時計回りに90度", "反時計回りに90度"]:
        out_width, out_height = height, width
    else:
        out_width, out_height = width, height
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    # 線を白と落ち着いたグレーに
    drawing_spec_points = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)
    drawing_spec_lines = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2)

    pose_data = []
    nose_x_list = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        bar = st.progress(0)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            
            if rotate_mode == "時計回りに90度": frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_mode == "反時計回りに90度": frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            frame_data = {"Frame": i, "Time": i/fps if fps>0 else 0, "L_Wrist_Y": None}

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                nose = [lm[mp_pose.PoseLandmark.NOSE].x, lm[mp_pose.PoseLandmark.NOSE].y]
                l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                l_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y]
                l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                r_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                spine_angle = get_vertical_angle(l_shoulder, l_hip)
                knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

                frame_data.update({
                    "L_Wrist_Y": l_wrist[1],
                    "Arm_Angle": arm_angle,
                    "Spine_Angle": spine_angle,
                    "R_Knee_Angle": knee_angle,
                    "Nose_X": nose[0]
                })
                nose_x_list.append(nose[0])
                
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec_points,
                    connection_drawing_spec=drawing_spec_lines
                )

            pose_data.append(frame_data)
            out.write(image)
            if frame_count > 0: bar.progress((i + 1) / frame_count)

    cap.release()
    out.release()
    df = pd.DataFrame(pose_data)
    
    if not df.empty and df['L_Wrist_Y'].notnull().any():
        top_idx = df['L_Wrist_Y'].idxmin()
        top_frame = df.loc[top_idx, 'Frame']
        addr_df = df[df['Frame'] < top_frame]
        address_frame = df.loc[addr_df['L_Wrist_Y'].idxmax(), 'Frame'] if not addr_df.empty else 0
        imp_df = df[df['Frame'] > top_frame]
        impact_frame = df.loc[imp_df['L_Wrist_Y'].idxmax(), 'Frame'] if not imp_df.empty else frame_count-1
        
        top_data = df.loc[top_idx]
        metrics = {
            'fps': fps,
            'top_frame': int(top_frame),
            'address_frame': int(address_frame),
            'impact_frame': int(impact_frame),
            'head_stability': np.std(nose_x_list) if nose_x_list else 0,
            'spine_angle_top': top_data['Spine_Angle'],
            'knee_angle_top': top_data['R_Knee_Angle'],
            'top_arm_angle': top_data['Arm_Angle']
        }
    else:
        metrics = None

    return output_path, df, metrics

def create_sync_video(pro_path, my_path, pro_metrics, my_metrics, output_path):
    cap_pro = cv2.VideoCapture(pro_path)
    cap_my = cv2.VideoCapture(my_path)
    h_pro = int(cap_pro.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_my = int(cap_my.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if h_pro == 0 or h_my == 0: return

    target_h = min(h_pro, h_my)
    w_pro = int(cap_pro.get(cv2.CAP_PROP_FRAME_WIDTH))
    w_my = int(cap_my.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_w_pro = int(w_pro * (target_h / h_pro))
    new_w_my = int(w_my * (target_h / h_my))
    target_w = new_w_pro + new_w_my

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, my_metrics['fps'], (target_w, target_h))

    pro_top = pro_metrics['top_frame']
    my_top = my_metrics['top_frame']
    offset = my_top - pro_top
    pro_delay = max(0, offset)
    my_delay = max(0, -offset)
    max_frames = int(max(cap_pro.get(cv2.CAP_PROP_FRAME_COUNT) + pro_delay, 
                         cap_my.get(cv2.CAP_PROP_FRAME_COUNT) + my_delay))

    bar = st.progress(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i in range(max_frames):
        if i < pro_delay:
            cap_pro.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_pro, frame_pro = cap_pro.read()
        else:
            ret_pro, frame_pro = cap_pro.read()
        
        if i < my_delay:
            cap_my.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_my, frame_my = cap_my.read()
        else:
            ret_my, frame_my = cap_my.read()
            
        if not ret_pro or not ret_my: break

        frame_pro_resized = cv2.resize(frame_pro, (new_w_pro, target_h))
        frame_my_resized = cv2.resize(frame_my, (new_w_my, target_h))
        concat_frame = cv2.hconcat([frame_pro_resized, frame_my_resized])
        
        if i == (pro_top + pro_delay): 
            text_size = cv2.getTextSize("TOP MATCH", font, 1.5, 3)[0]
            tx = (target_w - text_size[0]) // 2
            cv2.putText(concat_frame, "TOP MATCH", (tx, 100), font, 1.5, (0,0,255), 3)
        
        out.write(concat_frame)
        bar.progress((i+1)/max_frames)

    bar.progress(1.0)
    cap_pro.release()
    cap_my.release()
    out.release()
    return

def generate_advice(label, pro_val, my_val):
    diff = my_val - pro_val
    msg = ""
    score = 100
    abs_diff = abs(diff)
    if abs_diff < 5: score = 100
    else: score = max(0, int(100 - abs_diff * 2))

    if label == "Arm":
        if diff < -15: msg = "プロより肘が曲がっています。左腕を伸ばす意識を。"
        elif diff > 10: msg = "腕が伸びすぎています。少しリラックスを。"
        else: msg = "プロ同様、理想的なフォームです。"
    elif label == "Spine":
        if diff < -10: msg = "プロより上体が起きています。前傾キープ。"
        elif diff > 10: msg = "前傾が深すぎます。"
        else: msg = "前傾姿勢が安定しています。"
    elif label == "Knee":
        if diff > 10: msg = "膝が伸びて棒立ちになっています。"
        elif diff < -10: msg = "膝を曲げすぎています。"
        else: msg = "膝の角度が安定しています。"
    elif label == "Tempo":
        if my_val < 2.5: msg = "打ち急ぎです。バックスイングをゆったり。"
        elif my_val > 3.5: msg = "始動が遅すぎます。リズムよく。"
        else: msg = "理想的なリズム(3:1)です。"
        score = max(0, int(100 - abs(3.0 - my_val)*30))
    elif label == "Head":
        if my_val > pro_val * 2: msg = "頭のブレが大きいです。軸を意識して。"
        else: msg = "体幹が強く、軸が安定しています。"
        score = max(0, int(100 - (my_val * 1000)))
    return score, msg

# --- 3. リアルタイム分析クラス ---
class RealtimeCoach(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.target_metrics = None 

    def update_target(self, metrics):
        self.target_metrics = metrics

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        cv2.putText(img, "AI Coach", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1, cv2.LINE_AA)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            self.mp_drawing.draw_landmarks(
                img, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(200,200,200), thickness=1)
            )
            
            l_shoulder = [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            l_elbow = [lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            l_wrist = [lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x, lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            current_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            
            if self.target_metrics:
                target_arm = self.target_metrics['top_arm_angle']
                overlay = img.copy()
                cv2.rectangle(overlay, (10, 60), (300, 160), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
                
                cv2.putText(img, f"Current: {int(current_arm_angle)} deg", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"Target: {int(target_arm)} deg", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                
                diff = current_arm_angle - target_arm
                if abs(diff) < 15:
                    cv2.putText(img, "GOOD POSE", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2, cv2.LINE_AA)
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. サイドバー設定 ---
st.sidebar.title("メニュー")
selected_club = st.sidebar.selectbox("クラブ選択", ["ドライバー", "フェアウェイウッド", "7番アイアン", "ウェッジ", "パター"])
app_mode = st.sidebar.radio("モード選択", ["1. プロ動画登録", "2. スイング診断", "3. 比較動画作成", "4. リアルタイム・コーチ"])

# --- 5. 予約・検索リンク (アフィリエイトエリア) ---
st.sidebar.markdown("---")
st.sidebar.markdown("##### コース・レッスン予約")
st.sidebar.caption("※本ページはプロモーションが含まれています")

# 1. 楽天GORA
rakuten_affiliate_url = "https://hb.afl.rakuten.co.jp/hgc/4fb95961.88417fd4.4fb95962.222603ac/?pc=https%3A%2F%2Fgora.golf.rakuten.co.jp%2F&link_type=text&ut=eyJwYWdlIjoidXJsIiwidHlwZSI6InRleHQiLCJjb2wiOjF9" 

if rakuten_affiliate_url:
    st.sidebar.link_button("楽天GORAで予約", rakuten_affiliate_url)
else:
    st.sidebar.button("楽天GORA (設定待ち)", disabled=True)

# 2. じゃらんゴルフ
jalan_affiliate_url = "https://px.a8.net/svt/ejp?a8mat=4AUXWQ+EXMG1E+36SI+64C3M"

if jalan_affiliate_url:
    st.sidebar.link_button("じゃらんゴルフで検索", jalan_affiliate_url)
else:
    st.sidebar.button("じゃらんゴルフ (設定待ち)", disabled=True)

# 3. レッスン予約
lesson_affiliate_url = "https://www17.a8.net/0.gif?a8mat=4AUXWQ+F4RNAQ+CW6+BETIUA"

if lesson_affiliate_url:
    st.sidebar.link_button("スクールを探す", lesson_affiliate_url)
else:
    st.sidebar.button("レッスン (設定待ち)", disabled=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #888;">
    <strong>免責事項</strong><br>
    本アプリの解析結果はAIによる推定値です。正確性を保証するものではありません。
    周囲の安全に十分配慮してご利用ください。
</div>
""", unsafe_allow_html=True)


# --- 6. メインコンテンツ ---
st.title("K's Golf AI Coach")

# PAGE 1: プロ動画登録
if app_mode == "1. プロ動画登録":
    st.header("お手本動画の設定")
    st.markdown('<div class="info-box">プロのスイング動画を登録します。あなたのスイングと比較する基準になります。</div>', unsafe_allow_html=True)
    
    if selected_club not in st.session_state['club_data']:
        st.session_state['club_data'][selected_club] = {}

    tab_side, tab_front = st.tabs(["後方 (Down-the-line)", "体の正面 (Face-on)"])
    
    def register_pro_video(angle_key, angle_name):
        current_data = st.session_state['club_data'][selected_club].get(angle_key)
        if current_data:
            st.success(f"{angle_name}動画: 保存済み")
            st.video(current_data['video_path'])
            if st.button(f"動画を削除", key=f"del_{angle_key}"):
                del st.session_state['club_data'][selected_club][angle_key]
                st.rerun()
        else:
            pro_file = st.file_uploader(f"プロの{angle_name}動画をアップロード", type=['mp4', 'mov'], key=f"up_{angle_key}")
            pro_rotate = st.selectbox("回転オプション", ["なし", "時計回りに90度", "反時計回りに90度"], key=f"rot_{angle_key}")
            
            rot_map = {"なし": "なし", "時計回りに90度": "時計回りに90度", "反時計回りに90度": "反時計回りに90度"}
            
            if pro_file and st.button(f"解析して保存", key=f"btn_{angle_key}"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(pro_file.read())
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                with st.spinner('AI解析中...'):
                    processed_path, df, metrics = analyze_video_advanced(tfile.name, out_path, rot_map[pro_rotate])
                    if metrics:
                        st.session_state['club_data'][selected_club][angle_key] = {'video_path': processed_path, 'metrics': metrics}
                        st.success(f"{angle_name}データを保存しました！")
                        st.rerun()

    with tab_side:
        register_pro_video('Side', '後方')
    with tab_front:
        register_pro_video('Front', '体の正面')

# PAGE 2: ユーザー解析 & スコア
elif app_mode == "2. スイング診断":
    st.header("スイング診断 & スコア")

    if selected_club not in st.session_state['club_data'] or not st.session_state['club_data'][selected_club]:
        st.markdown('<div class="warning-box">プロ動画がまだ登録されていません。「プロ動画登録」から設定してください。</div>', unsafe_allow_html=True)
    else:
        available_angles = list(st.session_state['club_data'][selected_club].keys())
        target_angle = st.radio(
            "比較するアングルを選択", 
            available_angles, 
            format_func=lambda x: "体の正面 (Face-on)" if x=="Front" else "後方 (Down-the-line)"
        )
        
        pro_data = st.session_state['club_data'][selected_club][target_angle]
        pm = pro_data['metrics']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("お手本 (Pro)")
            st.video(pro_data['video_path'])
        with col2:
            st.subheader("あなた (You)")
            
            warning_msg = "体の正面（お腹側）" if target_angle == "Front" else "後方（背中側）"
            st.markdown(f"""
            <div class="info-box">
                撮影のヒント: 正確なスコアを出すため、プロと同じ <strong>「{warning_msg}」</strong> から撮影してください。
            </div>
            """, unsafe_allow_html=True)

            my_file = st.file_uploader("自分の動画をアップロード", type=['mp4', 'mov'])
            my_rotate = st.selectbox("回転オプション", ["なし", "時計回りに90度", "反時計回りに90度"])
            rot_map = {"なし": "なし", "時計回りに90度": "時計回りに90度", "反時計回りに90度": "反時計回りに90度"}
            
            if my_file and st.button("診断開始"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(my_file.read())
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                with st.spinner('現在解析中... AIがスイングを分析しています'):
                    processed_path, df, metrics = analyze_video_advanced(tfile.name, out_path, rot_map[my_rotate])
                    st.session_state['my_processed_video'] = processed_path
                    st.session_state['my_metrics'] = metrics
                st.rerun()

            if st.session_state['my_processed_video']:
                st.video(st.session_state['my_processed_video'])

        if st.session_state['my_metrics']:
            mm = st.session_state['my_metrics']
            m_back = mm['top_frame'] - mm['address_frame']
            m_down = mm['impact_frame'] - mm['top_frame']
            my_tempo = m_back / m_down if m_down > 0 else 0
            
            s_arm, m_arm = generate_advice("Arm", pm['top_arm_angle'], mm['top_arm_angle'])
            s_spine, m_spine = generate_advice("Spine", pm['spine_angle_top'], mm['spine_angle_top'])
            s_knee, m_knee = generate_advice("Knee", pm['knee_angle_top'], mm['knee_angle_top'])
            s_tempo, m_tempo = generate_advice("Tempo", 3.0, my_tempo)
            s_head, m_head = generate_advice("Head", pm['head_stability'], mm['head_stability'])

            total_score = int((s_arm + s_spine + s_knee + s_tempo + s_head) / 5)

            st.markdown("---")
            st.markdown(f"""
            <div class="soft-card">
                <div class="score-label">TOTAL SCORE</div>
                <div class="total-score-val">{total_score}</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            def show_card(col, title, score, msg):
                with col:
                    st.markdown(f"""
                    <div class="soft-card" style="padding: 16px; margin-bottom: 10px;">
                        <div class="score-label">{title}</div>
                        <div class="metric-val">{score}</div>
                        <div class="advice-text">{msg}</div>
                    </div>
                    """, unsafe_allow_html=True)

            show_card(c1, "テンポ", s_tempo, m_tempo)
            show_card(c2, "左腕の伸び", s_arm, m_arm)
            show_card(c3, "頭の固定", s_head, m_head)
            show_card(c4, "前傾維持", s_spine, m_spine)
            show_card(c5, "膝の粘り", s_knee, m_knee)

# PAGE 3: 比較動画 (Sync)
elif app_mode == "3. 比較動画作成":
    st.header("同期動画作成")
    
    if selected_club in st.session_state['club_data'] and st.session_state['my_metrics']:
        available_angles = list(st.session_state['club_data'][selected_club].keys())
        target_angle = st.radio(
            "結合するプロ動画のアングル", 
            available_angles, 
            format_func=lambda x: "体の正面 (Face-on)" if x=="Front" else "後方 (Down-the-line)"
        )
        
        if st.button("比較動画を生成"):
            sync_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            with st.spinner("処理中... トップ位置でタイミングを合わせています"):
                pro_path = st.session_state['club_data'][selected_club][target_angle]['video_path']
                my_path = st.session_state['my_processed_video']
                
                create_sync_video(
                    pro_path, 
                    my_path, 
                    st.session_state['club_data'][selected_club][target_angle]['metrics'],
                    st.session_state['my_metrics'],
                    sync_out
                )
                st.session_state['sync_video_path'] = sync_out
            st.success("完成しました！")
            
        if st.session_state['sync_video_path']:
            st.video(st.session_state['sync_video_path'])
    else:
        st.markdown('<div class="warning-box">解析データが見つかりません。先に「スイング診断」を行ってください。</div>', unsafe_allow_html=True)

# PAGE 4: リアルタイム・コーチ
elif app_mode == "4. リアルタイム・コーチ":
    st.header("リアルタイム・AIコーチ")
    st.write("カメラに向かって構えてください。")

    st.markdown('<div class="warning-box">安全警告：打球の進行方向には絶対に立たないでください。</div>', unsafe_allow_html=True)

    if selected_club not in st.session_state['club_data'] or not st.session_state['club_data'][selected_club]:
         st.markdown('<div class="warning-box">プロ動画が登録されていません。</div>', unsafe_allow_html=True)
    else:
        available_angles = list(st.session_state['club_data'][selected_club].keys())
        target_angle = st.radio(
            "チェックするアングル", 
            available_angles, 
            format_func=lambda x: "体の正面 (Face-on)" if x=="Front" else "後方 (Down-the-line)"
        )
        
        target_metrics = st.session_state['club_data'][selected_club][target_angle]['metrics']
        
        ctx = webrtc_streamer(
            key="realtime-coach", 
            mode=WebRtcMode.SENDRECV, 
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_processor_factory=RealtimeCoach,
            async_processing=True
        )
        
        if ctx.video_processor:
            ctx.video_processor.update_target(target_metrics)
