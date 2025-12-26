import os
import json
import base64
import time
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import re

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# =========================================================
# 基本設定與常數
# =========================================================
st.set_page_config(
    page_title="NutriTag AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 台灣每日參考值 (Daily Values)
DAILY_VALUES = {
    'calories': 2000,
    'protein': 60,
    'fat': 60,
    'saturatedFat': 18,
    'carbs': 300,
    'sodium': 2000,
    'transFat': None, # 未訂定
    'sugar': None     # 未訂定
}

# 欄位對照表 (用於顯示與處理)
COLUMN_CONFIG = {
    'productName': '產品名稱',
    'servingSize': '每一份量(g)',
    'calories': '熱量',
    'protein': '蛋白質',
    'fat': '脂肪',
    'saturatedFat': '飽和脂肪',
    'transFat': '反式脂肪',
    'carbs': '碳水化合物',
    'sugar': '糖',
    'sodium': '鈉',
    'ratio': '新比例(%)'
}

# 數值型欄位清單
NUMERIC_FIELDS = ['servingSize', 'calories', 'protein', 'fat', 'saturatedFat', 'transFat', 'carbs', 'sugar', 'sodium', 'ratio']

# =========================================================
# 環境變數與初始化
# =========================================================
load_dotenv()
'''
# Firebase 初始化 (使用 st.cache_resource 避免重複初始化)
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            cred_path = os.getenv("FIREBASE_CREDENTIAL_PATH")
            if not cred_path or not os.path.exists(cred_path):
                st.error("找不到 Firebase 金鑰檔案，請檢查 .env 設定")
                return None
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase 連線失敗: {e}")
        return None
'''
# Firebase 初始化（支援 Streamlit Cloud）
@st.cache_resource
def init_firebase():
    try:
        # 避免重複初始化
        if not firebase_admin._apps:

            # 讀取 secrets.toml 內的 firebase 欄位
            firebase_config = dict(st.secrets["firebase"])

            # Streamlit 會把 private_key 轉成單行字串，需還原換行
            if "private_key" in firebase_config:
                firebase_config["private_key"] = firebase_config["private_key"].replace("\\n", "\n")

            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)

        return firestore.client()

    except Exception as e:
        st.error(f"Firebase 初始化失敗：{e}")
        return None

db = init_firebase()

# 取得使用者 ID (這裡為了簡化，模擬一個固定的 User ID，實作時可改為 Session ID 或登入機制)
if 'user_id' not in st.session_state:
    st.session_state.user_id = "streamlit_user_v1"

USER_ID = st.session_state.user_id

# =========================================================
# Azure OpenAI 呼叫函數
# =========================================================
def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def call_azure_ai(image_base64, system_prompt):
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_API_VERSION") # 記得讀取這個變數
    
    if not all([endpoint, api_key, deployment, api_version]):
        st.error("Azure API 設定不完整，請檢查環境變數 (.env)。")
        return None

    # 1. 修正 URL 結構 (移除 endpoint 尾端的斜線以防重複，然後組裝完整路徑)
    base_url = endpoint.rstrip("/")
    url = f"{base_url}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    # 2. 設定 Header
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    # 3. 設定 Payload (針對 GPT-4o Vision 的格式)
    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "請分析這張圖片中的營養標示或配方。" # 這裡可以根據您的需求調整提示詞
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000, # 建議設定回傳上限
        "temperature": 0.0  # 分析數據建議設為 0 以求精準
    }

    try:
        # 4. 發送請求
        response = requests.post(url, headers=headers, json=payload)
        
        # 檢查是否有 HTTP 錯誤
        response.raise_for_status()
        
        # 回傳 JSON 結果
        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"API 請求失敗: {e}")
        # 如果有詳細錯誤訊息，印出來除錯
        if 'response' in locals():
            st.write(response.text)
        return None

# =========================================================
# 新增 Helper 函數：解析 AI 回傳的 JSON 內容
# =========================================================
def extract_json_content(api_response):
    """
    從 Azure API 的原始回應中提取 content，並清洗 markdown 標記，轉為 Python Dict
    """
    if not api_response or 'choices' not in api_response:
        return None
    
    try:
        # 1. 取得文字內容 (content)
        raw_content = api_response['choices'][0]['message']['content']
        
        # 2. 移除 Markdown 標記 (```json ... ```)
        cleaned_content = re.sub(r"```json\s*", "", raw_content)
        cleaned_content = re.sub(r"```\s*", "", cleaned_content)
        
        # 3. 移除前後空白
        cleaned_content = cleaned_content.strip()
        
        # 4. 轉成 JSON/Dict
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失敗: {e}")
        print(f"原始內容: {raw_content}") # 除錯用
        return None
    except Exception as e:
        print(f"未預期的錯誤: {e}")
        return None

def generate_nutrition_label_html(final_100g, input_serving_size, show_dv, daily_values):
    """
    生成三欄式營養標示 HTML (項目 | 每份 | 每100g或DV%)
    """
    
    # 定義 CSS：新增了 .col-name 和 .col-val 來控制欄位寬度
    style = """
<style>
    .nutrition-box {
        border: 2px solid #000;
        padding: 20px;
        width: 100%;
        max-width: 450px; /* 限制最大寬度，避免在寬螢幕太開 */
        background-color: #ffffff;
        color: #000000;
        font-family: "Microsoft JhengHei", sans-serif;
        line-height: 1.5;
        margin: 0 auto;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .nutrition-title {
        font-size: 22px;
        font-weight: 900;
        border-bottom: 3px solid #000;
        padding-bottom: 5px;
        margin-bottom: 10px;
        text-align: center;
    }
    .nutrition-meta {
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 2px;
    }
    .nutrition-thick-line {
        border-bottom: 3px solid #000;
        margin: 8px 0;
    }
    .nutrition-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #ccc;
        padding: 6px 0;
        font-size: 15px;
    }
    .nutrition-row:last-child {
        border-bottom: none;
    }
    /* 欄位寬度設定 */
    .col-name {
        flex: 1; /* 佔據剩餘空間 */
        text-align: left;
        font-weight: bold;
    }
    .col-val {
        width: 100px; /* 固定寬度，確保對齊 */
        text-align: right;
        white-space: nowrap;
    }
    .indent {
        padding-left: 20px;
        font-weight: normal;
    }
    .header-row {
        font-weight: 900;
        border-bottom: 2px solid #000;
        align-items: flex-end; /* 讓標題文字靠下對齊 */
    }
</style>
"""

    # 欄位定義
    nutrients_map = [
        ('calories', '熱量', '大卡', False),
        ('protein', '蛋白質', '公克', False),
        ('fat', '脂肪', '公克', False),
        ('saturatedFat', '飽和脂肪', '公克', True),
        ('transFat', '反式脂肪', '公克', True),
        ('carbs', '碳水化合物', '公克', False),
        ('sugar', '糖', '公克', True),
        ('sodium', '鈉', '毫克', False),
    ]

    # 設定右側標題
    col3_header = "每 100 公克" if not show_dv else "每日參考值百分比"
    
    # 建立標頭資訊
    rows_html = f"""
<div class="nutrition-meta">每一份量 {input_serving_size} 公克</div>
<div class="nutrition-meta">本包裝含 1 份</div>

<div class="nutrition-row header-row">
    <span class="col-name">項目</span>
    <span class="col-val">每份</span>
    <span class="col-val">{col3_header}</span>
</div>
"""
    
    # 建立數據行
    for key, label, unit, is_indent in nutrients_map:
        val_100g = final_100g.get(key, 0)
        
        # 1. 計算中間欄位：每份數值
        # 公式：每100g數值 * (份量 / 100)
        val_serving = val_100g * (input_serving_size / 100)
        
        # 格式化每份顯示字串
        str_serving = f"{val_serving:.1f} {unit}"

        # 2. 計算右側欄位：每100g 或 DV%
        if not show_dv:
            # 模式 A: 顯示每 100g
            str_right = f"{val_100g:.1f} {unit}"
        else:
            # 模式 B: 顯示每日參考值百分比
            dv_std = daily_values.get(key)
            if dv_std:
                # DV% = (每份數值 / 每日參考值) * 100
                dv_pct = (val_serving / dv_std) * 100
                str_right = f"{dv_pct:.1f} %"
            else:
                str_right = "*"

        # 設定縮排樣式
        indent_class = "indent" if is_indent else ""
        
        # 3. 組合三欄式 HTML
        rows_html += f"""
<div class="nutrition-row">
    <span class="col-name {indent_class}">{label}</span>
    <span class="col-val">{str_serving}</span>
    <span class="col-val">{str_right}</span>
</div>
"""

    # 4. 組裝最終 HTML
    final_html = f"""
{style}
<div class="nutrition-box">
    <div class="nutrition-title">營養標示</div>
    {rows_html}
</div>
"""
    
    return final_html
# =========================================================
# 資料庫操作函數
# =========================================================
def fetch_nutrition_data():
    if not db: return []
    try:
        docs = db.collection(f'artifacts/nutritag-app/users/{USER_ID}/nutritionLabels').stream()
        data = []
        for doc in docs:
            item = doc.to_dict()
            item['id'] = doc.id
            data.append(item)
        return data
    except Exception as e:
        st.error(f"讀取資料失敗: {e}")
        return []

def save_nutrition_item(data):
    if not db: return
    try:
        # 使用時間戳記當作 ID
        doc_id = f"{int(time.time())}_{data.get('productName', 'unknown')}"
        ref = db.collection(f'artifacts/nutritag-app/users/{USER_ID}/nutritionLabels').document(doc_id)
        # 補上預設欄位
        data['userId'] = USER_ID
        data['updatedAt'] = firestore.SERVER_TIMESTAMP
        data['ratio'] = 0.0 # 預設比例
        ref.set(data)
    except Exception as e:
        st.error(f"儲存失敗: {e}")

def update_nutrition_batch(df):
    if not db: return
    batch = db.batch()
    collection_ref = db.collection(f'artifacts/nutritag-app/users/{USER_ID}/nutritionLabels')
    
    try:
        for index, row in df.iterrows():
            doc_ref = collection_ref.document(row['id'])
            update_data = {
                'productName': row['productName'],
                'ratio': float(row['ratio'])
            }
            # 更新其他數值欄位
            for field in NUMERIC_FIELDS:
                if field != 'ratio':
                    update_data[field] = float(row[field])
            
            batch.update(doc_ref, update_data)
        
        batch.commit()
        st.toast("資料與比例已更新！", icon="✅")
    except Exception as e:
        st.error(f"批次更新失敗: {e}")

def delete_nutrition_item(doc_id):
    if not db: return
    try:
        db.collection(f'artifacts/nutritag-app/users/{USER_ID}/nutritionLabels').document(doc_id).delete()
        st.rerun()
    except Exception as e:
        st.error(f"刪除失敗: {e}")

# =========================================================
# UI 頁面佈局
# =========================================================

st.title("NutriTag AI 營養標示助手")
st.markdown("使用 AI 進行圖片分析，自動計算新產品的營養標示。")

# --- 步驟一：配方比例分析 ---
with st.expander("步驟一：配方比例分析 (選填)", expanded=False):
    st.info("上傳您的配方表照片，系統將自動擷取材料重量並換算成 100% 的比例。")
    recipe_file = st.file_uploader("上傳配方表", type=['jpg', 'png', 'jpeg'], key="recipe_upload")
    
    if recipe_file:
        if st.button("開始分析配方"):
            with st.spinner("正在分析配方結構..."):
                base64_img = encode_image(recipe_file)
                # 使用原本 HTML 中的 System Prompt
                prompt = """
                你是一位專業的食品配方分析助手。請分析圖片中的配方表，執行以下步驟：

                1. 辨識所有食材名稱與其重量（將單位統一換算為公克 g）。
                2. 計算所有食材的總重量。
                3. 計算每個食材佔總重量的百分比 (Percentage)，保留小數點後 1 位。
                4. 忽略沒有數值或無法辨識的項目。

                請**嚴格**只回傳以下 JSON 格式，不要包含任何 Markdown 標記或額外文字：
                {
                "ingredients": [
                    { "name": "材料名稱", "weight": 數值(float), "percentage": 數值(float) }
                ]
                }
                """
                # result = call_azure_ai(base64_img, prompt)
                
                # if result and 'ingredients' in result:
                #     ingredients = result['ingredients']
                #     df_recipe = pd.DataFrame(ingredients)
                    
                #     # 顯示結果
                #     st.dataframe(df_recipe, column_config={
                #         "name": "材料名稱",
                #         "weight": "重量 (g)",
                #         "percentage": "佔比 (%)"
                #     }, hide_index=True, use_container_width=True)
                    
                #     total_weight = sum(item['weight'] for item in ingredients)
                #     st.metric("總重量", f"{total_weight} g")
                #     st.caption("* 請參考上方的「佔比」填寫到步驟三的表格中。")
                # else:
                #     st.error("無法辨識配方內容，請確認圖片清晰度。")
                # --- 修正開始 ---
                # 使用解析函式處理原始回應

                # 呼叫 API
                raw_response = call_azure_ai(base64_img, prompt)

                result_data = extract_json_content(raw_response)
                
                if result_data and 'ingredients' in result_data:
                    ingredients = result_data['ingredients'] # 從解析後的資料拿
                    df_recipe = pd.DataFrame(ingredients)
                    
                    # 顯示結果
                    st.dataframe(df_recipe, column_config={
                        "name": "材料名稱",
                        "weight": "重量 (g)",
                        "percentage": "佔比 (%)"
                    }, hide_index=True, use_container_width=True)
                    
                    total_weight = sum(item['weight'] for item in ingredients)
                    st.metric("總重量", f"{total_weight} g")
                    st.caption("* 請參考上方的「佔比」填寫到步驟三的表格中。")
                else:
                    st.error("無法辨識配方內容，或 AI 回傳格式錯誤。")
                    # 開發階段建議印出原始回應除錯
                    # st.write(raw_response)

# --- 步驟二：上傳原料標示 ---
st.header("步驟二：上傳原料營養標示")
uploaded_files = st.file_uploader("支援多檔上傳 (JPG, PNG)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if st.button(f"開始分析 {len(uploaded_files)} 張圖片"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"正在處理: {file.name} ...")
            
            base64_img = encode_image(file)
            # 使用原本 HTML 中的 System Prompt，強調優先抓取 100g
            prompt = """
            你是一位營養標示分析專家。請分析圖片中的營養標示表格，規則如下：

            1. **優先順序**：請優先擷取「每 100 公克 (Per 100g)」或「每 100 毫升」的數值。如果沒有，才擷取「每份 (Per Serving)」的數值。
            2. **數值處理**：只擷取數字，**必須移除**所有單位（如 g, mg, kcal, % 等）。若數值為 "反式脂肪 0.0公克"，則 output 0。
            3. **欄位對應**：
            - calories: 熱量/大卡
            - protein: 蛋白質
            - fat: 脂肪 (總脂肪)
            - saturatedFat: 飽和脂肪
            - transFat: 反式脂肪
            - carbs: 碳水化合物 (總碳水)
            - sugar: 糖 (總糖)
            - sodium: 鈉

            若找不到該欄位，請填 0。

            請**嚴格**只回傳以下 JSON 格式，不要包含 Markdown (` ```json `)：
            {
            "productName": "產品名稱(若無則回傳空字串)",
            "servingSize": 數值(代表每份幾公克，若未標示請填 100),
            "calories": 數值,
            "protein": 數值,
            "fat": 數值,
            "saturatedFat": 數值,
            "transFat": 數值,
            "carbs": 數值,
            "sugar": 數值,
            "sodium": 數值
            }
            """

            # 呼叫 API
            raw_response = call_azure_ai(base64_img, prompt)

            # 1. 解析回應
            nutrition_data = extract_json_content(raw_response)
            
            if nutrition_data:
                # 2. 若 AI 沒抓到產品名，使用檔名
                if not nutrition_data.get('productName'):
                    nutrition_data['productName'] = file.name.split('.')[0]
                
                # 3. 儲存解析後的乾淨資料 (Dict)
                save_nutrition_item(nutrition_data)
            else:
                st.error(f"檔案 {file.name} 解析失敗")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("處理完成！請在下方表格確認資料。")
        time.sleep(1)
        st.rerun()

# --- 步驟三：資料編輯與計算 ---
st.header("步驟三：編輯資料與設定新比例")

raw_data = fetch_nutrition_data()

if not raw_data:
    st.info("目前沒有資料，請先在步驟二上傳圖片。")
else:
    # 轉換為 DataFrame 進行編輯
    df = pd.DataFrame(raw_data)
    
    # 確保欄位順序與類型
    cols_order = ['id', 'productName'] + NUMERIC_FIELDS
    # 若資料庫缺欄位補 0
    for col in cols_order:
        if col not in df.columns:
            df[col] = 0.0 if col in NUMERIC_FIELDS else ""
            
    df = df[cols_order]

    # 設定 data_editor
    st.markdown("""
    1. 直接點擊表格修改數值 (若是 AI 辨識錯誤)。
    2. 在 **「新比例(%)」** 欄位輸入配方百分比。
    3. 修改完畢後，請務必點擊 **「儲存變更並重新計算」**。
    """)
    
    edited_df = st.data_editor(
        df,
        column_config={
            "id": None, # 隱藏 ID
            **COLUMN_CONFIG
        },
        disabled=["id"], # ID 不可編輯
        hide_index=True,
        use_container_width=True,
        key="editor"
    )

    # 計算目前比例總和
    current_ratio_sum = edited_df['ratio'].sum()
    
    col_act1, col_act2 = st.columns([1, 1])
    with col_act1:
        if abs(current_ratio_sum - 100) > 1:
            st.warning(f"⚠️ 目前比例總和為 {current_ratio_sum:.1f}% (目標應為 100%)")
        else:
            st.success(f"✅ 目前比例總和為 {current_ratio_sum:.1f}%")
            
    with col_act2:
        if st.button("儲存變更並重新計算", type="primary"):
            update_nutrition_batch(edited_df)
            st.rerun()

    # 刪除功能 (需手動選擇)
    with st.expander("進階管理：刪除項目"):
        delete_options = {f"{row['productName']} (ID: {row['id']})": row['id'] for index, row in df.iterrows()}
        selected_to_delete = st.selectbox("選擇要刪除的項目", list(delete_options.keys()))
        if st.button("確認刪除"):
            delete_nutrition_item(delete_options[selected_to_delete])

    # --- 步驟四：計算結果 ---
    st.divider()
    st.header("步驟四：計算結果 - 最新營養標示")

    # 計算加權平均
    # 邏輯： sum(每項100g數值 * (比例/100))
    final_100g = {k: 0.0 for k in NUMERIC_FIELDS if k not in ['servingSize', 'ratio']}
    
    # 僅計算比例 > 0 的項目
    valid_items = edited_df[edited_df['ratio'] > 0]
    total_ratio_weight = valid_items['ratio'].sum()

    if total_ratio_weight > 0:
        for idx, row in valid_items.iterrows():
            weight_factor = row['ratio'] / 100
            for field in final_100g.keys():
                final_100g[field] += row[field] * weight_factor
        
        # 正規化：如果比例總和不是 100 (例如 99.5)，還是要還原成 100g 的數值
        # 這裡假設使用者輸入的是最終產品的配方比例，理論上總和應接近 100
        # 如果總和不是 100，我們將其視為這就是全部的組成，不需要額外除以 total_ratio_weight/100
        # 因為 "每100g" 的定義就是配方總和為 100 時的數值。
        pass
    



    # UI 控制項
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        input_serving_size = st.number_input("每一份量 (公克)", value=100.0, step=10.0)
    with col_res2:
        show_dv = st.checkbox("顯示「每日參考值百分比」")

    # 顯示營養標示 (模擬 HTML 樣式)
    def format_num(val):
        return f"{val:.1f}"
    
    # 準備顯示數據
    display_rows = []
    nutrients_map = [
        ('calories', '熱量', ''),
        ('protein', '蛋白質', '公克'),
        ('fat', '脂肪', '公克'),
        ('saturatedFat', '飽和脂肪', '公克'),
        ('transFat', '反式脂肪', '公克'),
        ('carbs', '碳水化合物', '公克'),
        ('sugar', '糖', '公克'),
        ('sodium', '鈉', '毫克'),
    ]

    #     st.markdown("### 營養標示")
        
    #     # 使用 Markdown 表格呈現
    #     table_md = f"""
    # | 項目 | 每份 | {'每 100 公克' if not show_dv else '每日參考值百分比'} |
    # | :--- | :--- | :--- |
    # | **每一份量** | **{input_serving_size} 公克** | |
    # | **本包裝含** | **1 份** | |
    # """
        
    #     for key, label, unit in nutrients_map:
    #         val_100g = final_100g[key]
    #         val_serving = val_100g * (input_serving_size / 100)
            
    #         col2_text = ""
    #         if not show_dv:
    #             col2_text = f"{format_num(val_100g)} {unit}"
    #         else:
    #             dv_std = DAILY_VALUES.get(key)
    #             if dv_std:
    #                 # DV% = (每份數值 / 每日參考值) * 100
    #                 dv_pct = (val_serving / dv_std) * 100
    #                 col2_text = f"{format_num(dv_pct)} %"
    #             else:
    #                 col2_text = "*"
            
    #         # 縮排顯示 (飽和脂肪, 反式脂肪, 糖)
    #         prefix = "&nbsp;&nbsp;&nbsp;&nbsp;" if key in ['saturatedFat', 'transFat', 'sugar'] else ""
    #         row_label = f"{prefix}{label}"
            
    #         table_md += f"| {row_label} | {format_num(val_serving)} {unit} | {col2_text} |\n"

    #     st.markdown(table_md)


    # # ... (上面計算完 final_100g 之後) ...

    # st.markdown("### 預覽結果")

    # # 呼叫我們剛剛寫的 HTML 生成函數
    # # 注意：這裡將 DAILY_VALUES 傳進去
    # html_label = generate_nutrition_label_html(final_100g, input_serving_size, show_dv, DAILY_VALUES)
    
    # # 使用 unsafe_allow_html=True 來渲染 HTML
    # st.markdown(html_label, unsafe_allow_html=True)

    # # ... (下面接著是儲存按鈕) ...
    
    # 1. 呼叫函式產生 HTML 字串
    html_label = generate_nutrition_label_html(final_100g, input_serving_size, show_dv, DAILY_VALUES)

    # 2. 顯示標題
    st.markdown("### 預覽結果")

    # 3. 關鍵修正：必須加上 unsafe_allow_html=True
    # st.write(html_label, unsafe_allow_html=True)
    st.markdown(html_label, unsafe_allow_html=True)

    if show_dv:
        st.caption("＊參考值未訂定")
        st.caption("每日參考值：熱量 2000 大卡、蛋白質 60 公克、脂肪 60 公克、飽和脂肪 18 公克、碳水化合物 300 公克、鈉 2000 毫克。")

    # 除錯訊息 (可選)
    with st.expander("查看詳細計算數據 (100g 基礎值)"):
        st.json(final_100g)