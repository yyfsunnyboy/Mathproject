import google.generativeai as genai
import os

# 讀取您在終端機設定的環境變數
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("錯誤：請先在終端機設定 GEMINI_API_KEY 環境變數！")
else:
    try:
        genai.configure(api_key=API_KEY)

        print("正在查詢您帳戶可用的模型...\n")

        found_models = False
        # 迴圈列出所有模型
        for m in genai.list_models():
            # 我們只關心支援 'generateContent' (聊天) 的模型
            if 'generateContent' in m.supported_generation_methods:
                print(f"找到可用模型: {m.name}")
                found_models = True

        if not found_models:
            print("找不到任何支援 'generateContent' 的模型。")

    except Exception as e:
        print(f"查詢失敗，發生錯誤：{e}")
        