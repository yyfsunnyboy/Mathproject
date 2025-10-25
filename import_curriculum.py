import pandas as pd  # <--- 使用 pandas
import re
import os
from app import app, db, Skill, SkillDependency # 匯入 app 和所有模型

def slugify(text):
    """ 簡單將中文轉為英文 ID """
    if not isinstance(text, str):
        return 'skill' # 如果是空值 (NaN)
    text = text.lower().replace(' ', '_').replace('(', '').replace(')', '')
    text = re.sub(r'\W+', '', text.replace('_', 'TEMP_UNDERSCORE')).replace('TEMP_UNDERSCORE', '_')
    return text or 'skill'

def import_curriculum():
    print("開始匯入新課綱資料 (使用 Pandas 讀取 XLSX)...")
    
    # --- 檔案路徑設定 ---
    basedir = os.path.abspath(os.path.dirname(__file__))
    excel_folder = '知識點鏈結' # 您的子資料夾名稱
    excel_file = '課綱.xlsx' # <--- 修改：使用 .xlsx 檔案
    excel_filename = os.path.join(basedir, excel_folder, excel_file)
    
    sheet_name = '工作表1' # <--- 指定 Excel 中的工作表名稱
    
    print(f"正在嘗試讀取 Excel: {excel_filename}")
    print(f"正在嘗試讀取工作表: {sheet_name}")

    # === 步驟 1: 使用 Pandas 讀取 Excel ===
    try:
        # <--- 修改：使用 pd.read_excel ---
        df = pd.read_excel(excel_filename, sheet_name=sheet_name)
        
        # 關鍵步驟：填補 Excel 合併儲存格造成的空值
        df['年級'].ffill(inplace=True)
        df['大單元'].ffill(inplace=True)
        
        # 移除「小單元」欄位為空的無效資料
        df.dropna(subset=['小單元'], inplace=True)
            
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{excel_filename}'")
        print(f"請確認 '{excel_folder}' 資料夾中存在 '{excel_file}' 檔案。")
        return
    except Exception as e:
        if "No sheet named" in str(e):
             print(f"錯誤：在 Excel 檔案中找不到名為 '{sheet_name}' 的工作表。")
             print("請打開您的 Excel 檔，確認工作表名稱是否完全一致。")
        else:
            print(f"讀取 Excel 時發生錯誤: {e}")
        return

    print(f"從 Excel 中讀取到 {len(df)} 筆「小單元」資料。")

    # === 步驟 2: 清空並重建資料庫 ===
    # (這一段和前一個教學完全相同)
    print("正在清空並重建資料庫表格 (包含新的 main_unit 欄位)...")
    try:
        db.drop_all() # 刪除所有舊表格
        db.create_all() # 根據 app.py (含 main_unit) 建立新表格
        print("資料庫表格已成功重建。")
    except Exception as e:
        print(f"重建資料庫時發生錯誤: {e}")
        return

    # === 步驟 3: 將所有「小單元」作為 Skill 存入資料庫 ===
    # (這一段和前一個教學完全相同)
    print("正在匯入所有「小單元」...")
    for index, row in df.iterrows():
        try:
            new_skill = Skill(
                display_name = row['小單元'].strip(),
                name = slugify(row['小單元']), # 自動產生英文 ID
                description = row['內容'].strip() if pd.notna(row['內容']) else "...", # 處理內容為空的情況
                grade_level = row['年級'].strip(),
                main_unit = row['大單元'].strip()
            )
            db.session.add(new_skill)
        except Exception as e:
            print(f"處理行 {index} ( {row['小單元']} ) 時出錯: {e}")

    try:
        db.session.commit()
        print(f"=== 成功！ {len(df)} 筆「小單元」已全部匯入 Skill 資料表 ===")
    except Exception as e:
        db.session.rollback()
        print(f"存入 Skill 時發生錯誤: {e}")
        return

# --- 主程式 ---
if __name__ == "__main__":
    # 確保在 app context 中執行，這樣才能操作資料庫
    with app.app_context():
        import_curriculum()