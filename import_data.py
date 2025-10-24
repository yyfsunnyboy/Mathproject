import pandas as pd  # <--- 使用 pandas
import re
import os
from app import app, db, Skill, SkillDependency

def slugify(text):
    """ 簡單將中文轉為英文 ID """
    if not isinstance(text, str):
        return 'skill' # 如果是空值 (NaN)
    text = text.lower().replace(' ', '_').replace('(', '').replace(')', '')
    text = re.sub(r'\W+', '', text.replace('_', 'TEMP_UNDERSCORE')).replace('TEMP_UNDERSCORE', '_')
    return text or 'skill'

def import_skills_and_dependencies():
    print("開始匯入資料 (使用 Pandas)...")
    
    # --- vvv ---
    # 檔案路徑設定
    basedir = os.path.abspath(os.path.dirname(__file__))
    excel_folder = '知識點鏈結' # 您的子資料夾名稱
    excel_file = '多項式知識點鏈結.xlsx' # 您的 Excel 檔案名稱
    excel_filename = os.path.join(basedir, excel_folder, excel_file)
    
    # 工作表名稱 (從您的 .csv 檔名推測)
    sheet_name = '工作表1' 
    # --- ^^^ ---
    
    print(f"正在嘗試讀取 Excel: {excel_filename}")
    print(f"正在嘗試讀取工作表: {sheet_name}")

    # 儲存所有技能的字典 { "中文名稱": SkillObject }
    skill_map = {}
    # 儲存所有依賴關係的列表 [ (prerequisite_name, target_name) ]
    dependencies = []
    # 儲存所有不重複的技能名稱
    skill_names = set()

    # === 步驟 1: 使用 Pandas 讀取 Excel 並找出所有技能 ===
    try:
        # 讀取 Excel 檔案
        df = pd.read_excel(excel_filename, sheet_name=sheet_name)
        
        # 取得關鍵欄位
        prereq_col = '來源節點 (先備知識)'
        target_col = '目標節點 (學習目標)'
        
        if prereq_col not in df.columns or target_col not in df.columns:
            print(f"錯誤：Excel 中找不到 '{prereq_col}' 或 '{target_col}' 欄位。")
            print(f"請檢查 Excel 欄位名稱是否完全正確。")
            return
            
        # 迭代 Excel 的每一行
        for index, row in df.iterrows():
            # .strip() 是為了去除前後空白
            prereq_name = str(row[prereq_col]).strip() if pd.notna(row[prereq_col]) else None
            target_name = str(row[target_col]).strip() if pd.notna(row[target_col]) else None
            
            if prereq_name and prereq_name != 'nan':
                skill_names.add(prereq_name)
            if target_name and target_name != 'nan':
                skill_names.add(target_name)
            
            if prereq_name and target_name and prereq_name != 'nan' and target_name != 'nan':
                dependencies.append((prereq_name, target_name))
    
    except FileNotFoundError:
        print(f"錯誤：在以下路徑找不到檔案 '{excel_filename}'")
        print(f"請確認 '{excel_folder}' 資料夾存在，且內含 '{excel_file}' 檔案。")
        return
    except Exception as e:
        # 可能是 '工作表1' 名稱錯誤
        if "No sheet named" in str(e):
            print(f"錯誤：在 Excel 中找不到名為 '{sheet_name}' 的工作表。")
            print("請檢查 Excel 中的工作表名稱是否完全正確。")
        else:
            print(f"讀取 Excel 時發生錯誤: {e}")
        return

    print(f"從 Excel 中找到了 {len(skill_names)} 個不重複的單元。")

    # === 步驟 2: 將所有技能存入資料庫 ===
    # ( ... 這一段程式碼和之前完全一樣 ... )
    for name in skill_names:
        if not name:
            continue
        existing_skill = Skill.query.filter_by(display_name=name).first()
        if not existing_skill:
            new_skill = Skill(
                display_name=name,
                name=slugify(name), 
                description=f"關於「{name}」的練習"
            )
            if '國中' in name:
                new_skill.school_type = '共同'
                new_skill.grade_level = '國中'
            else:
                new_skill.school_type = '普高'
                new_skill.grade_level = '高一'
            db.session.add(new_skill)
            print(f"  [建立 Skill]: {name}")
            skill_map[name] = new_skill
        else:
            print(f"  [Skill 已存在]: {name}")
            skill_map[name] = existing_skill
    try:
        db.session.commit()
        print("=== 所有 Skill 已成功存入資料庫 (並取得 ID) ===")
    except Exception as e:
        db.session.rollback()
        print(f"存入 Skill 時發生錯誤: {e}")
        return
        
    # === 步驟 3: 建立依賴關係 ===
    # ( ... 這一段程式碼和之前完全一樣 ... )
    print("正在建立技能依賴關係...")
    dependency_count = 0
    for prereq_name, target_name in dependencies:
        prereq_skill = Skill.query.filter_by(display_name=prereq_name).first()
        target_skill = Skill.query.filter_by(display_name=target_name).first()
        if prereq_skill and target_skill:
            existing_dep = SkillDependency.query.filter_by(
                prerequisite_id=prereq_skill.id,
                target_id=target_skill.id
            ).first()
            if not existing_dep:
                new_dep = SkillDependency(
                    prerequisite_id=prereq_skill.id,
                    target_id=target_skill.id
                )
                db.session.add(new_dep)
                dependency_count += 1
    try:
        db.session.commit()
        print(f"=== 成功！建立了 {dependency_count} 條新的依賴關係 ===")
    except Exception as e:
        db.session.rollback()
        print(f"存入依賴關係時發生錯誤: {e}")
        return

# --- 主程式 ---
if __name__ == "__main__":
    with app.app_context():
        # ... 清空舊資料的程式碼 (完全一樣) ...
        print("正在清空舊的 Skills 和 Dependencies...")
        try:
            db.session.query(SkillDependency).delete()
            db.session.query(Skill).delete()
            db.session.commit()
            print("舊資料已清空。")
        except Exception as e:
            db.session.rollback()
            print(f"清空舊資料時出錯: {e}")

        import_skills_and_dependencies()