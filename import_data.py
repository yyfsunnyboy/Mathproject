import csv
import re
from app import app, db, Skill, SkillDependency

def slugify(text):
    """ 簡單將中文轉為英文 ID """
    text = text.lower().replace(' ', '_').replace('(', '').replace(')', '')
    # 移除所有非英文、數字、底線的字元
    text = re.sub(r'\W+', '', text.replace('_', 'TEMP_UNDERSCORE')).replace('TEMP_UNDERSCORE', '_')
    return text or 'skill'

def import_skills_and_dependencies():
    print("開始匯入資料...")

    # 您的 CSV 檔案名稱
    csv_filename = '多項式知識點鏈結.xlsx - 工作表1.csv'

    # 儲存所有技能的字典 { "中文名稱": SkillObject }
    skill_map = {}
    # 儲存所有依賴關係的列表 [ (prerequisite_name, target_name) ]
    dependencies = []
    # 儲存所有不重複的技能名稱
    skill_names = set()

    # === 步驟 1: 讀取 CSV 並找出所有不重複的技能名稱 ===
    try:
        with open(csv_filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # 清理前後空白
                prereq_name = row['來源節點 (先備知識)'].strip()
                target_name = row['目標節點 (學習目標)'].strip()

                if prereq_name:
                    skill_names.add(prereq_name)
                if target_name:
                    skill_names.add(target_name)

                if prereq_name and target_name:
                    dependencies.append((prereq_name, target_name))

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{csv_filename}'")
        print("請確認檔案名稱是否完全正確，並且和 import_data.py 在同一個資料夾。")
        return
    except Exception as e:
        print(f"讀取 CSV 時發生錯誤: {e}")
        return

    print(f"從 CSV 中找到了 {len(skill_names)} 個不重複的單元。")

    # === 步驟 2: 將所有技能存入資料庫 ===
    for name in skill_names:
        if not name:
            continue

        # 檢查是否已存在
        existing_skill = Skill.query.filter_by(display_name=name).first()
        if not existing_skill:
            # 建立新的 Skill 物件
            new_skill = Skill(
                display_name=name,
                name=slugify(name), # 自動產生英文 ID
                description=f"關於「{name}」的練習" # 預設描述
            )

            # 自動分類
            if '國中' in name:
                new_skill.school_type = '共同'
                new_skill.grade_level = '國中'
            else:
                new_skill.school_type = '普高'
                new_skill.grade_level = '高一' # 預設為高一，您之後可以手動改

            db.session.add(new_skill)
            print(f"  [建立 Skill]: {name}")
            skill_map[name] = new_skill # 存入 map (雖然還沒有 ID)
        else:
            print(f"  [Skill 已存在]: {name}")
            skill_map[name] = existing_skill # 從資料庫取得

    try:
        db.session.commit()
        print("=== 所有 Skill 已成功存入資料庫 (並取得 ID) ===")
    except Exception as e:
        db.session.rollback()
        print(f"存入 Skill 時發生錯誤: {e}")
        return

    # === 步驟 3: 建立依賴關係 ===
    # 經過 commit 之後，skill_map 裡的所有物件現在都有 ID 了
    # (或者我們重新查詢一次以確保安全)

    print("正在建立技能依賴關係...")
    dependency_count = 0
    for prereq_name, target_name in dependencies:
        # 從資料庫中取得剛剛建立的 Skill (包含 ID)
        prereq_skill = Skill.query.filter_by(display_name=prereq_name).first()
        target_skill = Skill.query.filter_by(display_name=target_name).first()

        if prereq_skill and target_skill:
            # 檢查依賴關係是否已存在
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
    # 確保在 app context 中執行，這樣才能操作資料庫
    with app.app_context():
        # 為了安全，我們先刪除舊資料，避免重複
        print("正在清空舊的 Skills 和 Dependencies...")
        try:
            # 注意：這會刪除所有技能和依賴關係，但不會刪除 User 和 Progress
            db.session.query(SkillDependency).delete()
            db.session.query(Skill).delete()
            db.session.commit()
            print("舊資料已清空。")
        except Exception as e:
            db.session.rollback()
            print(f"清空舊資料時出錯: {e}")

        # 執行匯入
        import_skills_and_dependencies()