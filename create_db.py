from app import app, db

# 這會載入 app.py 裡的設定
with app.app_context():
    print("正在刪除舊資料庫 (如果存在的話)...")
    # 為了安全起見，我們先 drop 掉舊的 (雖然您已經刪了)
    db.drop_all() 

    print("正在根據 app.py 裡的新模型建立資料庫...")
    # 根據您在 app.py 裡修改好的模型 (包含 school_type) 建立新表格
    db.create_all()

    print("="*30)
    print("成功！ 新的 kumon_math.db 資料庫已建立！")
    print("="*30)