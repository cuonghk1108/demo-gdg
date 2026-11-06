# demo-gdg

## Đề tài: C1. DECISION TREE VÀ ENSEMBLE: NGHIÊN CỨU CƠ SỞ LÝ THUYẾT, THỰC NGHIỆM ĐỐI CHỨNG VÀ GIẢI THÍCH MÔ HÌNH TRÊN DỮ LIỆU BẢNG

**Đơn vị thực hiện:** Nhóm 4 – Phân ban Data & AI (Google Developer Group on Campus HUST - Gen 6)

### Thành viên nhóm:
1. **Phan Hoàng Long** (*Team Lead*) – Nghiên cứu & Hiện thực hóa Decision Tree from Scratch (NumPy).
2. **Võ Thảo Nguyên** – Tiền xử lý dữ liệu, Bagging / Random Forest, AdaBoost / GBM, Giải thích SHAP.
3. **Võ Xuân Cường** – Hệ GBDT hiện đại (XGBoost, LightGBM, CatBoost), Thực nghiệm Stress-Test & Kiến trúc Web Demo Streamlit.
4. **Trần Đại Nghĩa** – Cơ sở lý thuyết toán học, Đối chứng Non-Tree Models & Tổng hợp Báo cáo.
- **Mentor hướng dẫn:** Nguyễn Quốc Thái.

---

### Hướng dẫn cài đặt & Khởi chạy Web Demo

```bash
# 1. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 2. Khởi chạy ứng dụng Streamlit
streamlit run app.py
```

Truy cập tại: `http://localhost:8501`

### Các module & Tab chức năng:
- **Tab 1: 🛡️ Anti-Leakage Pipeline & Data Explorer**: Trực quan hóa chống rò rỉ nhãn (Label Leakage) và phân phối 989 tin thuê phòng trọ TP.HCM.
- **Tab 2: 🌳 CART Scratch vs Sklearn**: So sánh Decision Tree tự viết bằng NumPy thuần với Scikit-learn.
- **Tab 3: 📊 Ensemble Benchmark (14 Models)**: Đối chứng 14 mô hình (Scratch, Sklearn Tree, Bagging, Random Forest, Extra Trees, AdaBoost, GBM, HistGBM, XGBoost, LightGBM, CatBoost, Ridge, SVR, MLP).
- **Tab 4: 🧪 Stress-Test Lab (4D)**: Thực nghiệm 4 chiều (Nhiễu & Ngoại lai, Thiếu mẫu N<=100, Mã hóa Phân loại Target Encoding vs OHE, Giới hạn Ngoại suy).
- **Tab 5: 💡 Định giá Phòng trọ & SHAP Cục bộ**: Consensus dự đoán giá phòng trọ theo thời gian thực + Khoảng tin cậy 90% (Random Forest) + SHAP Waterfall Chart.
- **Tab 6: 🌐 Global XAI & Decision Matrix**: Ma trận quyết định chọn mô hình thực tế và tương quan giữa các đặc trưng.
