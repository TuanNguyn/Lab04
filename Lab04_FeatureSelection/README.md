# House Price Project - Lab04: Feature Selection

Dự án dự đoán giá nhà Ames: tiền xử lý, baseline modeling và đánh giá các kỹ thuật Feature Selection (Filter, Wrapper, Embedded) với SVR, RandomForest, XGBoost.

## Môi trường
```powershell
python -m pip install -r requirements.txt
```

## Cấu trúc
```text
data/raw/train.csv            dữ liệu gốc
data/processed/               X/y train-test, feature_names, baseline_results, feature_selection_results
models/                       preprocessor.pkl, baseline_best_model.pkl
src/preprocess.py             làm sạch + tạo feature
src/evaluate.py               CV 5-fold + đánh giá test
src/feature_selection.py      Pipeline selector + mô hình
figures/                      biểu đồ xuất từ notebook 03
```

## Thứ tự chạy (chạy từ thư mục gốc, "Restart & Run All")
1. `01_preprocessing.ipynb`
2. `02_baseline_modeling.ipynb`
3. `03_feature_selection.ipynb` (mất khoảng 10-15 phút trên máy 1 nhân)

## Quy ước
- Target `log_SalePrice = log1p(SalePrice)`; `random_state=42`; 5-fold CV chỉ trên train.
- Selector nằm trong `Pipeline` để fit lại trong từng fold (không leakage); test chỉ dùng để báo cáo.
- Cột `technique`: `baseline`, `filter_kbest`, `filter_mi`, `wrapper_rfe`, `embedded_lasso`, `embedded_rf`.
