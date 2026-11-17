# Khóa luận tốt nghiệp: Ảnh hưởng của tiền xử lý dữ liệu đến bài toán phân loại và phân cụm

Khóa luận ngành Khoa học Dữ liệu, HUFLIT (05/2025 – 08/2025, nhóm 2 người). Hai thí nghiệm so sánh các pipeline
tiền xử lý (chọn đặc trưng, chuẩn hóa, cân bằng lớp, loại ngoại lai, giảm chiều) với scikit-learn:

1. **Phân loại** 2.708 bài báo khoa học của bộ Cora vào 7 chủ đề (SVM tuyến tính).
2. **Phân cụm** 98.913 người dùng một sàn thời trang C2C theo hành vi (K-Means).

## Kết quả chính

### Phân loại Cora

Chia train/test (80/20, phân tầng) trước mọi bước tiền xử lý; tập test 542 bài giữ nguyên phân phối gốc.
Chọn đặc trưng, chuẩn hóa và oversampling nằm trong `imblearn.Pipeline`, chỉ fit trên tập train.
Chi tiết: [`Cora_Classification_v2.ipynb`](Cora_Classification_v2.ipynb).

| Pipeline | Accuracy | F1-macro |
|---|---|---|
| Baseline: LinearSVC, không tiền xử lý | 0.753 | 0.733 |
| **VarianceThreshold → SelectKBest(chi², k=572) → Normalizer → RandomOverSampler → LinearSVC** | **0.784** | **0.776** |
| Cùng pipeline, bỏ oversampling | 0.775 | 0.751 |
| VarianceThreshold → Normalizer → RandomOverSampler → SVC(linear) | 0.777 | 0.759 |
| VarianceThreshold → StandardScaler → RandomOverSampler → LinearSVC | 0.749 | 0.738 |

Tiền xử lý tốt nhất tăng F1-macro thêm **4,3 điểm**. Normalizer (chuẩn hóa theo từng bài báo) hợp với dữ liệu
từ nhị phân hơn StandardScaler; oversampling góp khoảng 2,5 điểm nhờ cải thiện recall của các lớp nhỏ.

### Phân cụm hành vi người dùng

Chi tiết: [`users_behavior_Clustering_FINAL.ipynb`](users_behavior_Clustering_FINAL.ipynb).

| Bước | Kết quả |
|---|---|
| Isolation Forest loại ngoại lai | 98.913 → 90.202 người dùng (loại 8.711) |
| QuantileTransformer (phân phối chuẩn) | giảm độ lệch của các cột đếm (skewness ban đầu tới hơn 200) |
| One-Hot Encoding, chọn đặc trưng theo phương sai | dữ liệu số hoàn toàn cho K-Means |
| StandardScaler + PCA | 10 thành phần giữ 95,3% phương sai |
| K-Means, k = 2…9 | chọn **k = 4**: Silhouette 0,585 (cao nhất), Davies–Bouldin 0,636, Calinski–Harabasz 29.249 |
| Độ ổn định (50 lần chia train/test) | chênh lệch Silhouette train–test trung bình 0,0002 (CI 95%: −0,006 đến 0,010) |

Silhouette trên dữ liệu thô (0,769) không so trực tiếp được với con số trên: hai chỉ số tính trên hai không gian
đặc trưng khác nhau, và khi chưa chuẩn hóa thì khoảng cách bị vài cột có giá trị lớn chi phối.

## Cấu trúc repo

```text
.
├── Cora_Classification_v2.ipynb           # phân loại, quy trình đánh giá đã sửa (khuyên đọc)
├── Cora_Classification_FINAL.ipynb        # phân loại, bản nộp khóa luận (giữ nguyên để đối chiếu)
├── users_behavior_Clustering_FINAL.ipynb  # phân cụm
├── Báo cáo_FINAL.docx                     # báo cáo khóa luận
└── requirements.txt
```

## Cách chạy

```bash
pip install -r requirements.txt
jupyter notebook
```

- `Cora_Classification_v2.ipynb` tự tải `cora.content` vào `data/cora/` nếu chưa có (khoảng 15 phút chạy trên máy 2 nhân).
- `users_behavior_Clustering_FINAL.ipynb` cần file CSV từ Kaggle (xem bên dưới), đặt tại `data/user_dataset.csv`;
  trên Google Colab thì đọc từ Google Drive như bản gốc.

## Dữ liệu

- **Cora:** 2.708 bài báo, vector nhị phân 1.433 từ, 7 chủ đề. Bản dùng ở đây lấy từ
  [tkipf/pygcn](https://github.com/tkipf/pygcn/tree/master/data/cora) (gốc của LINQS).
- **Người dùng sàn C2C:** [E-commerce Users of a French C2C Fashion Store (Kaggle)](https://www.kaggle.com/datasets/jmmvutu/ecommerce-users-of-a-french-c2c-fashion-store),
  98.913 người dùng, 24 cột (mạng xã hội, sản phẩm đăng bán/đã bán/đã mua, ứng dụng di động, thâm niên...).

## Công nghệ

Python, pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, Seaborn, Jupyter (Kaggle, Google Colab).
