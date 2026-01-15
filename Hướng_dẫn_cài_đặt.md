# Hướng dẫn chi tiết Đồ án: CLIP4Cir-MoE

Bước 1: Downdload source code được cung cấp về máy.
Hướng dẫn sẽ được cung cấp sau đây:
Đồ án này triển khai hệ thống **Truy vấn hình ảnh kết hợp (Composed Image Retrieval - CIR)**. Hệ thống sử dụng các đặc trưng từ mô hình CLIP, kết hợp với một mô tơ **Combiner (mạng Combiner)** nâng cao tích hợp cơ chế **Mixture of Experts (MoE)** để hội tụ đa phương thức một cách linh hoạt.

Hệ thống thực hiện truy vấn hình ảnh mục tiêu dựa trên:
- Một **hình ảnh tham chiếu (reference image)**.
- Một **văn bản mô tả thay đổi (modification text description)**.
---

## 1. Giới thiệu về Đồ án

### Tổng quan
Nhiệm vụ CIR nhằm tìm kiếm các hình ảnh có đặc điểm thị giác tương đồng với ảnh tham chiếu, đồng thời phản ánh được các thay đổi về mặt ngữ nghĩa được mô tả trong văn bản đi kèm.

Đồ án xây dựng dựa trên bộ mã hóa hình ảnh và văn bản của CLIP, tuân theo quy trình huấn luyện **hai giai đoạn**:
1. **Giai đoạn 1: Tinh chỉnh (Fine-tuning) CLIP:** Fine-tune cả hai bộ encoder của CLIP với query là vector kế hợp của đặc trưng ảnh reference và relative caption, target là vector đặc trưng của ảnh đích. Đặc trưng ảnh reference và relative caption được kết hợp qua phép cộng phần tử (element-wise sum) và sử dụng hàm mất mát tương phản (contrastive loss).
2. **Giai đoạn 2: Huấn luyện mạng Combiner:** Đóng băng các bộ mã hóa CLIP đã fine-tuned và huấn luyện mạng Combiner từ đầu để học cách hội tụ biểu diễn đa phương thức với hình ảnh mục tiêu.

### Kiến trúc mạng Combiner (với MoE)
Mạng Combiner bao gồm:
- Các lớp chiếu (Projection layers) cho embedding văn bản và hình ảnh.
- Cơ chế chú ý (Attention).
- Cơ chế kết hợp chuyên gia (MoE).
- Kết nối cổng (Gating) theo từng phần tử và kết nối tắt (Residual connections).
---

## 2. Cài đặt (Installation)
```bash
git clone https://github.com/phuongthanhcute/20210790_VuPhuongThanh_20251.git ## Repo 
cd CLIP4CIR_MoE_ver2
conda create -n clip4cir -y python=3.8 # Tạo môi trường ảo conda python 3.8
conda activate clip4cir # Kích hoạt môi trường ảo conda
conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0 # Cài đặt pytorch 1.11.0 và torchvision 0.12.0
conda install -y -c anaconda pandas=1.4.2 # Cài đặt pandas 1.4.2
pip install comet-ml==3.21.0 # Cài đặt comet-ml (optional - có thể không cài, vì trong quá trình training có thể không sử dụng)
pip install git+https://github.com/openai/CLIP.git # Cài đặt CLIP
```
---

## 3. Chuẩn bị dữ liệu (Data Preparation)
- Dataset **CIRR**: Tải về từ link Google Drive sau đây: https://drive.google.com/file/d/150dBD5iHg9tqnLF1FH4nFiF4A3mYQ1iz/view?usp=drive_link
- Dataset **FashionIQ**: Tải về từ link Google Drive sau đây: https://drive.google.com/file/d/1axKaa3sCKGteMcaNt0BsHJIUjYpegjZh/view?usp=drive_link
- Giải nén 2 file zip đã tải về và cấu trúc lại các datasets theo hướng dẫn sau:
Để mã nguồn hoạt động chính xác, các tập dữ liệu **FashionIQ** và **CIRR** cần được cấu trúc:

```
CLIP4CIR_MoE
└───  fashionIQ_dataset
      └─── captions
            | cap.dress.test.json
            | cap.dress.train.json
            | cap.dress.val.json
            | ... 
            
      └───  images
            | B00006M009.jpg
            | B00006M00B.jpg
            | B00006M6IH.jpg
            | ...
            
      └─── image_splits
            | split.dress.test.json
            | split.dress.train.json
            | split.dress.val.json
            | ...

└───  cirr_dataset  
       └─── train
            └─── 0
                | train-10108-0-img0.png
                | train-10108-0-img1.png
                | ...
            └─── 1
                | train-10056-0-img0.png
                | ...
       └─── dev
            | dev-0-0-img0.png
            | ...
       └─── test1
            | test1-0-0-img0.png
            | ...
       └─── cirr
            └─── captions
                | cap.rc2.test1.json
                | cap.rc2.train.json
                | cap.rc2.val.json
            └─── image_splits
                | split.rc2.test1.json
                | split.rc2.train.json
                | split.rc2.val.json
```

---

## 4. Các bước huấn luyện và đánh giá
### 4.1. Tinh chỉnh CLIP (Stage 1)
Chạy lệnh sau để tinh chỉnh mô hình CLIP trên FashionIQ hoặc CIRR:

```sh
python src/clip_fine_tune.py \
   --dataset {'CIRR' or 'FashionIQ'} \ # Chọn dataset CIRR hoặc FashionIQ để thực hiện finetune CLIP
   --num-epochs 100 \
   --clip-model-name RN50x16 \
   --encoder both \
   --learning-rate 2e-6 \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1
```

### 4.2. Huấn luyện mạng Combiner (Stage 2)
Sau khi tinh chỉnh CLIP, chạy lệnh này để huấn luyện mạng Combiner:

**Với dataset CIRR**
```sh
python src/combiner_train.py \
   --dataset CIRR \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --num-epochs 300 \
   --clip-model-name RN50x16 \
   --clip-model-path clip_finetuned\ 
   --combiner-lr 2e-5 \
   --batch-size 4096 \
   --clip-bs 32 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1
```

**Với dataset FashionIQ**
```sh
python src/combiner_train.py \
   --dataset FashionIQ \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --num-epochs 300 \
   --clip-model-name RN50x16 \
   --clip-model-path models_final/clip_finetuned \ 
   --combiner-lr 2e-5 \
   --batch-size 2048 \
   --clip-bs 32 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1
```
### 4.3. Đánh giá (Validation)
Tính toán các chỉ số trên server Test của tập dữ liệu CIRR và trên tập kiểm định (validation set) của tập dữ liệu FashionIQ:

**Với dataset CIRR**: Tạo file submission để nộp lên hệ thống đánh giá của tập dữ liệu CIRR:
```sh
python src/cirr_test_submission.py \
   --submission-name my_submission \
   --combining-function combiner \
   --combiner-path models_final/combiner_trained_on_cirr \ 
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --clip-model-name RN50x16 \
   --clip-model-path models_final/clip_finetuned_on_cirr \ 
   --target-ratio 1.25 \
   --transform targetpad
```
Nộp file submission của dataset **CIRR** lên server sau để đánh giá kết quả: https://cirr.cecs.anu.edu.au/test_process/

**Với dataset FashionIQ**: Đánh giá trên tập kiểm định (validation set) của tập dữ liệu FashionIQ.
```sh
python src/validate.py \
   --dataset FashionIQ \
   --combining-function combiner \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --clip-model-name RN50x16 \
   --target-ratio 1.25 \
   --transform targetpad \
   --combiner-path models_final/moe_combiner_trained_on_fiq \ 
   --clip-model-path models_final/clip_finetuned_on_fiq \ 
```
## 5. Đường dẫn tới các checkpoint đã huấn luyện (với mạng Combiner MoE)
- Tất cả các checkpoint đã huấn luyện được lưu trữ trong link Google Drive sau đây: https://drive.google.com/file/d/1nE-azEls_RZTQXuuXxSlB3s6b6J5ecFO/view?usp=drive_link
- Tải về tất cả các checkpoint và giải nén vào thư mục models_final
- Các lệnh ở mục 4: cần thay thế đường dẫn các model thành các đường dẫn tới các checkpoint đã huấn luyện.