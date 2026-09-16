# Deep Learning - Nhật ký thực hành cá nhân (Personal Workspace)

Kho lưu trữ (repository) này dùng để ghi lại toàn bộ quá trình học tập, thực hành và giải các bài tập môn Học sâu (Deep Learning), trọng tâm là làm việc với **PyTorch**, xử lý **Tensor**, cơ chế tính đạo hàm tự động **Autograd** và thuật toán tối ưu **Gradient Descent**.

---

## 📅 Lịch sử làm việc & Tiến độ học tập (Work Log)

### 🗓️ Ngày 16/09/2026
> **Nội dung:** Khởi tạo repository, làm quen với Tensor và các thao tác định dạng cơ bản trên PyTorch.  
> **Commit:** `Hoan thanh 5 bai ngay 16/9/26` (`8d5c06c`)

* **Bài 1: Tensor cơ bản**
  * Khởi tạo tensor từ danh sách (list) với kiểu dữ liệu chỉ định (`torch.int32`).
  * Kiểm tra các thuộc tính nền tảng: kích thước (`.shape`), số chiều (`.ndim`), kiểu dữ liệu (`.dtype`).
* **Bài 2: Tạo các Tensor thường dùng**
  * Tìm hiểu và phân biệt `torch.rand()` (phân phối đều trong khoảng [0, 1]) và `torch.randn()` (phân phối chuẩn với trung bình 0, phương sai 1, có cả số âm và dương).
  * Khởi tạo tensor đặc biệt: `torch.zeros()`, `torch.ones()`, `torch.eye()` (ma trận đơn vị), `torch.full()` (điền giá trị cố định).
* **Bài 3: Kiểm tra thông tin Tensor**
  * Hiểu rõ ý nghĩa kích thước đa chiều (ví dụ: `torch.Size([2, 3, 4])` gồm 2 khối/ma trận, mỗi ma trận có 3 hàng và 4 cột).
  * So sánh thuộc tính `.shape` và phương thức `.size()`.
* **Bài 4: Slicing và Indexing**
  * Trích xuất lát cắt ma trận với chỉ mục đơn lẻ: `a[1]` (lấy ma trận thứ 2, giảm số chiều từ 3D xuống 2D).
  * Cắt lát đa chiều kết hợp: `a[1:, 2:4]`.
  * Trích xuất các phần tử cuối cùng trên một trục cụ thể bằng chỉ mục âm: `a[:, :, -2:]` (hoặc `a[..., -2:]`).
* **Bài 5: Reshape và Flatten**
  * Trải phẳng tensor nhiều chiều thành vector 1D với `ravel()`.
  * Biến đổi hình dạng tensor bằng `reshape()` (từ 3x4 sang 3x2x2 và 2x6).
  * Kiểm tra và kiểm chứng nguyên tắc bảo toàn số lượng phần tử trước và sau khi reshape bằng hàm `.numel()`.

---

### 🗓️ Ngày 17/09/2026
> **Nội dung:** Thực hiện các phép toán đại số tuyến tính, thống kê dữ liệu, cơ chế Autograd và hoàn thành trọn vẹn thuật toán Gradient Descent.  
> **Commits:** 
> * `461a4c2`: `Hoan thanh bai 8 va tham khao bai 9 ngay 17/9/26`
> * `b92ffb8`: `hoan thanh loi giai bai 9 ngay 17/9`

* **Bài 6: Các phép toán trên Tensor & Nhân ma trận**
  * Phân biệt rõ sự khác nhau giữa:
    * `a * b`: Phép nhân từng phần tử (Element-wise / Hadamard product).
    * `a @ b.T`: Phép nhân ma trận đại số tuyến tính (Matrix multiplication), yêu cầu ma trận thứ hai phải chuyển vị `.T` để thỏa mãn quy tắc nhân ma trận.
  * Phân tích và kiểm tra shape kết quả của `a @ b.T` là `torch.Size([2, 2])`.
* **Bài 7: Các hàm thống kê trên Tensor (So sánh `dim=0` và `dim=1`)**
  * Khảo sát các hàm: `torch.mean()`, `torch.std()`, `torch.cumsum()`.
  * Rút ra quy luật:
    * `dim=0`: Thao tác theo chiều dọc (duyệt qua các hàng để tính kết quả cho từng cột).
    * `dim=1`: Thao tác theo chiều ngang (duyệt qua các cột để tính kết quả cho từng hàng).
* **Bài 8: Đạo hàm tự động với Autograd cơ bản**
  * Khởi tạo biến theo dõi gradient với `requires_grad=True`.
  * Sử dụng lệnh `.backward()` để lan truyền ngược tính đạo hàm.
  * So sánh kiểm chứng giữa tính toán đạo hàm giải tích bằng tay và giá trị `x.grad` tự động của PyTorch trên:
    * Hàm số $y = x^2$ tại $x = 3.6$ ($y' = 2x = 7.2$).
    * Hàm đa thức $y = 3x^2 + 2x + 1$ tại $x = 3.6$ ($y' = 6x + 2 = 23.6$).
* **Bài 9: Gradient Descent tìm hệ số đa thức**
  * **Mục tiêu:** Tìm lại bộ 3 hệ số `[1, 2, 3]` của đa thức mục tiêu $y = 1x^2 + 2x + 3$.
  * **Kết quả học:**
    * Hệ số khởi tạo ban đầu (ngẫu nhiên): `[-0.2870, 0.1926, 0.9336]`
    * Hệ số học được sau 1000 epochs: `[1.0026, 2.0069, 2.8255]`
    * **Nhận xét:** Các hệ số đã hội tụ cực kỳ sát với giá trị thực tế `[1, 2, 3]` ($1.0026 \approx 1$, $2.0069 \approx 2$, $2.8255 \approx 3$).
  * **Vai trò bộ ba câu lệnh huấn luyện:**
    * `optimizer.zero_grad()`: Xóa sạch gradient cũ về 0 ở đầu mỗi epoch, ngăn ngừa hiện tượng cộng dồn gradient sai lệch trong PyTorch.
    * `mse.backward()`: Lan truyền ngược (Backpropagation) để tính toán đạo hàm riêng của hàm mất mát theo từng trọng số $w$ và ghi nhận vào `w.grad`.
    * `optimizer.step()`: Cập nhật giá trị trọng số $w$ theo hướng ngược chiều gradient dựa trên thuật toán tối ưu NAdam để giảm thiểu sai số.

---

## 📁 Cấu trúc dự án

```text
deeplearning_personal/
├── README.md        # Nhật ký làm việc, ghi chú tiến độ và kiến thức
└── lab02.ipynb      # Notebook thực hành các bài tập Lab 02 (Bài 1 -> Bài 9)
```

## 🛠️ Môi trường & Thư viện
* **Python**: 3.10+ (khuyên dùng)
* **PyTorch**: `torch`
* **NumPy**: `numpy`
