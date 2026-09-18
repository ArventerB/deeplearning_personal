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

### 🗓️ Ngày 18/09/2026
> **Nội dung:** Giải hệ phương trình tuyến tính bằng Autograd, khảo sát nghiệm vô số, thực hành tổng hợp các thao tác Tensor và khôi phục hệ số đa thức bậc 2 bằng Gradient Descent.  

* **Bài 10: Giải hệ phương trình bằng Autograd**
  * **Giải hệ phương trình 4 ẩn:** $A + B = 9$, $C - D = 1$, $A + C = 8$, $B - D = 2$.
  * **Kiểm tra nghiệm:**
    * Bộ nghiệm tối ưu thu được: $A \approx 4.4299$, $B \approx 4.5701$, $C \approx 3.5701$, $D \approx 2.5701$.
    * Thay vào 4 phương trình đều thỏa mãn với tổng sai số bình phương cực nhỏ ($\approx 5.45 \times 10^{-9}$).
  * **Giải thích vì sao bài toán có vô số nghiệm:**
    * *Về đại số tuyến tính:* Phương trình (4) chính là tổ hợp tuyến tính $\text{PT}(4) = \text{PT}(1) - \text{PT}(3) + \text{PT}(2)$. Hệ thực chất chỉ có 3 phương trình độc lập nhưng có tới 4 ẩn số ($4 > 3$), dẫn đến hệ có 1 bậc tự do và vô số nghiệm theo dạng $(7 - t, 2 + t, 1 + t, t)$.
    * *Về mặt tối ưu hóa:* Đáy của hàm mất mát ($\mathcal{L} = 0$) là một đường thẳng liên tục trong không gian 4 chiều $\mathbb{R}^4$. Các trọng số khởi tạo ngẫu nhiên khác nhau sẽ dẫn thuật toán Gradient Descent trượt về các điểm nghiệm khác nhau dọc theo đường thẳng này.

* **Bài 11: Thực hành tổng hợp Tensor, Autograd & Khôi phục đa thức bậc 2**
  * **Thao tác Reshape Tensor (YC 1):**
    * Khởi tạo Tensor $X$ có 24 phần tử (`torch.arange(24)`).
    * Reshape lần lượt thành các kích thước: $4 \times 6$, $2 \times 12$ và $2 \times 3 \times 4$ dựa trên nguyên tắc tích các chiều bằng 24.
  * **Các phép toán và hàm thống kê trên Tensor 3 x 4 (YC 2):**
    * Khởi tạo hai Tensor $A, B$ kích thước $3 \times 4$.
    * Thực hiện phép cộng $A + B$, nhân từng phần tử $A * B$ (Element-wise), chia từng phần tử $A / B$.
    * Tính trung bình theo cột (`torch.mean(A, dim=0)`) và độ lệch chuẩn theo cột (`torch.std(A, dim=0)`).
  * **Tính đạo hàm với Autograd & Kiểm chứng thủ công (YC 3):**
    * Cho hàm số $y = 2x^2 + 5x + 3$, tính đạo hàm tại $x = 2$.
    * Autograd: `y.backward()` cho kết quả `x.grad = 13.0`.
    * Giải tích: $y' = 4x + 5 \implies y'(2) = 4(2) + 5 = 13$. Kết quả hai cách tính hoàn toàn trùng khớp.
  * **Khôi phục hệ số đa thức $y = 2x^2 - 3x + 5$ bằng Gradient Descent (YC 4):**
    * Tạo dữ liệu mẫu và tối ưu hóa ma trận trọng số $w = [w_1, w_2, w_3]^T$ qua 2000 epochs với thuật toán NAdam.
    * Kết quả hội tụ: $w_1 \approx 2.0020$ (tiệm cận 2), $w_2 \approx -2.9991$ (tiệm cận -3), $w_3 \approx 4.8922$ (tiệm cận 5), khôi phục thành công đa thức gốc.

---

## 📁 Cấu trúc dự án

```text
deeplearning_personal/
├── README.md        # Nhật ký làm việc, ghi chú tiến độ và kiến thức
└── lab02.ipynb      # Notebook thực hành các bài tập Lab 02 (Bài 1 -> Bài 11)
```

## 🛠️ Môi trường & Thư viện
* **Python**: 3.10+ (khuyên dùng)
* **PyTorch**: `torch`
* **NumPy**: `numpy`

