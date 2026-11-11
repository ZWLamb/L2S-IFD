# L2S-IFD: Instance Feature Borrowing Distillation for Dense Instance Segmentation

**CVPR 2027**

> First Author, Second Author, Third Author

Official implementation of **"From Large Instances to Small: Instance Feature Borrowing Distillation for Dense Instance Segmentation"**.

L2S-IFD lets each small instance borrow feature content from the teacher's large instances of the same category, forming an enriched distillation target. The teacher and memory are used **only during training** — inference incurs **zero overhead**.

---

## Highlights

- **Instance-level borrowing**: small instances query a lightweight memory of teacher large-instance features via category-masked cross-instance attention.
- **Enriched distillation target**: `f̃_i^T = f_i^T + γ · c_i`, with residual connection and γ warm-up ramp.
- **No inference overhead**: teacher, memory, and projections are discarded at test time.
- **Plug-and-play**: complementary to spatial-weighting methods (e.g., SARD) and standard feature distillation.

---

## Installation

```bash
git clone https://github.com/<your-org>/L2S-IFD.git
cd L2S-IFD
pip install -r requirements.txt
