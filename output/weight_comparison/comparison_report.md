# Edge Weight Comparison Analysis

Comparison of three edge weighting schemes:
- **w_1**: Uniform weighting (w_e = 1)
- **w_n**: Linear weighting (w_e = n)
- **w_n2**: Quadratic weighting (w_e = n²) ← Current default

---

## NetzHast

| Metric | w_1 | w_n | w_n2 |
|--------|-----|-----|------|
| Initial Error | 0.14 | 4.18 | 130.36 |
| Final Error | 0.03 | 0.41 | 6.70 |
| Error Reduction | 77.3% | 90.2% | 94.9% |
| Mean |Deviation| | 0.0230 | 0.0225 | 0.0207 |
| Max |Deviation| | 0.0962 | 0.0598 | 0.0331 |

### Largest Edge Performance

| Weight | n_e | Target | Achieved | Deviation |
|--------|-----|--------|----------|----------|
| w_1 | 39 | 0.8016 | 0.8978 | +0.0962 |
| w_n | 39 | 0.8016 | 0.8614 | +0.0598 |
| w_n2 | 39 | 0.8016 | 0.8277 | +0.0261 |

---

## demoVerbundnetz_SimulationNeu

| Metric | w_1 | w_n | w_n2 |
|--------|-----|-----|------|
| Initial Error | 45.67 | 15800.50 | 7660035.15 |
| Final Error | 4.91 | 197.49 | 5827.67 |
| Error Reduction | 89.2% | 98.8% | 99.9% |
| Mean |Deviation| | 0.0674 | 0.0664 | 0.0719 |
| Max |Deviation| | 0.2662 | 0.3808 | 0.5416 |

### Largest Edge Performance

| Weight | n_e | Target | Achieved | Deviation |
|--------|-----|--------|----------|----------|
| w_1 | 692 | 0.4759 | 0.6509 | +0.1750 |
| w_n | 692 | 0.4759 | 0.5254 | +0.0495 |
| w_n2 | 692 | 0.4759 | 0.4777 | +0.0019 |

---

## test3

| Metric | w_1 | w_n | w_n2 |
|--------|-----|-----|------|
| Initial Error | 18.52 | 1299.00 | 95896.70 |
| Final Error | 0.99 | 11.14 | 108.79 |
| Error Reduction | 94.6% | 99.1% | 99.9% |
| Mean |Deviation| | 0.0410 | 0.0175 | 0.0083 |
| Max |Deviation| | 0.0762 | 0.0558 | 0.0347 |

### Largest Edge Performance

| Weight | n_e | Target | Achieved | Deviation |
|--------|-----|--------|----------|----------|
| w_1 | 85 | 0.6200 | 0.6962 | +0.0762 |
| w_n | 85 | 0.6200 | 0.6460 | +0.0260 |
| w_n2 | 85 | 0.6200 | 0.6163 | -0.0037 |

---

## Summary

### Key Findings

**NetzHast**:
- Best performance: **w_n2** (mean deviation: 0.0207)

**demoVerbundnetz_SimulationNeu**:
- Best performance: **w_n** (mean deviation: 0.0664)

**test3**:
- Best performance: **w_n2** (mean deviation: 0.0083)

