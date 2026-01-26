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
| Final Error | 0.14 | 4.18 | 6.70 |
| Error Reduction | 0.0% | 0.0% | 94.9% |
| Mean |Deviation| | 0.1179 | 0.1179 | 0.0207 |
| Max |Deviation| | 0.1984 | 0.1984 | 0.0331 |

### Largest Edge Performance

| Weight | n_e | Target | Achieved | Deviation |
|--------|-----|--------|----------|----------|
| w_1 | 39 | 0.8016 | 1.0000 | +0.1984 |
| w_n | 39 | 0.8016 | 1.0000 | +0.1984 |
| w_n2 | 39 | 0.8016 | 0.8277 | +0.0261 |

---

## demoVerbundnetz_SimulationNeu

| Metric | w_1 | w_n | w_n2 |
|--------|-----|-----|------|
| Initial Error | 45.67 | 15800.50 | 7660035.15 |
| Final Error | 45.67 | 5827.67 | 5827.67 |
| Error Reduction | 0.0% | 63.1% | 99.9% |
| Mean |Deviation| | 0.2456 | 0.0719 | 0.0719 |
| Max |Deviation| | 0.5241 | 0.5416 | 0.5416 |

### Largest Edge Performance

| Weight | n_e | Target | Achieved | Deviation |
|--------|-----|--------|----------|----------|
| w_1 | 692 | 0.4759 | 1.0000 | +0.5241 |
| w_n | 692 | 0.4759 | 0.4777 | +0.0019 |
| w_n2 | 692 | 0.4759 | 0.4777 | +0.0019 |

---

## test3

| Metric | w_1 | w_n | w_n2 |
|--------|-----|-----|------|
| Initial Error | 18.52 | 1299.00 | 95896.70 |
| Final Error | 18.52 | 108.79 | 108.79 |
| Error Reduction | 0.0% | 91.6% | 99.9% |
| Mean |Deviation| | 0.2855 | 0.0083 | 0.0083 |
| Max |Deviation| | 0.3800 | 0.0347 | 0.0347 |

### Largest Edge Performance

| Weight | n_e | Target | Achieved | Deviation |
|--------|-----|--------|----------|----------|
| w_1 | 85 | 0.6200 | 1.0000 | +0.3800 |
| w_n | 85 | 0.6200 | 0.6163 | -0.0037 |
| w_n2 | 85 | 0.6200 | 0.6163 | -0.0037 |

---

## Summary

### Key Findings

**NetzHast**:
- Best performance: **w_n2** (mean deviation: 0.0207)

**demoVerbundnetz_SimulationNeu**:
- Best performance: **w_n** (mean deviation: 0.0719)

**test3**:
- Best performance: **w_n** (mean deviation: 0.0083)

