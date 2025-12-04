# PCQA Codebase Guide

## Project Overview

PCQA (Point Cloud Quality Assessment) is a robotic point cloud quality assessment framework that evaluates point cloud quality using n-jet fitting and statistical analysis. The system operates in two main stages: 1) fitting geometric surfaces to point cloud patches using DeepFit, and 2) computing quality scores by analyzing how points deviate from fitted surfaces.

## Core Architecture & Data Flow

### Three-Stage Pipeline

1. **N-Jet Surface Fitting** (`n-jet_fitting.py`)
   - Loads `.xyz` point cloud files and extracts PCA-aligned patches (256 points default)
   - Uses DeepFit with order-3 polynomial fitting to estimate surface normals and beta coefficients
   - Outputs: `{filename}_order3_normal_beta.txt` containing points, normals, and polynomial coefficients

2. **Quality Score Computation** (`pcqa_demo.py`)
   - Maps each scanned point to a 2D distribution based on distance and angular metrics
   - Generates synthetic points from fitted surface parameters using the Vandermonde matrix
   - Compares scanned vs. synthetic distributions using grid-based coverage analysis
   - Outputs: `{filename}_quality_score_ours.txt` with quality scores per point

3. **Visualization** (`quality2color.py`)
   - Converts quality scores to color values using rainbow colormap
   - Outputs: `{filename}_quality_score_colored.txt` for CloudCompare visualization

### Key Dependencies & Their Roles

- **DeepFit** (`models/DeepFit.py`): Implements weighted n-jet polynomial fitting using Vandermonde matrices and Cholesky decomposition
- **scipy.spatial.cKDTree**: Efficient nearest-neighbor search for patch extraction (avoid scikit-learn's KDTree for performance)
- **torch**: Batch processing for surface fitting (CPU-only by default)
- **open3d**: Mesh operations and geometric processing
- **pygeodesic/gdist**: Geodesic distance calculations on mesh surfaces

## Critical Algorithms & Patterns

### Patch Processing Pattern
```python
# All patch operations follow this PCA-centering pattern:
patch_points = torch.from_numpy(points[indices, :])
patch_points = patch_points - torch.from_numpy(points[query_idx, :])  # Center on query point
patch_points, trans = pca_points(patch_points)  # Align to principal axes
```
Used in both `tutorial_utils.py` (fitting) and `pcqa_utils.py` (quality scoring).

### 2D Mapping for Quality Assessment
Quality scores derive from analyzing scanned vs. synthetic point distributions in 2D parameter space:
- Compute convex hull of 2D mapped points
- Divide into grid cells (gridnum_test_0 = 15 for 256-point patches)
- Calculate coverage metrics and mean squared error
- Flag outliers where mean distance > 1.5 × average spacing

### Surface Fitting Validation
```python
# Check if fit is degenerate before using:
mean_distance = np.mean(nearestDis)
if mean_distance > 1.5 * alpha:  # alpha = average point spacing
    skipIndices.append(i)  # Skip poor fits
```

## Important Conventions

### Parameter Naming
- `jet_order`: Polynomial order (default=3, valid: 1-4) determines beta coefficient count (order 3 = 10 coefficients)
- `points_per_patch` / `searchK`: Neighborhood size for fitting (default=256)
- `skip_number`: Grid sampling frequency in `pcqa_demo.py` (smaller = slower, higher precision)
- `thread_number`: CPU threads for parallel quality computation

### File Format Conventions
- Input: `.xyz` files with whitespace-separated x, y, z coordinates
- Intermediate: `{name}_order{n}_normal_beta.txt` → columns: [x, y, z, nx, ny, nz, beta_0...beta_k]
- Output: `{name}_quality_score_ours.txt` → columns: [x, y, z, quality_metric]
- Preprocessed: Shapes normalized to unit sphere: `(points - mean) / (0.5 * bbdiag)`

### Threading Model
Quality computation uses `ThreadPCQA` with `ThreadPool` for per-point processing. Each thread processes a disjoint subset of indices and reports progress via tqdm. **Note**: Single-threaded debugging recommended; set `-tn 1` for reproducibility.

## Development Workflow

### Running the Pipeline
```bash
# Stage 1: Surface fitting (outputs normal_beta.txt)
python n-jet_fitting.py

# Stage 2: Quality assessment with 14 threads, grid skip=8
python pcqa_demo.py -tn 14 -sn 8

# Stage 3: Visualization preparation
python quality2color.py
```

### Testing & Debugging
- **Jupyter workflow**: `QualityScoreEstimation.ipynb` mirrors `pcqa_demo.py` with visualization
- **Small samples**: Test on `sample.xyz` (faster) before bunny model
- **Intermediate outputs**: Save checkpoint files like `tmpBug.xyz` for mesh inspection

### Common Pitfalls
- **Torch dtype mismatches**: Beta coefficients and points must be same dtype (float32/64)
- **Numerical stability**: DeepFit adds noise to XtX if Cholesky fails (see line 78 in DeepFit.py)
- **Memory leaks in loops**: Clear intermediate tensors when processing large batches
- **KDTree reuse**: Build once in preprocessing, don't rebuild per-point

## Utility Module Organization

| Module | Purpose |
|--------|---------|
| `pcqa_utils.py` | Core PCQA functions: 2D mapping, quality metrics, mesh distances |
| `tutorial_utils.py` | Dataset loading and PCA normalization for fitting stage |
| `normal_estimation_utils.py` | GMM-based normal estimation (rarely used in main pipeline) |
| `visualization.py` | 3D plotting helpers for development |
| `provider.py` | Point cloud augmentation (rotation, jitter, occlusion) |

## Integration Points & Extension Areas

### Adding New Fitting Models
- Implement `fit_Wjet` compatible interface in `models/{NewModel}.py`
- Must return: `(beta, normals, neighbor_normals)` tensors
- Update `n-jet_fitting.py` line ~35 to instantiate new model

### Custom Quality Metrics
- Core computation in `pcqa_demo.py` `local_pcqa()` function (lines 44-120)
- Key metrics: `averageDisPoints()` (spacing), `calculateDistancesMesh()` (geodesic), convex hull coverage
- Modify grid parameters (`gridnum_test_*`) for different precision/speed tradeoffs

### Parallel Scaling
- Current: CPU ThreadPool with per-point granularity
- Consider: Batch-level parallelization or GPU transfer for fitting stage if scaling beyond 1M points
