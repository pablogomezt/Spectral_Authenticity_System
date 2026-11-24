# Hyperspectral datasets

The source ZIP contained 19 Git LFS pointers. The real files were recovered from `pablogomezt/SA` and verified against every SHA-256 identifier in the ZIP. They are stored with Git LFS here.

```bash
git lfs install
git lfs pull
```

GitHub ZIP downloads may contain pointers. Clone with LFS; actual MATLAB files must not be text beginning with `version https://git-lfs.github.com/spec/v1`.

## Main loader mapping

| CLI dataset | Cube file / MATLAB key | Label file / MATLAB key |
| :--- | :--- | :--- |
| `salinas` | `Salinas.mat` / `salinas_corrected` | `Salinas_gt.mat` / `salinas_gt` |
| `indian_pines` | `Indian_pines.mat` / `indian_pines_corrected` | `Indian_pines_gt.mat` / `indian_pines_gt` |
| `pavia` | `Pavia.mat` / `pavia` | `Pavia_gt.mat` / `pavia_gt` |
| `circular` | `circular.mat` / `circular` | `circular_gt.mat` / `circular_gt` |
| `cuadrada` | `cuadrada.mat` / `cuadrada` | `cuadrada_gt.mat` / `cuadrada_gt` |

Cubes use H × W × bands and labels H × W. The loader converts cubes to float32 and divides by their global maximum; Circular labels are shifted by -1. Variable names are dataset-specific.

`indian_pines_crop` and `multi` exist in the loader but lack wavelength settings in the entry point. Additional laboratory files are retained as source assets; MATLAB v7.3 files need an HDF5 loader and are not readable by the current SciPy loader.

## Verified inventory

Dimensions were inspected directly from MATLAB metadata (HDF5 variable names for v7.3).

| File | Variables and dimensions |
| :--- | :--- |
| `Indian_pines.mat` | `indian_pines_corrected` (145, 145, 200) |
| `Indian_pines_gt.mat` | `indian_pines_gt` (145, 145) |
| `Pavia.mat` | `pavia` (1096, 715, 102) |
| `Pavia_gt.mat` | `pavia_gt` (1096, 715) |
| `SCENA_CIRCULAR_ICLR2.mat` | MATLAB v7.3 (HDF5): `wavelengths`, `y`; requires an HDF5 loader |
| `SCENA_Cuadrada_madera_ICLR2.mat` | MATLAB v7.3 (HDF5): `wavelengths`, `y`; requires an HDF5 loader |
| `Salinas.mat` | `salinas_corrected` (512, 217, 204) |
| `Salinas_gt.mat` | `salinas_gt` (512, 217) |
| `circular.mat` | `circular` (512, 512, 31) |
| `circular_gt.mat` | `circular_gt` (512, 512) |
| `class_matrix_Escena_CIRCULAR1.mat` | `class` (1038, 1388) |
| `class_matrix_Escena_COLOR1.mat` | `class` (1038, 1038) |
| `class_matrix_Escena_madera1.mat` | `class` (1038, 1388) |
| `cropped_class_image.mat` | `cropped_class_image` (60, 60) |
| `cropped_cube.mat` | `cropped_cube` (60, 60, 200) |
| `cuadrada.mat` | `cuadrada` (512, 512, 31) |
| `cuadrada_gt.mat` | `cuadrada_gt` (512, 512) |
| `lab_scena.mat` | `scena` (768, 1024, 31) |
| `pavia_multispectral.mat` | `multispectral_cube` (1096, 715, 10) |

## Attribution and reuse

Files are preserved from the supplied archive and its LFS objects. No dataset license or complete provenance statement was supplied; consult the authors and original providers for reuse terms.
