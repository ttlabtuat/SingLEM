# Raw benchmark datasets

Raw EEG recordings are not distributed in this repository. This directory
contains zero-byte placeholders only for raw data files that the preprocessing
code reads directly. Obtain each dataset from its official source and replace
the empty files at the same paths:

```text
raw_datasets/
  ATTEN_28/
    VP001/
    ...
  Dreyer_MI_25/
    DATA B/
  WBCIC_MI_23/
    sourcedata/
```

Keep these canonical folder names unchanged. Do not paste downloaded datasets
in a way that creates sibling folders such as `ATTEN_28 copy`,
`Dreyer_MI_25 copy`, or `WBCIC_MI_23 copy`. Merge the downloaded contents into
the existing placeholder folders so real files replace the zero-byte files at
the same relative paths. The preprocessing scripts read only the canonical
paths shown above.

The six configured tasks are derived from three source datasets:

- ATTEN n-back, DSR, and word-generation tasks: [Scientific Data 2018](https://doi.org/10.1038/sdata.2018.3)
- Dreyer motor imagery: [Scientific Data 2023](https://doi.org/10.1038/s41597-023-02445-z)
- WBCIC two- and three-class motor imagery: [Scientific Data 2025](https://doi.org/10.1038/s41597-025-04826-y)

Dataset access and redistribution remain subject to the terms of the original
providers. Trial-building code raises a clear error if a required raw data file
is still a zero-byte placeholder.

This directory intentionally does not include placeholders for files that are
not required by the preprocessing code, such as dataset article PDFs,
supplementary PDF documents, Jupyter notebooks (`.ipynb`), or MATLAB helper
scripts (`.m`). Public users do not need to create those omitted files to run
the provided preprocessing pipeline.
