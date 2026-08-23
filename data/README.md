# Data Rules

## `public/`

Store public datasets only when their licenses permit local research use. For each dataset, add a small provenance note containing the source URL, license or terms, download date, expected files, and checksum.

## `company/`

Confidential company data is local-only and ignored by Git.

- `raw/`: immutable originals; never preprocess in place.
- `anonymized/`: approved de-identified working copies.
- `annotations/`: masks and labels linked to anonymized identifiers only.

Before adding data, document authorization, intended use, access control, anonymization, retention, backup, and deletion. Signatures are biometric and identity-linked material; a public repository is not an acceptable storage location.

## `synthetic/`

Generated signatures, stamps, composites, and exact ground-truth masks. Record the generator version and parameters so every dataset can be recreated.

## Split rule

Use writer-disjoint train, validation, and test partitions for writer-independent claims. Store the split manifest separately from model code and never tune on the final test writers.
