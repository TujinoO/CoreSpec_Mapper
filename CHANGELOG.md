# Changelog

## v4.0.0 - 2026-07-19

CoreSpec Mapper V4 Adaptive Mineral Evidence Engine desktop release.

### Added

- Added a Catalog-driven mineral evidence layer with dynamic wavelength, FWHM, data-physics, bad-band, and reference-availability gating.
- Added full-depth stratified scene calibration and sensor capability cards.
- Added automatic FWHM-aware spectral-library resampling, canonical-anchor retention, representative diversity, and complete selection/rejection audit artifacts.
- Added group-level SAM detection, continuum-removed SFF evidence, mineral diagnostic gates, unlocked within-group competition, confidence, stability, and rejection products.
- Added shared-evidence conservative, balanced, and sensitive ENVI Classification outputs.
- Added fixed-column correction for candidate detection and auditable post-classification removal of directional, elongated, small-component, and residual multicolumn artifacts with strong-evidence protection.
- Added non-overwriting project runs, input fingerprints, output manifests, PNG previews, CSV summaries, and JSON/Markdown/HTML quality reports.
- Added V4 CLI audit, run, catalog, and PySide6 desktop commands.
- Added desktop configuration round-trip, structured progress, elapsed/remaining time, and safe cancellation handling.

### Compatibility

- V3 commands, configurations, algorithms, and tests remain available.
- VNIR reflectance and TIR emissivity experts are registered but intentionally blocked until their domain-specific implementations are validated.

### Validation

- NC-1 800-line pilot produces non-zero balanced and sensitive detections for all seven validated SWIR minerals.
- Residual fixed-column sulfate artifacts are removed without forcing mineral quotas or manufactured detections.
- NC-1 full-depth validation run completed for all 5,446 lines with 332,815 valid mask pixels.
- Final validation quality grade is B / Warning, publishable with FWHM and no-ground-truth limitations stated.
- Balanced output contains non-zero detections for all seven validated SWIR minerals.
- Strict, balanced, and sensitive final classifications passed the nested policy checks.
- Two full-depth recomputations produced identical hashes for the 9 final classification `.dat` files.
- Final automated test suite result recorded for this release: 34 passed.

### Known Limits

- Missing sensor FWHM metadata requires constrained spectral interpolation and is reported as a quality warning.
- Quality grades measure input, numerical, stability, and artifact-control consistency; mineralogical accuracy still requires XRD, Raman, thin-section, or point-spectral truth data.
- ENVI ROI/SHP import, in-app ROI drawing, Windows installer packaging, VNIR, and TIR production experts remain subsequent product phases.

## v3.0.0 - 2026-07-19

CoreSpec Mapper V3 Automatic Library Ensemble release.

### Added

- Added automatic pure-mineral spectral library ensemble construction from bundled ENVI libraries.
- Added balanced representative selection with four references per target mineral.
- Added full-depth detector-column SAM percentile calibration.
- Added unlocked mineral subclass classification within carbonate, sulfate, and clay groups.
- Added domain offset calibration for mineral score comparison.
- Added V3 ENVI outputs for raw scores, calibrated scores, column thresholds, stripe masks, pre-spatial classes, final classes, previews, JSON summaries, and CSV counts.
- Added V3 balanced and relaxed NC-1 configuration profiles.

### Changed

- V3 no longer locks the mineral subclass directly from the first SAM endpoint.
- JHU spectral library wavelength units are interpreted from ENVI headers and converted from micrometers to nanometers when needed.
- README now records V3 validation status, output contracts, production commands, and rollback commands.

### Validation

- First 800 NC-1 lines validated for balanced and relaxed profiles.
- Full 5,446-line NC-1 balanced and relaxed production outputs generated and reopened through the project ENVI reader.
- Seven target minerals are non-zero in both full-depth profiles.
- Automated test suite result recorded in project validation notes: 20 passed.

### Known Limits

- V3 validation is an engineering workflow validation, not a ground-truth accuracy assessment.
- Balanced output should be treated as the default geological review baseline; relaxed output is a recall-check companion.
- GUI and Windows packaging are deferred until the backend interface is frozen after geological review.
