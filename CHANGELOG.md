# Changelog

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
