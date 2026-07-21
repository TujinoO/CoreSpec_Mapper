# Changelog

## v5.3.0 - 2026-07-21

### Added

- Added a nine-step desktop workflow with a dedicated standard-spectrum review step between mineral selection and mode configuration.
- Added automatic threshold trial on audited scene samples, editable percentile/SAM controls, explicit rerun, candidate/feature-ready counts, and persisted run-time overrides.
- Added the V5.3 joint threshold objective: Catalog-safe bounds, stratified scene evidence, spatial support, candidate coverage, edge/isolation risk, candidate-weighted fixed-column risk, and downstream mineral-feature support.
- Added the application-owned CoreMaskUNet v2 foreground model, manifest, and weights; automatic masking now has one integrated model and requires RGB input.
- Added immediate selected-input inventory, external-ENVI-mask preview, advanced-setting expansion/reset, observed-progress ETA fallback, and Chinese result-view labels.
- Added a silent `pythonw.exe` desktop-shortcut installer so normal desktop launch shows only the main window.
- Added project-scoped calibration from reviewed same-grid weak labels with interleaved depth-block holdout validation; only compact thresholds, score offsets, feature bounds, and evidence gates are learned, and no classification pixels are copied.
- Added robust diagonal-QDA evidence gates with a bounded single-feature physical fallback for rare, non-linearly separable mineral classes.
- Added shared cross-group clay competition so illite, montmorillonite, and kaolinite cannot claim the same pixel independently.

### Changed

- Replaced the eight-step combined evidence page with separate standard-spectrum and threshold-trial pages.
- Removed the selectable project-engine, placeholder quick-calibration, and non-functional output checkboxes from the user path; V5.3 runs the integrated adaptive engine and always writes the nested three-profile result set.
- Simplified the run page to three user-facing readiness states and one primary start action.
- Threshold overrides are re-evaluated on the full scene before use and cannot leave the Catalog-safe envelope or violate the conservative-to-sensitive ladder.
- Reference consensus, evidence stability, confidence, and spatial-cleanup protection are now calibrated per mineral when reviewed project evidence is available, rather than allowing a large class to set the gate for a rare class.
- Rebalanced the technical route to use high-recall group SAM discovery first, mineral-specific SFF/feature/QDA discrimination second, and repeated detector-column cleanup last.
- Project evidence calibration now resolves separate nested conservative, balanced, and sensitive gates instead of sharing one Boolean QDA gate across all profiles.
- Sensitive mineral feature gates require at least one explicit diagnostic feature plus SFF support, while balanced/conservative profiles retain all-feature competition.

### Fixed

- Fixed empty mineral-specific confidence samples producing an impossible confidence threshold of 1.0.
- Fixed one isolated detector column dominating the entire threshold objective; column risk is now scaled by its share of selected candidates.
- Fixed selected data and external masks appearing to do nothing before a full audit.
- Fixed missing standard-spectrum confirmation, inert threshold navigation, inert advanced settings, missing ETA fallback, raw English preview keys, and result preview selection.
- Fixed optional unavailable spectral features contaminating an otherwise valid mineral score with NaN.
- Fixed single-mineral groups being rejected by a nonexistent competition margin.
- Fixed three-profile cleanup nesting by preserving stronger-profile evidence instead of letting a weaker cleanup veto it.
- Fixed rare minerals being permanently zeroed by duplicated, uncalibrated consensus/stability/confidence gates after project evidence calibration.
- Fixed reviewed clay-like geological lineaments bypassing the final residual fixed-column check; directional geological protection and repeated narrow-column cleanup are now separate decisions.
- Fixed low project-retention fallback disabling directional filtering for an entire clay class; low retention is now audited and upstream evidence breadth is corrected instead.
- Fixed multi-mineral project evidence domains allowing rare classes to inherit a group-scale area cap, which could inflate dolomite or anhydrite far beyond their reviewed project prior.
- Fixed SFF quality being taken from any library reference instead of references belonging to the winning mineral.
- Fixed short detector-column responses recurring in several separated core boxes escaping the whole-image column-density gate.
- Added a post-nesting, iterative repeated-column terminal filter that preserves laterally or obliquely supported geology and cascades removals without breaking the three-profile subset relation.
- Added a final post-clay-competition terminal pass so cross-group deletion cannot expose new narrow-column peaks after artifact control has already run.
- Added the `REPEATED_NARROW_COLUMN_RESIDUAL` publish gate with per-class segment repetition, narrow-column, and removable-pixel metrics.
- Fixed nearby repeated peak bands inside one detector corridor protecting each other as apparent oblique geology. A corridor up to 24 columns wide is now suppressed when it contains more than 55% of one class across at least four separated core sections, unless the response extends geologically across both sides of the full corridor.

### Validation

- Automated source regression suite: 144 passed. All V4/V5 evidence and stripe-control tests pass, including dominant-corridor suppression and bilateral geological-structure preservation cases.
- Built the Windows x64 stable installer from the r7 CPU-only frozen runtime. Both pre-install and installed-runtime self-tests pass with Qt, 27 Catalog minerals, 27 reference-mineral groups, Torch 2.7.1+cpu, and foreground-model CPU inference available.
- Verified the per-user installation, desktop and Start Menu shortcuts, uninstall registration, bundled database/Catalog/model/documentation, and GUI event-loop startup. No V5.2 application directory or shortcut remains.
- Re-ran the full 5,446 × 320 × 212 NC-1 adaptive workflow with the approved 332,814-pixel external mask under independent run id `project_calibrated_v14_dominant_corridor_20260721`; quality B and `publishable=true`.
- Balanced detections are Calcite 2,426; Dolomite 1,442; Anhydrite 233; Gypsum 6,317; Illite 1,298; Montmorillonite 2,265; and Kaolinite 2,559. No requested mineral is zero.
- Sensitive detections are Calcite 5,402; Dolomite 3,080; Anhydrite 468; Gypsum 24,531; Illite 1,888; Montmorillonite 5,318; and Kaolinite 4,517, retaining a materially wider profile without bypassing mineral evidence gates.
- Against the reviewed project weak labels, balanced Calcite reached 63.64% precision / 48.27% recall, Dolomite 31.28% / 28.98%, Gypsum 57.10% / 15.85%, Montmorillonite 9.49% / 5.82%, and Kaolinite 19.30% / 12.27%. These are project regression metrics, not independent mineralogical truth.
- All three final profiles contain zero removable repeated-column pixels, zero repeated narrow columns, and zero dominant repeated corridors in all five mineral groups. Maximum balanced column densities are 6.41% for Illite, 6.39% for Montmorillonite, and 5.37% for Kaolinite.

## v5.2.0 - 2026-07-21

### Added

- Added NC-1 full-scene `validated_v3` recipes and a reusable `v5-validate` regression gate with per-mineral IoU, precision, recall, and area-difference limits.
- Added GeoCore M1-2 deployment inspection and live adapter support; incomplete model code, manifest, or weights are now reported explicitly.
- Added mask interior-fill diagnostics, edge-network hard blocking, structured engine provenance, and a required visual approval record.
- Added an eight-step desktop workflow, a distinct run gate, threshold JSON export, non-increasing ETA display, and fit/50%/1:1/200% result zoom with scroll panning.
- Added validated-recipe rejection waterfalls and explicit handling of optional orphan ENVI headers.

### Fixed

- Restored the NC-1 expected mineral maps by isolating mask error from classification and executing the frozen, previously validated reference/threshold recipes.
- Prevented C/Warning or non-publishable runs from being shown as an unconditional green completion.
- Prevented V5.1 tray-outline/crack masks from passing only because their total area was large enough.

### Validation

- Completed a 5,446 × 320 × 212 NC-1 run with the approved 332,814-pixel mask.
- Balanced counts are Calcite 3,199; Dolomite 1,556; Anhydrite 423; Gypsum 22,757; Illite 5,759; Montmorillonite 3,692; Kaolinite 4,025.
- All 14 balanced/sensitive mineral comparisons achieved IoU = precision = recall = 1.000 against the frozen historical workflow weak labels. Quality grade A and `publishable=true` apply to regression equivalence, not independent mineralogical truth.

## v5.1.0 - 2026-07-20

### Added

- Added runnable experimental SWIR definitions for jarosite, nontronite, talc, tremolite, actinolite, biotite, phlogopite, siderite, sepiolite, vermiculite, and buddingtonite.
- Added dedicated amphibole Mg-Fe-OH and ammonium-feldspar experts, and expanded carbonate, acid-sulfate, smectite, and Mg-Fe-OH competition features.
- Added a packaged disabled recognition reserve with 241 automatically parsed phase labels and 914 pure-reference measurements awaiting expert definitions and validation.

### Changed

- Expanded the knowledge and runtime catalogs from 16 to 27 targets and the runnable SWIR set from 13 to 24 targets.
- Rebuilt the private database with 428 eligible declared-pure measurements; the 11 added targets contribute 140 measurements.
- The seven validated SWIR targets remain selected by default. All 17 experimental targets are user-selectable but remain unchecked by default.
- Tremolite/actinolite and biotite/phlogopite remain separate outputs inside shared competition families; siderite and tremolite-actinolite phase labels are explicitly conditional.

### Validation

- Catalog, database, runtime-feature, reference-selection, and desktop tests were extended for the new targets and disabled reserve.
- Experimental availability means the end-to-end evidence chain is runnable; it does not claim ground-truth mineralogical accuracy.

## v5.0.0 - 2026-07-20

CoreSpec Mapper V5 Adaptive Mineral Evidence Engine release.

### Added

- Added a packaged, read-only SQLite private spectral database with 27 source ENVI libraries, 1,783 measurements, 1,143 consolidated samples, 16 catalogued mineral targets, and 288 eligible declared-pure reference measurements.
- Added source-role, data-physics, measurement-geometry, sample identity, purity, numeric QC, label-conflict, wavelength-axis, curve, and source-file provenance records.
- Added a two-level user taxonomy and an independent spectral-competition layer for carbonates, calcium sulfates, white mica–illite, smectites, kaolin doublets, short-wave Al-OH, Mg-Fe-OH, acid sulfates, epidote, iron oxides, and TIR silicates.
- Added runtime support for 13 SWIR targets: seven validated defaults and six opt-in experimental targets. Hematite/goethite remain family-resource only, and quartz remains disabled without TIR.
- Added sensor-aware reference selection from the built-in database: required-window gating, FWHM convolution or constrained interpolation, physical-sample deduplication, robust medoid selection, and greedy spectral-shape/source/geometry diversity.
- Added automatic reflectance material masks with border-robust/Otsu thresholding, cropped-ROI fallback, component cleanup, hard coverage gates, and structured quality flags.
- Added multi-role input records for a primary analysis cube plus optional RGB, NIR, and SWIR companion datasets.
- Added Catalog-bounded adaptive SAM threshold search with complete candidate objectives, coverage, spatial-support, fixed-column, edge, isolated-component, and overcoverage terms.
- Added the vectorised segmented-linear continuum path used by V5.
- Added strict reference-consensus, evidence-stability, safe confidence-floor, mineral feature, depth, margin, and nested three-profile gates.
- Added Al-OH short/medium/long wavelength subtype rasters and previews for the white-mica–illite family.
- Added V5 run manifests with database/knowledge/runtime Catalog hashes, selected reference curves and reasons, resolved thresholds, timing, input identities, material-mask audit, output inventory, and an explicit non-recursive self-inventory policy.
- Added `v5-audit`, `v5-run`, `v5-catalog`, `v5-library`, `desktop`, and `desktop-v4` CLI commands.
- Added a seven-step PySide6 desktop workflow for Project, Data, Mask, Minerals, Mode, Evidence Review, and Run & Results.
- Added V5 configuration templates for default, 3DSSZ raw, and ZKH3 raw workflows.

### Changed

- The V5 desktop and service no longer accept or display a user spectral-library root. Packaged resources are resolved automatically.
- The primary analysis cube is the only pixel-level classification grid. Optional unregistered companion datasets are fingerprinted but are not silently fused.
- The default mineral selection remains the seven validated SWIR targets; white mica and dickite may be included as internal competitors without forcing them into the output classes.
- Scene-domain mineral median offsets are disabled, so absent minerals are not shifted into competition.
- Scene-derived confidence thresholds now act as an additional floor rather than weakening the Catalog threshold.
- Column spectral correction is confined to high-recall group detection; subclass continuum, feature centers, and competition use the original SG spectral domain.
- SFF matrices are skipped when the expert fit weight is zero.
- Scientific audit sampling is deterministically capped while retaining full-depth strata.
- ENVI outputs carry source paths, source dimensions, output line bounds, and inherited spatial metadata.
- `corespec-desktop` now launches V5; `corespec-desktop-v4` retains the legacy UI.

### Fixed

- Fixed V4-era spectral-name pollution that could associate Prehnite with Chlorite, Talc with Pyrophyllite, Ilmenite with Hematite, or Quartz Monzonite with pure Quartz.
- Fixed missed IGCP underscore-style mineral names through source-aware normalization.
- Fixed duplicate measurements of one physical sample inflating reference prior weight.
- Fixed canonical anchors consuming an entire representative budget; V5 reserves at most two configured anchors and otherwise starts from a robust medoid.
- Fixed SG smoothing peak-memory growth caused by materialising a pixel-by-window sliding view.
- Fixed platform-sensitive small-matrix and tall-by-short BLAS paths by using explicit small-system solving and safe NumPy contractions.
- Fixed hidden same-process audit repetition by caching the mask, sample, capability card, column bias, and selected reference ensemble in a V5 audit snapshot.
- Fixed adaptive-threshold evidence normalization to use one Catalog-safe reference scale instead of each candidate's own absolute threshold, removing the structural bias toward the widest absolute SAM bound.
- Fixed equal-objective threshold ties so the deterministic resolver prefers the lower absolute SAM threshold and then the lower column percentile instead of retaining an unnecessarily permissive candidate.
- Fixed the prior manifest self-size/self-hash inconsistency by excluding `run_manifest.json` from its own inventory.

### Validation

- Completed final full-resolution 3DSSZ raw SWIR validation at 5,663 × 320 × 212 with an external 222,783-pixel material mask. The run completed in 293.375 seconds and reported B / Warning.
- Completed final full-resolution ZKH3 raw SWIR validation at 4,341 × 320 × 212 with an automatic 521,852-pixel material mask. Mapping completed in 236.188 seconds, total runtime was 243.000 seconds, and the run reported B / Warning.
- Completed final harmonized low-resolution regressions at 293 × 160 × 367 in 12.484 seconds for 3DSSZ and 297 × 158 × 367 in 14.000 seconds for ZKH3; both reported B / Warning.
- Each of the four final runs contains 312 manifest inventory entries, 24 preview assets (23 PNG previews plus one `legend.json`), including material-mask and Al-OH subtype images, and complete three-profile group outputs.
- Final raw-mask audits retain 159 components for the external 3DSSZ mask and 263 components for the automatic ZKH3 mask; both carry fragmentation warnings and require geological review.
- Zero detections remain explicit warnings; no mineral quota is used to manufacture positive output.
- All four runs report missing FWHM and no pixel-level mineral ground truth. They demonstrate end-to-end engineering execution, not mineralogical accuracy.

### Compatibility

- V3 and V4 commands, configurations, and the legacy desktop entry point remain available.
- The current V5 mapper internally reuses the stable raster executor behind the V5 service, Runtime Catalog, database, calibration, continuum, and safety configuration.

### Known Limits

- RGB/NIR/SWIR companion files are not automatically registered or fused. Joint analysis requires an already registered harmonized cube as the primary analysis input.
- The current NIR begins at 691 nm and RGB is not a calibrated hyperspectral VNIR measurement; hematite/goethite mineral-level separation is disabled.
- Quartz recognition is disabled because no TIR sensor and no compatible emissivity reference set are present.
- The desktop Quick Calibration control is disabled with an explicit no-pixel-truth explanation and configuration is forced to `quick_calibration=false`; supervised ROI/point-truth calibration is not implemented in 5.0.0.
- Three-profile output is fixed on and configuration is forced to `write_all_profiles=true`; `mode.profile` controls only the default results view.
- Audit shows threshold-safe candidate ranges; final optimized thresholds become available after the mapping stage.
- The spectral overlay provider protocol is reserved, but external overlay database merging is not implemented in 5.0.0.
- `--start-line/--stop-line` crop written outputs but group SAM and threshold calibration still scan and allocate the full image; they are not an efficient ROI execution mode.
- All 27 source-library license records are currently `unverified` and require review before external database redistribution.
- Quality grades measure engineering consistency and artifact control, not mineralogical accuracy.

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
