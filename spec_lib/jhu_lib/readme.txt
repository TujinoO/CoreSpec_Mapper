This directory contains files from the Johns Hopkins University Spectral
Library provided by Jack Salisbury at JHU.  These data are also available
on-line from JHU via anonymous ftp at rocky.eps.jhu.edu. Also see the the data
on the ASTER homepage at:  http://asterweb.jpl.nasa.gov/speclib/.

The following files are provided:

readme.txt		This File
ign_crs.sli		Igneous Rocks - Coarse (0.4 - 14 um)
ign_crs.hdr		ENVI Header for Above
ign_fn.sli		Igneous Rocks - Fine (0.4 - 14 um)
ign_fn.hdr		ENVI Header for Above	
lunar.sli		Lunar Materials (2.08 - 14 um)
lunar.hdr		ENVI Header for Above
manmade1.sli		Man Made Materials (0.42 - 14 um)
manmade1.hdr		ENVI Header for Above
manmade2.sli		Man Made Materials (0.3 - 12.5 um)
manmade2.hdr		ENVI Header for Above
meta_crs.sli		Metamorphic Rocks - Coarse (0.4 - 14.98 um)
meta_crs.hdr		ENVI Header for Above
meta_fn.sli		Metamorphic Rocks - Fine (0.4 - 14.98 um)
meta_fn.hdr		ENVI Header for Above
meteor.sli		Meteorites (2.08 - 25 um)
meteor.hdr		ENVI Header for Above
minerals.sli		Minerals (2.08 - 25 um)
minerals.hdr		ENVI Header for Above
sed_crs.sli		Sedimentary Rocks - Coarse (0.4 - 14 um)
sed_crs.hdr		ENVI Header for Above
sed_fn.sli		Sedimentary Rocks - Fine (0.4 - 14.98 um)
sed_fn.hdr		ENVI Header for Above
snow.sli		Snow (0.3 - 14 um)
snow.hdr		ENVI Header for Above
soils.sli		Soils (0.42 - 14 um)
soils.hdr		ENVI Header for Above
veg.sli			Vegetation (0.3 - 14 um)
veg.hdr			ENVI Header for Above
water.sli		Water (2.08 - 14 um)
water.hdr		ENVI Header for Above


The following is the README file provided by JHU with the data.

With the exception of manmade materials, all spectra in the Johns Hopkins 
Library were measured under the direction of John W. Salisbury.  Most 
measurements were made by Dana M. D'Aria, either at Johns Hopkins University 
in Baltimore, MD, or at the U.S. Geological Survey in Reston, VA.

This text is a general introduction to the library, with an overview of
measurment techniques, which do differ for different materials.  There is a
separate introductory text for each kind of material (rocks, minerals, lunar
soils, terrestrial soils, meteorites, etc.) that contains more detailed
information.

MEASUREMENT TECHNIQUE

Two different kinds of spectral data are resident in this library.  Spectra of
minerals and meteorites were measured in bidirectional (actually biconical) 
reflectance (see two Salisbury et al., 1991 references below for details).
These spectra, recorded from 2.08-25 micrometers, cannot be used to
quantitatively predict emissivity because only hemispherical reflectance can be
used in this way.  However, when recorded properly, as described in the
meteorite paper, curve shape is accurate and can be used for remote sensing
applications.

All other spectral data, with the exception of portions of generic snow and 
vegetation spectra (see the introductory text for each type of material), were 
measured in directional hemispherical reflectance.  Under most conditions, the 
infrared portion of these data can be used to calculate emissivity using
Kirchhoff's Law (E=1-R), which has been verified by both laboratory and field
measurements (Salisbury et al., 1994; Korb et al., 1996).  The unusual
circumstances (e.g., the lunar environment) where thermal gradients may cause
significant departure from Kirchhoffian behavior are discussed in Salisbury
et al., 1994.

The apparently seamless reflectance spectra from 0.4 to 14 micrometers of rocks 
and soils were generated using two different instruments, both equipped with 
integrating spheres for measurement of directional hemispherical reflectance, 
with source radiation impinging on the sample from a centerline angle 10
degrees from the vertical.  

Unless specified otherwise (see relevant introductory texts for generic snow
and vegetation spectra, and spectra of manmade materials), all
visible/near-infrared (VNIR) spectra were recorded using a Beckman Instruments
model UV 5240 dual-beam, grating spectrophotometer at the U.S. Geological
Survey, Reston, VA.  The data were obtained digitally and corrected for both
instrument function and the reflectance of the Halon reference using standards
traceable to the U. S. National Institute of Science and Technology.
Measurements of such standards indicate an absolute reflectance accuracy of
plus or minus 3 percent.  Wavelength accuracy was checked using a holmium oxide
reference filter and is reproducible and accurate to within plus or minus
0.004 micrometers, or 4 nm (one digitization step).  Spectral resolution is
variable because the Beckman uses an automatic slit program to keep the energy
on the detector constant.  The result is a spectral bandwidth typically less
than 0.008 micrometers over the 0.4 to 2.5 micrometers spectral range measured,
but slightly larger at the two extremes of the range of the lead sulfide
detector (0.8-0.9 micrometers and 2.4-2.5 micrometers).  This instrument has a
grating change at 0.8 micrometers, which sometimes results in a spectral
artifact (either a small, sharp absorption band, or a slight offset of the
spectral curve) at that wavelength.

Two similar instruments were used to record reflectance in the infrared range
(2.08 to 15 micrometers).  Briefly, both are Nicolet FTIR spectrophotometers
and both have a reproducibility and absolute accuracy better than plus or minus
one percent over most of the spectral range.  Early measurements of igneous
rocks with an older detector were noisy in the 13.5-14 micrometers range and do
not quite meet this standard.  Because FTIR instruments record spectral data in
frequency space, both wavelength accuracy and spectral resolution are given in
wavenumbers (reciprocal centimeters).  Wavelength accuracy of an
interferometer type of instrument is limited by the spectral resolution, which
yields a data point every 2 wavenumbers for these measurements.  The X-axis was
changed from wavenumbers to micrometers for all of these data before the 
infrared segment was joined to the VNIR data from the Beckman.

Spectra from the Beckman and the FTIR instruments were compared in the
overlap range of 2.08-2.5 micrometers.  If the difference was greater than 3
percent, measurements were repeated.  Typically, however, the agreement was
within the 3 percent limit.  In view of the greater accuracy of the FTIR
measurements, any small discrepancy between the two spectral segments was
resolved by adjusting the Beckman data to fit the reflectance level of the
segment measured by the FTIR instruments.

REFERENCES

Korb, A. R., Dybwad, P., Wadsworth, W., and Salisbury, J. W., 1996, 1996,
Portable FTIR spectrometer for field measurements of radiance and emissivity:
Applied Optics, v. 35, p. 1679-1692.

Salisbury, J. W., D'Aria, D. M., and Jarosevich, E., 1991a, Midinfrared
(2.5-13.5 micrometers) reflectance spectra of powdered stony meteorites:
Icarus, v. 92, p. 280-297.

Salisbury, J. W., Wald, A., and D'Aria, D. M., 1994, Thermal-infrared remote
sensing and Kirchhoff's law  1. Laboratory measurements:  Jour. of Geophysical
Research, v. 99, p. 11,897-11,911.

Salisbury, J. W., Walter, L. S., Vergo, N., and D'Aria, D. M., 1991b, 
Infrared (2.1-25 micrometers) Spectra of Minerals: Johns Hopkins University
Press, 294 pp.
