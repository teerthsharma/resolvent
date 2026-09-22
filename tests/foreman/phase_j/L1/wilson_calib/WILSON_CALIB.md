# WILSON: calibration of precision wave machines, plus EPSILON-PHASE and epsilon-cli

What the labels mean:
- **[V] verified**: Wilson checked the text against the source directly, by his own fetch, a Europe PMC abstract, a PDF read, or a GitHub API GET.
- **[U] unverified**: only a nurse reported it, the source would not load for Wilson, or no fetched page was cited.

Twelve haiku nurses were sent out in two batches (10 logged, then 2). Several of their "verbatim" quotes turned out to be paraphrases or had the wrong source. Each case found is flagged below.

---

## 1. Diagnostic ultrasound

### (a) Mode definitions
- [V] **A-mode:** the signal amplitude was proportional to the strength of the reflected signal. Source: Powers J, Kremkau F, "Medical ultrasound systems", *Interface Focus* 2011, https://pmc.ncbi.nlm.nih.gov/articles/PMC3262275/
- [V] **M-mode:** "Motion, or M-mode, was developed in which the signal amplitude was displayed as a function of time." Same source.
- [V] **B-mode:** two-dimensional brightness mode, first built with position-sensing arms that recorded where the beam was and which way it pointed. Same source.
- [U] StatPearls NBK570593 (Grogan SP, Mount CA, "Ultrasound Physics and Instrumentation") defines M-mode as a single beam along a defined track, recorded over time. A nurse reported this. Wilson's fetch hit a reCAPTCHA.
- [U] Radiopaedia describes M-mode as a single scan line emitted, received and displayed graphically: https://radiopaedia.org/articles/m-mode-ultrasound
- [U] No mode definition was retrieved from AIUM, ACR or IEC (IEC Electropedia). The AIUM PDF came back to the nurse as binary, and the IEC term pages were not fetched.

### (b) Speed of sound, mismatch error, correction
- [V] Clinical scanners typically assume 1540 m/s for transmit and receive beamforming. A wrong beamforming sound speed adds the wrong curvature and bulk shifts to the delay profiles. The curvature causes defocusing and the shifts cause mis-registration.
  - Source: Ahmed R, Trahey G, "Spatial Ambiguity Correction in Coherence-based Average Sound Speed Estimation", IEEE TUFFC 2024, https://pmc.ncbi.nlm.nih.gov/articles/PMC11575430/
  - Estimation methods the paper names: Anderson and Trahey curve-fitting on arrival-time profiles, coherence-factor maximisation, and the authors' own 1D/2D spatial-ambiguity correction.
- [V] **Phase-aberration correction (O'Donnell M, Flax SW, IEEE TUFFC 1988, doi 10.1109/58.9334, PMID 18290214):**
  - Near-field velocity inhomogeneities were modelled with grooved plates.
  - An iterative correction used cross-correlation between neighbouring array elements on signals from diffuse scatterers.
  - The corrected images were nearly identical to images taken without aberration.
  - Checked against the Europe PMC abstract.
- [V] **Other correction methods (Yen J, IEEE TUFFC 2012, https://pmc.ncbi.nlm.nih.gov/articles/PMC3525140/):**
  - Phase aberration comes from sound-speed inhomogeneities in tissue.
  - Named methods: time-shift compensation using element-to-element or element-to-beamsum correlations, backpropagation of the received field, time-reversal focusing, and speckle brightness as a quality factor.
  - The paper itself uses nearest-neighbour cross-correlation (NNCC), after O'Donnell et al.
- [U] NBK570593: 1540 m/s is the accepted soft-tissue speed. The nurse's quote says this is "known as acoustic impedance". Wilson could not load the page to check it.
- [U] PMC9127706 (2021, local sound speed in layered media): a speed-of-sound error misaligns channel signals.
- [U] PMC10842235 (2023): breast tissue sound speed can vary by up to 100 m/s.
- [U] PMC3183956 (2011): a beamformer/tissue speed-of-sound mismatch shifts the power spectra.
- [U] Vendor features (Zonare mean-velocity estimation, Siemens "Fatty Tissue Imaging", Philips fat-layer assumption): these came from a search summary, not a fetched page.

### (c) Speckle
- [V] Ultrasound imaging is a coherent process. Scattering from a diffuse medium interferes constructively and destructively, and this produces coherent speckle. Source: Powers and Kremkau 2011 (PMC3262275).
- [V] **Frequency compounding (Trahey GE, Allison JW, Smith SW, von Ramm OT, "A quantitative approach to speckle reduction via frequency compounding", *Ultrasonic Imaging* 8(3):151–164, 1986, PMID 3548000):**
  - The method averages images whose speckle has been changed by shifting the pulse spectrum.
  - The paper quantifies the trade-off between lost resolution and reduced speckle.
  - It concludes: "Our analysis indicates that simple frequency compounding is counterproductive in improving image quality." (Europe PMC abstract.)
  - The nurse's "verbatim" quote from this paper was a paraphrase and left this conclusion out.
- [U] Rayleigh first-order amplitude statistics for fully developed speckle: Wagner RF, Smith SW, Sandrik JM, Lopez H, "Statistics of speckle in ultrasound B-scans", IEEE Trans Sonics Ultrason 30(3):156–163, 1983. Europe PMC returned no record, and Wilson did not fetch the paper.
- [U] Burckhardt CB, "Speckle in ultrasound B-mode scans", IEEE Trans Sonics Ultrason 25(1):1–6, 1978. Nurse note: compounding reduces speckle in proportion to the number of independent measurements.
- [U] Non-Rayleigh models:
  - Tuthill, Sperry and Parker 1988, *Ultrasonic Imaging* 10(2):81–89: K and Rician deviations.
  - Shankar 2000, IEEE TUFFC 47(3):727–736: Nakagami distribution.
- [U] Speckle reduction:
  - Yu and Acton 2002, IEEE TIP 11(11):1260–1270: SRAD.
  - Magnin, von Ramm and Thurstone 1982: frequency compounding.
- [U] No primary source on spatial compounding was fetched.

### (d) Time-gain compensation
- [U] NBK570593: TGC is power control at specific image depths to counter attenuation with depth. A nurse reported this; Wilson's fetch hit a reCAPTCHA.
- [U] PMC2151761: attenuation is typically compensated with a TGC curve.
- [U] PMC3183956 gives soft-tissue attenuation of 0.54 dB/cm/MHz (fat 0.48, liver 0.5, breast 0.75).
- [U] No named automatic-TGC method was retrieved. The nurse cited only a patent search.

### (e) Quality assurance
- [V] **AIUM, "Routine Quality Assurance for Diagnostic Ultrasound Equipment" (©2008; ISBN 1-932962-12-3; subcommittee Boote, Forsberg, Garra, J. Ophir, K. Ophir, Zagzebski; http://aium.s3.amazonaws.com/resourceLibrary/rqa.pdf). Wilson read pp. 1–5 of the PDF.**
  - **p.1:** the procedures are for grayscale imaging mode. Basic performance tests may be done in grayscale only. Because the tests focus on the transducer, the results apply "to a limited degree to Doppler and color flow performance".
  - **p.2:** periodic tissue-mimicking-phantom results are compared with baseline results taken at acceptance or when the programme starts. It refers to Goodsitt et al. for tolerance and action levels when results drift from baseline.
  - **p.3:** phantoms have been known to drift or deteriorate over time, so the manufacturer's recertification schedule should be followed. AIUM recommends annual periodic tests.
  - **p.5, Section B, daily tests (sonographer, mandatory):**
    - B.1: brightness and contrast controls are still at their calibration points.
    - B.2: the full gray bar is displayed.
    - B.3: hard-copy and workstation gray levels match the monitor.
    - B.4: no vertical shadows or streaks from dead elements.
  - **p.5, Section B, annual tests:**
    - B.5: transducer cables, housing and surfaces are free of cracks (mandatory).
    - B.6: transducer uniformity (mandatory).
    - B.7: maximum depth of visualisation, repeated for each frequency setting (mandatory).
    - B.8: target detection and imaging (recommended).
    - B.9: horizontal and vertical distance measurement accuracy, including offline and 3D measurements (mandatory).
  - The pages Wilson read give no numeric tolerances.
- [V] AAPM Ultrasound Task Group No. 1: Goodsitt MM, Carson PL, Witt S, Hykes DL, Kofler JM, "Real-time B-mode ultrasound quality control test procedures", *Med Phys* 25(8):1385–1406, 1998, PMID 9725125, doi 10.1118/1.598404. Only the bibliographic record was checked; Europe PMC has no abstract.
- [U] Goodsitt 1998 action levels, as the nurse reported them:
  - Vertical distance accuracy: 1.5 mm or 1.5%.
  - Horizontal distance accuracy: 2 mm or 2%.
  - Depth of penetration: more than 0.6 cm change from baseline.
  - The Wiley page returned 403 to Wilson.
- [V] **BMUS QA guidelines (Dudley N, Russell S, Ward B, Hoskins P, *Ultrasound* 2013/2014, https://pmc.ncbi.nlm.nih.gov/articles/PMC4760519/):**
  - Air-reverberation test tolerance: ± the distance to the adjacent reverberation line.
  - Electronic-noise tolerance: set subjectively, by repeating the measurement to establish a likely range.
  - Changes against baseline images are to be reported.
- [V] **IEC agar tissue-mimicking material, as measured (Sun C et al., *Ultrasound Med Biol* 38(7):1262–1270, 2012, PMC3377968):**
  - Speed of sound 1547.4 ± 1.4 m/s (Vevo 770) and 1548.0 ± 6.1 m/s (SAM).
  - Attenuation 0.40f + 0.0076f² dB/cm, with f in MHz.
  - The nurse attributed an IEC specification of "1540 ± 15 m/s, 0.5 ± 0.05 dB/cm/MHz" to this paper. Two fetches by Wilson found no "1540" in it, so that specification stays **[U]**.
- [U] ACR–AAPM technical standard (US-Equip.pdf): a Qualified Medical Physicist performs or supervises acceptance testing and routine evaluations.
- [U] IEC 61391-1:2006+A1:2017: calibration of spatial measurement, 0.5–15 MHz.
- [U] IEC 61391-2:2010: maximum depth of penetration and local dynamic range, 1–15 MHz, fundamental and harmonic modes.
- [U] The claim that IEC 61391-1 requires point-spread-function target placement within ±0.1 mm came from an unnamed source.

## 2. MRI

- [V] **B0 homogeneity (Chen HH, Boykin RD, Clarke GD, Gao JH, Roby JW, "Routine testing of magnetic field homogeneity on clinical MRI systems", *Med Phys* 2006, PMID 17153408):**
  - Poor B0 homogeneity leads to artifacts and signal loss.
  - The ACR MRI QC manual mandates annual homogeneity checks, suggesting spectral-linewidth and phase-difference (Δφ) maps.
  - A small receiver bandwidth combined with poor homogeneity causes geometric distortion.
  - Measured homogeneity ranged from 0.11–0.32 ppm to 6.7–12.9 ppm for spherical volumes (DSV) of 13–22.6 cm, across 7 systems at 0.2–3.0 T.
  - The nurse attributed a prescan sentence to this PMID ("prescan ... sets shim, CF, RF transmit gain, receiver gain"). That sentence is **not** in the abstract.
- [V] **Prescan, per mriquestions.com (secondary source, no author or date; https://mriquestions.com/automatic-prescan.html):**
  - Centre frequency is set from the dominant water resonance.
  - RF transmit power is calibrated so pulses give the intended flip angles.
  - Receiver gain is scaled to the receiver chain and ADC.
  - Shim currents are adjusted with the patient in the magnet.
  - The page does not say whether prescan is repeated for each series.
- [V] **Routine shimming, per mriquestions.com (https://mriquestions.com/routine-shimming.html):**
  - A quick shim takes only a few seconds.
  - Reshim when imaging conditions change meaningfully: table position, anatomic region, coil setup or gradient mode.
  - Shimming matters most for off-resonance-sensitive methods: spectral fat suppression, EPI and balanced SSFP.
- [V] **B1 mapping by Actual Flip-angle Imaging (Yarnykh VL, *MRM* 2007, PMID 17191242, Europe PMC abstract):**
  - Two identical RF pulses are followed by delays TR1 < TR2.
  - The flip angle follows from r = S2/S1 = (1 + n·cosFA)/(n + cosFA), where n = TR2/TR1.
  - The measurement is highly insensitive to T1.
- [U] B1 mapping by Bloch–Siegert shift (Sacolick et al., *MRM* 63(5):1315–1322, 2010, PMC2933656): the phase difference between two off-resonance RF applications removes unwanted off-resonance effects.
- [U] Double-angle method (PMC6491230): images at α and 2α, with the signal ratio giving the B1+ map.
- [U] Juchem and de Graaf 2017 (PMC5148734): B0 shimming is among the most important factors in MRS.
- [U] PubMed 28602943 (7 T review): active shim coils follow spherical-harmonic shapes.
- [U] PMC4856011: B0 inhomogeneity gives spatially varying off-resonance and accumulated phase.
- [U] PMC5740018: concomitant fields add unwanted phase to k-space.
- [V] **Noise model (Gudbjartsson H, Patz S, "The Rician distribution of noisy MRI data", *MRM* 1995, PMID 8598820):**
  - Magnitude-image intensity follows a Rician distribution.
  - Low signal (SNR < 2) is biased, and a correction scheme is given.
  - Phase-image noise behaves very differently from magnitude-image noise.
  - Both are nearly Gaussian for SNR > 2.
- [V] **NEMA MS 1-2008, "Determination of Signal-to-Noise Ratio (SNR) in Diagnostic Magnetic Resonance Imaging" (NEMA). Wilson read the PDF at https://mriquestions.com/uploads/3/4/5/7/34572113/nema_snr_standards_2008.pdf, pp. i–9.**
  - **Scope:** single-channel volume receive coils. Special-purpose coils are covered by MS 6 and multi-channel coils by MS 9.
  - **Rationale:** changes in system calibration, gain, coil tuning or RF shielding usually show up as a change in SNR.
  - **Preamble:** the measurement uncertainty must be reported, with justification.
  - **§1.3:** the measurement ROI covers at least 75% of the phantom's signal area.
  - **§2.2 scan conditions:** temperature 22 ± 4 °C, TR ≥ 3 × T1, slice ≤ 10 mm, FOV ≤ 110% of the coil dimension.
  - **§2.3.1(b):** perform the standard clinical pre-scan calibration.
  - **Method 1 (§2.3.2.1):** two images under identical conditions, with no adjustment or calibration between the scans. Methods 1 and 2 require less than 5 minutes between scans. Method 1 is subject to system drift artifacts. Noise = SD/√2 (Eq. 3). The successive-difference SD (Eq. 2) reduces the effect of low-frequency instability.
  - **Method 2:** a noise-only scan with no RF. For magnitude images, noise = SD/0.66, where 0.66 ≅ √((4−π)/2); NEMA cites Henkelman, *Med Phys* 12:232–233, 1985.
  - **Method 3:** odd/even k-space decimation, which is minimally sensitive to drift.
  - **Method 4:** background noise ROIs in the four corners, at least 1000 px total, with SD/0.66.
  - None of the methods is compatible with parallel imaging.
- [V] **ACR phantom criteria as stated by Wong OL et al., *QIMS* 7(2):205–214, 2017 (PMC5418154).** This is a secondary account, not the ACR manual:
  - Geometric accuracy: ±2 mm.
  - Slice thickness: 5.0 ± 0.7 mm.
  - Slice position: ≤ 5 mm.
  - High-contrast resolution: 1.0 mm.
  - Percent integral uniformity (PIU): ≥ 87.5% for B0 < 3 T.
  - Ghosting: ≤ 0.025.
  - Low-contrast detectability: ≥ 9 spokes below 3 T.
- [U] Dietrich et al., *JMRI* 26(2):375–385, 2007: parallel imaging changes the noise characteristics.

## 3. Radiation-therapy linear accelerators

- [V] **TG-51 (Almond PR, Biggs PJ, Coursey BM, Hanson WF, Huq MS, Nath R, Rogers DWO, *Med Phys* 26(9):1847–1870, 1999, PMID 10505874, Europe PMC abstract):**
  - Ion chambers carry absorbed-dose-to-water calibration factors N_D,w(⁶⁰Co) that are traceable to national primary standards.
  - Dose is D_w^Q = M·k_Q·N_D,w(⁶⁰Co), with M = P_ion·P_TP·P_elec·P_pol·M_raw.
  - Beam quality is %dd(10)x for photons and R50 for electrons.
  - Measurements are made in a water phantom. Reference depth is 10 cm for photons and 0.6R50 − 0.1 cm for electrons.
- [U] The NIST/ADCL wording of that chain: the nurse's quote is not in the abstract.
- [V] **The nurse misattributed a source.** PMC5874962 is **not** McEwen et al. 2014. It is McCaw TJ, Hwang MS, Jang SY, Huq MS, "Comparison of the recommendations of the AAPM TG-51 and TG-51 addendum reference dosimetry protocols", *JACMP* 2017, PMID 28574211. It reports:
  - Dose differences between the two protocols of 0.1–0.3% for flattened beams.
  - Up to 0.2% and 0.8% for flattening-filter-free beams, with scanning and Farmer chambers respectively.
  - "Combined uncertainty was between 0.91% and 1.2% (k = 1), varying by protocol and detector."
  - Its uncertainty budget lists 0.65% for the ⁶⁰Co N_D,w.
- [U] McEwen et al. 2014 photon addendum: reference-class chamber criterion 0.996 ≤ P_pol ≤ 1.004, with less than 0.5% variation across energies. This came from a search result.
- [V] **2024 electron addendum (Muir B et al., "AAPM WGTG51 Report 385: Addendum to the AAPM's TG-51 protocol ... electron beams", *Med Phys* 2024, PMID 38980220):**
  - Covers 4–22 MeV beams.
  - New Monte Carlo k_Q values.
  - Cylindrical chambers in all electron beams, with no gradient correction.
  - Dose per MU is expected up to about 2% higher than under the original TG-51.
- [U] IAEA TRS-398 (2000; Rev. 1 2024): SSDLs pass calibration factors from a PSDL (or the BIPM) on to hospital users. The IAEA page returned HTTP 402 to Wilson.
- [U] TRS-398 stated uncertainties: the "about 1%" and "1.4%" figures were nurse search summaries, not the IAEA text.
- [V] **TG-142 (Klein EE et al., "Task Group 142 report: Quality assurance of medical accelerators", *Med Phys* 36(9):4197–4212, 2009, doi 10.1118/1.3190392). Wilson read the PDF copy at w2.fisica.unam.mx, pp. 4197–4205.**
  - **Table I, daily:**
    - X-ray output constancy and electron output constancy: 3% (electron weekly, except for machines with unique e-monitoring).
    - Laser localization: 2 mm non-IMRT, 1.5 mm IMRT, 1 mm SRS/SBRT.
    - Distance indicator (ODI) at isocentre: 2 mm.
    - Collimator size indicator: 2/2/1 mm.
    - Door interlock, audiovisual monitors, radiation area monitor and beam-on indicator: "Functional".
  - **Table II, monthly:**
    - X-ray, electron and backup monitor chamber constancy: 2%.
    - Typical dose-rate output constancy: 2% (IMRT and SRS/SBRT only).
    - Photon and electron beam profile constancy: 1%.
    - Electron energy constancy: 2%/2 mm.
    - Light/radiation field coincidence: 2 mm or 1% per side.
    - Localizing lasers: ±2 mm, ±1 mm, <±1 mm.
    - Respiratory-gating beam output constancy: 2%.
  - **Table III, annual:**
    - X-ray and electron output calibration (TG-51): ±1% absolute.
    - X-ray flatness change from baseline: 1%. Symmetry: ±1%.
    - X-ray beam quality (PDD10 or TMR20/10): **±1%** from baseline. The nurse reported ±2%, which is wrong.
    - Electron beam quality (R50): ±1 mm.
    - X-ray MU linearity: ±2% for ≥5 MU (non-IMRT); ±5% at 2–4 MU and ±2% for ≥5 MU (IMRT and SRS/SBRT).
    - Output constancy versus dose rate: ±2%. Versus gantry angle: ±1%.
    - Collimator, gantry and couch rotation isocentre: ±1 mm.
  - **§II.C:**
    - The original tolerances, from AAPM Report 13 by quadratic summation, were meant to allow an overall dosimetric uncertainty of ±5% and a spatial uncertainty of ±5 mm.
    - Tolerances are deviations from baseline, not from the machine specification.
  - **§II.C.3, action levels:**
    - Level 1, inspection: a sudden, significant deviation from the expected value goes to the physicist even inside tolerance. Treatment continues.
    - Level 2, scheduled: triggered by consecutive results at or near tolerance, or a single result slightly over it. Investigate within one to two working days.
    - Level 3, immediate: stop treatment or take corrective action.
    - Institutions set the Level 2 and 3 thresholds. Level 1 thresholds evolve from the QA data.
  - **§II.C.4, measurement repeatability:** two standard deviations of three or more repeated consecutive measurements should be below the tolerance. The report's example chamber has 1.5% measurement uncertainty on absolute dose.
  - **§II.B:** each independent monitor chamber system should be checked daily.
- [U] AAPM TG-198 (Hanley J et al., *Med Phys* 48(10):e830–e885, 2021, doi 10.1002/mp.14992) is an implementation guide for TG-142 that adds VMAT tests. This came from a search summary.
- [U] The daily ±3% / ±5% "do not treat" escalation came from a search summary with no source identified.

## 4. Mode switching

- [V] Electronic beam steering made simultaneous modes possible, such as a real-time B-mode image with a simultaneous M-mode display. Source: Powers and Kremkau 2011 (PMC3262275).
- [V] **Duplex imaging (Grönlund C et al., *BioMed Eng OnLine* 2013, https://pmc.ncbi.nlm.nih.gov/articles/PMC3856612/):**
  - Doppler images are interleaved with B-mode images and colour-coded over the grayscale.
  - B-mode frame rates are relatively low, typically 20–40 Hz, with a typical CDI:B-mode frame-rate ratio of 3:1.
  - B-mode in duplex uses fewer image lines than standalone B-mode.
  - The paper does not say whether the B-mode image is used to place the Doppler gate.
- [V] AIUM 2008 (p.1): QA is done in grayscale mode only, and the results carry over to Doppler and colour flow only "to a limited degree". So the transducer is the reference shared across modes, and Doppler-specific performance is not re-established by these tests.
- [V] **NEMA MS 1-2008:**
  - §2.3.1(b): the standard clinical pre-scan calibration comes before the signal scan.
  - Methods 1 and 2: no system adjustment or calibration between the paired scans.
  - Method 2: receiver attenuation, gain and reconstruction scaling must be identical to the first scan.
- [V] mriquestions.com: reshim after a meaningful change in table position, anatomic region, coil setup or gradient mode. The prescan page does not say whether prescan repeats for each series.
- [V] TG-142 Table I: electron output is checked weekly or daily depending on the machine's e-monitoring design. §II.B: each independent monitor chamber system is checked daily.
- [U] PMC7251754 (2020): real-time frequency correction recalculates RF and ADC frequency and phase for both the navigator and the CEST sequence.
- [U] Radiology Cafe: MI below 0.7 for general use. This is not a standard, and whether IEC 60601-2-37 / AIUM ODS recompute MI and TI per mode was not retrieved.
- [U] PMC8607315: a 60° Doppler angle is a practical compromise.
- [U] Nothing retrieved states which MRI calibrations are per-patient and which are per-sequence for a named vendor (GE auto-prescan, Siemens adjustments).

## 5. The author's repositories (GitHub API GETs by Wilson; all [V])

### EPSILON-PHASE
teerthsharma/EPSILON-PHASE. Default branch main; description null; last push 2026-04-15.

- **Positioning.**
  - README.md:3–5 calls the project a wind-simulation engine "powered by a numeric robustness layer for quantization and noise handling".
  - README.md:12–13 says it stabilises numeric updates with "stochastic resonance + subtractive dithering".
  - docs/ARCHITECTURE.md:7–8 says it "hardens floating-point arithmetic against quantization and signal-correlated error".
- **Noise normalisation.**
  - README.md:45–47 gives z = (n − μ)/(σ + ε).
  - The code differs: stochastic_resonance.py:61–64 and rdma_hooks.py:130–135 divide by σ only when σ > 1e-8 and add no ε.
- **Adaptive gain.**
  - README.md:65–71 gives the rule.
  - stochastic_resonance.py:10–16 sets the constants: base 0.02, min 0.005, max 0.30, stagnation threshold 1e-3, growth 1.15, decay 0.96.
  - stochastic_resonance.py:39–45 uses slope = |m_t − m_{t−1}|. The gain grows when the slope is below the threshold and decays otherwise.
- **Subtractive dither.**
  - subtractive_dither.py:10–22: quant_step = 1/4096, seed 42, uniform dither in ±Δ/2.
  - Output is `np.round((arr + dither)/Δ)*Δ − dither`.
- **SNR as coded.**
  - floating_core.py:60: `in_snr = self._snr_db(signal, signal - noise)`. The input-error term is `signal − noise`, not `noise`.
  - floating_core.py:61: `out_snr = self._snr_db(signal, signal - out)`.
  - floating_core.py:83–84: `_snr_db` adds 1e-12 to both powers. README.md:112 adds 10⁻¹² to the denominator only.
- **SDE and "Lyapunov" governor.**
  - aether_sde.py:12–45: `LyapunovGovernor` uses the same six constants and the same |slope| rule on a loss proxy.
  - aether_sde.py:69–123: an Euler–Maruyama step, optional dither, then a hard norm clamp to 8.0 (lines 114–116).
- **Noise source.**
  - rdma_hooks.py:1–6 and README.md:252–254: a "compatibility hook with simulation fallback".
  - The fallback fills the ring with `normal(0, 0.15)` (lines 60, 68). Default sample rate is 192 000 Hz (line 20).
- **In the wind engine.**
  - wind_engine.py:367–377: progress_metric = mean |divergence|. The layer is applied to u and v, and then the field is projected again.
  - wind_engine.py:341–343: turbulence is Gaussian forcing.
  - wind_engine.py:31–45: every config value is clipped to a range, e.g. turbulence 0–0.35 and dt 0.02–0.2.
- **Benchmarks.**
  - targets.py:56–58: in the stability benchmark "drift" is the discrete Laplacian of the state.
  - targets.py:59: the state update is s ← 0.92s + 0.08y.
- **Tests.**
  - test_architecture_core.py:22–31 asserts only that the gain does not decrease under stagnation.
  - The tests check that SNR and MSE keys exist and their types. No test asserts that SNR improves.
- **Rust prototype thresholds.**
  - rust/epsilon_prefetch/src/main.rs:153–154: `target_latency_met: p50 <= 15` (ns) and `target_hit_rate_met: hit_rate >= 0.95`.
  - The predictor is `predicted_next = current` (line 126).
- **Stated but not yet implemented.**
  - docs/ROADMAP.md:11: precision profiles for fp32/fp16/bf16.
  - docs/ROADMAP.md:13: "Publish error budgets per precision profile".
  - docs/ROADMAP.md:23: "Define acceptance thresholds for mean MSE, p95 stability norm, and SNR delta".
  - No such thresholds exist in the Python code.
- **Search of every non-.pyc text file at main.**
  - There are zero hits for "toleran", "calib", "jitter", "mismatch" or "dissonan".
  - "phase" appears only in the project name and in the roadmap's "Phase 1–4" headings.
  - No code handles signal phase.

### epsilon-cli
teerthsharma/epsilon-cli. Default branch **master**; last push 2026-05-12.

- **The GitHub description does not match the files.**
  - The description reads "Epsilon CLI — minimal error propagation and tolerance analysis tool."
  - README.md:3 and pyproject.toml:8 describe instead "Context Window Optimization via Stochastic Resonance — adaptive noise injection for LLM training".
- **What the text files contain.**
  - "toleran" appears in none of them.
  - "propagation" appears only at README.md:11 and resonance.py:5, both as "signal propagation through saturated layers".
  - None of the modules (cli.py, resonance.py, _types.py) contains error-propagation or tolerance-analysis code.
- **Gain rule.**
  - resonance.py:13–20 uses the same six constants as EPSILON-PHASE.
  - resonance.py:62–78 differs from EPSILON-PHASE in two ways. It uses a signed delta, with the metric "higher = better" (line 53). The gain grows only after at least 2 consecutive stagnant steps (line 67).
  - README.md:22–26 pseudocode shows neither the 2-step counter nor the signed delta.
- **Noise injection.**
  - resonance.py:83–94 computes values + g·N(0, I).
  - In the benchmark, resonance.py:141 sets step_size = 0.05·gain + 0.001, and resonance.py:142 subtracts 0.01·gain·noise.
- **Stated baseline with no baseline arm.** README.md:51 says the benchmark shows SR reaching lower final loss than a no-noise baseline. `run_optimization_benchmark` (resonance.py:105–149) has no no-noise arm.
- **README example output the code cannot print.**
  - README.md:102 shows a "step 200" line for `--steps 200`.
  - cli.py:43 loops `range(0, len(gain_schedule), 10)`. With 200 entries the last index printed is 190.
- **Tests.** test_resonance.py:59 asserts `final_loss < 1.0`.
