# Analysis plan (written before looking at the data)

Written on 27 October 2026, before any eye-movement data from EMIP was loaded or inspected. Changes made after this point are listed at the end of this file with a reason.

## Data

EMIP, the Eye Movements in Programming dataset (Bednarik et al., 2020): 216 participants reading two short programs (Rectangle and Vehicle) in Java, Python or Scala, recorded with an SMI RED250 tracker, plus a background questionnaire with self-rated programming expertise and years of experience. Licence CC-BY 4.0.

## Study 1: does predictability shape how code is read?

**Unit of analysis.** One code token (whitespace-separated, as in the EMIP Toolkit; Al Madi et al., 2021) read by one participant in one trial.

**Dependent variables.** First-pass gaze duration and total reading time on the token (ms), log-transformed; skipping (token not fixated) as a binary outcome.

**Predictor of interest.** Token surprisal, −log₂ p(token | preceding code), computed with a small open code language model and summed over the model's sub-word pieces of each token.

**Controls.** Token length in characters, line number, position in line, and log token frequency within the program set.

**Hypotheses.**
- **H1.** Higher surprisal predicts longer gaze duration and total reading time, over and above the controls (as for text; Demberg and Keller, 2008).
- **H2.** The surprisal effect differs by expertise: we expect it to be weaker for high-expertise readers, who rely less on token-by-token prediction and more on structure. The direction is not certain; a stronger effect would also be reported.
- **H3.** Surprisal reduces the probability of skipping a token.

**Model.** Linear mixed-effects regression with random intercepts for participants and for tokens within programs (statsmodels MixedLM); random slopes for surprisal by participant if the model converges. Expertise enters as the self-rating and as a median split for plots. Significance at α = .05, two-sided; effect sizes reported in ms per bit at the mean.

**Exclusions.** Trials with tracking loss above 30 %, fixations shorter than 80 ms or longer than 1,200 ms, and participants whose fixations fall mostly outside the code area (calibration failure) are excluded. All exclusion counts are reported.

## Study 1b: interpretable reader types

Per trial, compute reading-strategy features: linearity of reading order (vertical next-line rate), regression rate, share of time on method signatures vs. bodies, mean fixation duration, and coverage. Standardise, cluster with k-means for k = 2…5 chosen by silhouette, and describe each cluster by its centroid in plain language. Test whether cluster membership is associated with expertise (χ² or Kruskal–Wallis). This is exploratory.

## Study 2: profile-conditioned explanations and their faithfulness

A developer profile lists programming constructs of the stimulus code as known or unknown. An LLM proposes several candidate explanations; a Rational Speech Acts style listener built from the profile scores them (Goodman and Frank, 2016), with a cost for length. **Faithfulness test:** flipping one construct from unknown to known should change the explanation of the lines using that construct and leave other lines unchanged. Measured as (a) the change rate on targeted lines and (b) the change rate on untargeted lines, over all single-construct flips. No LLM is used as a judge of quality.

## Reporting

All hypotheses are reported whether supported or not. Code, processed data and figures are public.

## Changes after data inspection

1. **Extra exclusion rule (added while building the pipeline, before fitting any model).** Trials with fewer than 30 detected fixations were also excluded (33 trials), because the dispersion-based detector finds almost no fixations in very noisy recordings. Plan-conform exclusions: tracking loss above 30 % (21 trials) and gaze mostly off the code (9 trials).
2. **Only Java trials analysed.** 207 of the 216 participants read Java; the 9 Python and Scala readers are too few for a separate model.
3. **Random slopes not fitted.** The models with random intercepts for participants and token items converged; random slopes for surprisal were not attempted, to keep the analysis within the time available. This is a limitation.
4. **Robustness check added after the main results.** H1 (total reading time) refitted separately for each program.
5. **Study 2 thresholds.** Two explanations of a line count as unchanged when their text similarity is at least 0.6 (difflib ratio); this threshold was set before running Study 2. Prompt-only run-to-run noise was 2.5 % in the first full run and 0 % in the second; the reported numbers are from the second run.
6. **Study 2 candidate set made concrete.** Instead of several free-form candidate explanations, the language model writes one note per (line, construct) pair once, and the speaker chooses among subsets of these notes per line. This keeps content selection separate from wording, which is what the faithfulness test is meant to probe.
7. **Random-effects structure corrected (27 October 2026).** The first Study 1 models were fitted with statsmodels `groups=participant` and the token item as a variance component, which nests items within participants instead of crossing them. All models were refitted with crossed random intercepts (one constant group, participants and items as variance components), and a by-item regression was added as a check. The corrected H1 result is weaker than first reported: p = .004 instead of p < .001, the by-item check is not significant (p = .16), and neither program alone reaches significance (p = .07, .10). All reported numbers are the corrected ones.

---

# Addendum (27 October 2026): extension to the open EEG data of Bergum et al. (2026)

Written before any of the analyses below were run. At this point we had read the published papers (Bergum et al., 2026; Abdelsalam et al., 2025) and their published snippet-level results, and we had checked that our own perplexity code reproduces theirs for deepseek-coder-1.3b-base (Pearson r = 1.00 over 144 files). We had not yet loaded single-trial EEG, reading metrics or participant data.

**Data.** Bergum et al. (2026), Zenodo 10.5281/zenodo.14229848, CC-BY 4.0: 24 programmers, 72 pairs of Java snippets with and without an atom of confusion, single-trial fixation-related potentials (32 channels, −0.3 to 1.0 s) time-locked to the first fixation on the human-annotated area of interest (AOI), first-pass dwell time in the AOI, and a questionnaire with years of programming experience. AOI perplexity per snippet from the replication package of Abdelsalam et al. (2025) for six models, plus our own runs of Qwen2.5-Coder 0.5B, 1.5B and 3B with the same method.

Each study below targets a gap stated by these papers.

**Study 3 – model size (their future work: is there a turning point where larger models align less?).** For each model, Spearman correlation between maximum AOI perplexity and the snippet-mean late frontal positivity (Fz, 400–800 ms) for confusing snippets, as in their RQ2. H5: alignment does not increase monotonically with size; within the Qwen2.5-Coder family (0.5B to 32B) the smaller models fit at least as well as the 32B model. Exploratory, no significance test of the trend with only four sizes.

**Study 4 – single trials and individual differences (their limitation: individual differences not analysed, sample of intermediate programmers).** Linear mixed-effects models on single trials with random intercepts for participants and snippets: H6: AOI surprisal (log₂ of maximum AOI perplexity, Qwen2.5-Coder-32B) predicts single-trial late frontal positivity amplitude at Fz, 400–800 ms. H7: the effect is modulated by years of programming experience (direction open). H8 (their future work on other components): surprisal also predicts amplitude in an N400 window (300–500 ms, mean of Cz, CPz, Pz) and a P600 window (600–900 ms, mean of CPz, Pz, P3, P4); direction open, reported either way. H9 (eye-tracking measures not analysed inferentially): AOI surprisal predicts longer first-pass dwell time in the AOI (log seconds), also modulated by experience.

**Study 5 – detecting confusing code beyond perplexity (their detection: 64 % of atoms).** Classify confusing vs clean snippets with leave-one-pair-out logistic regression from (a) AOI perplexity, (b) mean first-pass dwell across readers, (c) both. H10: combining gaze and perplexity gives a higher area under the ROC curve than perplexity alone.

**Study 6 – a listener learned from human data (open point in RSA work: listener models set by hand).** On EMIP, estimate per programming construct how much longer lines with that construct are read by lower- versus higher-expertise readers (residual total reading time after the controls of Study 1), turn this into a probability that a construct is unknown, and use it as the RSA listener in place of hand-set values. Report how the selected notes differ for a novice and an expert profile, and re-run the faithfulness test. Exploratory.

Exclusions follow the original authors' (artifact-rejected epochs are already removed in the published files). α = .05, two-sided.

## Changes to the addendum

1. **Study 3: Qwen2.5-Coder-3B not included.** Planned sizes were 0.5B, 1.5B and 3B; the 3B run in bfloat16 did not finish on the available CPU within the time available and was stopped. Study 3 therefore uses Qwen2.5-Coder 0.5B and 1.5B (ours) and the six published models, including Qwen2.5-Coder-32B.
2. **Study 3: one single-trial fit failed.** For Artigenz-Coder-DS-6.7B the likelihood-ratio gain came out negative, which indicates a failed optimisation; the value is reported but not interpreted.
3. **Study 4: experience main effects.** The planned tests were the surprisal effects and their interaction with experience; main effects of experience are reported as exploratory (24 participants).
