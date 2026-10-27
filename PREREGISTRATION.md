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
