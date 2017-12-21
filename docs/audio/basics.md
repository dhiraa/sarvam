---
layout: page
title: Audio Basics
permalink: /audio/basics/
---

#### Reference:

- http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/



### What is the Mel scale?

The Mel scale relates perceived frequency, or pitch,
of a pure tone to its actual measured frequency.
Humans are much better at discerning small changes
in pitch at low frequencies than they are at high frequencies.
Incorporating this scale makes our features match more closely
what humans hear.

The formula for converting from frequency to Mel scale is:

$$
M(f) = 1125 ln(1 + f/700) \\

M^-1(m) = 700 (\exp(m/1125) -1)
$$

### Audio Features

Audio Signal File : 0 to N seconds

To frames : Interval of 20 - 40 ms ---> default 25 ms ---> 0.025 * 16000 = 400 samples

Frame step : Default 10 ms ---> 160 samples

First sample: 0 to 400 samples

Second sample: 160 to 560 samples etc.,