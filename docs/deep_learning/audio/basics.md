---
layout: page
title: Audio Basics
permalink: /deep_learning/audio/basics/
---

#### Reference:
- [MEL Frequency](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)



**What Does the Unit kHz Mean in Digital Music?** 

kHz is short for kilohertz, and is a measurement of frequency (cycles per second). In digital audio, this measurement describes the number of data chunks used per second to represent an analog sound in digital form. These data chunks are known as the sampling rate or sampling frequency.

This definition is often confused with another popular term in digital audio, called [bitrate](https://www.lifewire.com/what-is-mp3-bitrate-2438538) (measured in kbps). However, the difference between these two terms is that bitrate measures how much is sampled every second (size of the chunks) rather than the number of chunks (frequency).

**Note:** kHz is sometimes referred to as *sampling rate*, *sampling interval*, or *cycles per second*.



**What is the Mel scale?**

The Mel scale relates perceived frequency, or pitch,
of a pure tone to its actual measured frequency.
Humans are much better at discerning small changes
in pitch at low frequencies than they are at high frequencies.
Incorporating this scale makes our features match more closely
what humans hear.

The formula for converting from frequency to Mel scale is:

$$
M(f) = 1125 *  \ln(1 + \frac{f}{700}) \\

M^{-1}(m) = 700 *  (\exp(\frac{m}{1125}) -1)
$$

**Audio Features**

We start with a speech signal, we'll assume sampled at 16kHz.

Frame the signal into 20-40 ms frames. 25ms is standard. 
This means the frame length for a 16kHz signal is 0.025*16000 = 400 samples. 
Frame step is usually something like 10ms (160 samples), which allows some overlap to the frames. 
The first 400 sample frame starts at sample 0, the next 400 sample frame starts at sample 160 etc.
until the end of the speech file is reached. If the speech file does not divide into an even 
number of frames, pad it with zeros so that it does.

Audio Signal File : 0 to N seconds

Audio Frame : Interval of 20 - 40 ms ---> default 25 ms ---> 0.025 * 16000 = 400 samples

Frame step : Default 10 ms ---> 0.010 * 16000 ---> 160 samples

First sample: 0 to 400 samples

Second sample: 160 to 560 samples etc.,

​       25ms	   25ms		25ms	  25ms ...  	

​	400		    400		400	           400  ...

|---------------|---------------|---------------|---------------|---------------|---------------|---------------|-------------|    

|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

 10   10 10 10 ...

Still dont get it? Consider the audio signal to be a time series sampled at an interval of 25ms with step size of 10ms


Check out this jupyet notebook @ [https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html](https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html)

Forked version @ [https://github.com/dhiraa/python_spectrograms_and_inversion](https://github.com/dhiraa/python_spectrograms_and_inversion)