This code uses a semi-analytic approach to compute the velocity distribution of light dark matter after propagation through the atmosphere or an overburden. Please note that it is a work in progress.

To use the code, just download and run the script "prop.py", followed by 5 input parameters: whether to use crust or atmospheric shielding (1 for crust, 2 for atmosphere), the DM mass in GeV, Log_10 of the DM-nucleon cross section in units of cm^2, the detector depth in meters, and the number of convolutions to perform. The example input for a run through 100 collisions, with a dark matter mass of 100 MeV and DM-nucleon cross section of 10^-30 cm^2, for a detector 1400 meters underground, is shown below:

% python3 prop.py 1 .1 -30 1400 100

While running, the script saves energy distributions as text files in the directories "depth" and "energy." This speeds up subsequent runs, in that if you want to run the script for several cross sections at the same mass, the energy distribution only needs to be computed once. It also means that if you forget to save the results of a run, they will still be there and the code can reproduce the figures and event rates quickly. However, be aware that this can lead to a very large number of files being saved if you do not preiodically delete them.


NOTE: When running at cross sections much larger than direct detection ceilings, the Fast Fourier Transform convolution method can produce unphysical results. In this case, the "method" of the colvolution function may be set to "direct", which slows down the code but is more reliable. 
