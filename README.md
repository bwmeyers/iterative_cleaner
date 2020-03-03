# iterative_cleaner
RFI removal tool for pulsar archives.

Based on the surgical cleaner included in the `CoastGuard` pipeline by 
Patrick Lazarus, which can be found [here](https://github.com/plazar/coast_guard).
The original `CoastGuard` pipeline is described in 
[this paper](http://adsabs.harvard.edu/abs/2016MNRAS.458..868L).

The "surgical cleaner" of `CoastGuard` was altered to in order to be more useful 
for LOFAR scintillation studies by Lars Kuenkel (which is described 
[here](https://www2.physik.uni-bielefeld.de/fileadmin/user_upload/radio_astronomy/Publications/Masterarbeit_LarsKuenkel.pdf)). 
It has been further updated to work in Python 2 and 3, to be an installable package, 
and the template subtraction has been overhauled (though is still being developed).

For an unbiased cleaning approach, I recommend cleaning without template 
subtraction. It is possible to provide a 1D or 2D template for subtraction prior 
RFI mitigation, OR you can allow a template to be built iteratively from the data.
The template removal has not been robustly tested (and takes significantly more 
time to run). In future, a way of specifying user-defined channels/subintegrations 
to mask will be implemented.



---
## Installation
NOTE: This package requires a [psrchive](http://psrchive.sourceforge.net/) 
installation, including the Python interface.

To install:
```bash
git clone https://github.com/bwmeyers/iterative_cleaner.git
cd iterative_cleaner
git checkout chime-psr
pip install . --user
```

This will put the `iterative_cleaner` executable on your path, after which 
you can simply run 
```bash
iterative_cleaner -h
```
to view available options. Typically, running
```bash
iterative_cleaner --memory archive.ar
```
is sufficient, and will produce a log file (`iterative_cleaner_<datetime>.log`) 
and the cleaned archive (`*_clean.ar`).