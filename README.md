
# lmmvar

An R package for computing test statistics and confidence intervals for
a variance component or proportion in a linear mixed model using the
method described in ["Fast and reliable confidence intervals for a
variance component"](https://arxiv.org/abs/2404.15060).

This package will be updated sporadically. Please contact
<yiqiaozhang@ufl.edu> with any questions or comments.

### Installation

lmmvar can be loaded into R through the `devtools` package:

``` r
# install.packages("devtools")
# library(devtools)
# devtools::install_github("yqzhang5972/lmmvar")
```

### Citation instructions

Please cite the most recent version of the article mentioned above. As
of April 2024, this was the following (in bibtex):

    @misc{zhang2024fast,
          title={Fast and reliable confidence intervals for a variance component or proportion}, 
          author={Yiqiao Zhang and Karl Oskar Ekvall and Aaron J. Molstad},
          year={2024},
          eprint={2404.15060},
          archivePrefix={arXiv},
          primaryClass={stat.ME}
    }

### Vignette

Please visit [this example
page](http://koekvall.github.io/files/lmmvar-vignette.html) for details
on implementation and usage.

### Reproducing simulation study results

Code to reproduce simulation results from the article can be found at
[this repository](https://github.com/koekvall/varcomp-suppl).
