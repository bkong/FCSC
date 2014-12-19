This code implements the algorithm by Bristow et al. in their CVPR 2013 paper titled: "Fast Convolutional Sparse Coding". We have a tech report that corrects a number of errors in the original paper and discuss implementation details of the algorithm. This report can be found <a href="http://baileykong.com/files/KongF_UCI_2014.pdf">here</a>.

If you're looking for help in implementing the algorithm in your preferred programming language, take a look at fcsc_simple.jl first. The code there is mostly self-contained and implements the algorithm in the most straightforward way, but as a consequence it does not scale up well.

If you use this code in a publication, we would be grateful if you cite:
```
@techreport{KongF_UCI_2014,
  author =      {Bailey Kong and Charless C. Fowlkes},
  title =       {Fast Convolutional Sparse Coding},
  institution = {Department of Computer Science, University of California, Irvine},
  year  =       {2014},
}
```
