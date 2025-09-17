# Quadspec
## Recovering signals from intensity, and integrated intensity of quadratic functions of spectrum
This repository will house the code-base and select worked examples of 
theory demonstrated in :
- D. Rosen and M.B. Wakin, [**Bivariate Retrieval from intensity of cross-correlation**](https://www.sciencedirect.com/science/article/pii/S0165168423003419), *Signal Processing*, vol 215, 109267, 2024.
- D. Rosen, D. Scarbrough, J. Squier, M.B. Wakin, [**Phase retrieval from integrated intensity of auto-convolution**](https://www.sciencedirect.com/science/article/pii/S0165168424000835), *Signal Processing*, vol 220, 109464, 2024.
- D. Rosen, [**Structured inverse problems in ultrafast optics**](https://mines.primo.exlibrisgroup.com/permalink/01COLSCHL_INST/1jb8klt/alma998214358002341), *Colorado School of Mines, Arthur Lakes Library*, 2024.

Test code is currently combined with the base code and will eventually be sectioned off on its own along with a few examples and sample datasets.
Currently the structure of the code is:
- measurements.py -- a collection of Numpy functions that replicate the mathematical operations of opticial pulse characterization setups, specifically systems that produce intensity of auto/cross-correlation/convolution (ICC, IAC) or integrated intensity of auto/cross-correlation/convolution (IIAC, IICC).
- gradient.py -- a collection of Wiritinger gradient tools for IAC,ICC,IIAC,IICC problems and their conversions into scipy minimize format. I have a few functions for computing Wiringer Hessian, but these are much less tested than gradient and require further development.
- inits.py -- a collection of alternating tensor thresholding tools that lift the IAC,ICC,IIAC,IICC problems to fourth order tensors and refine through iterative hard thresholding.

I will update this code periodically and will provide worked examples at some point on [**radioptimist.com**](https://radioptimist.com) under posts labeled "Pulse Characterization".
