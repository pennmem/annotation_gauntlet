# annotation_gauntlet
Annotation training for the Computational Memory Lab

## Set up environment
Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) (or miniconda) to manage python environemnts. 

Run `conda create -n gauntlet python=2.7 numpy matplotlib` followed by `conda activate gauntlet`.

## Annotation Gauntlet Instructions

1)	 Clone this repository to your local machine

2)	 Within this newly created "annotation_gauntlet" folder on your local computer, annotate the "session_0" folder as you normally would using [PennTotalRecall](https://memory.psych.upenn.edu/TotalRecall).

3)	 Once you have finished annotating, double-click the verify_annotation script in the folder on your local computer (or run `bash verify_annotation` on command line). This will produce an output comparing your performance to that of the master file ("session_0_QC").

4)	 In order to review this performance output, go back through the annotations from your "session_0" folder and identify the discrepancies, trying to understand why your annotation marks do not align with the master file. If you can't identify or don't agree with your performance output, try consulting a more experienced annotator. If after consulting someone you still have questions about the scoring, feel free to email your study contact at Penn so they can address your concerns.

