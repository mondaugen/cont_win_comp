#! /bin/bash

# Builds the thesis.tex file complete with bibliography.

# Convert all eps files to pdf

pdflatex poster.tex
BIBINPUTS=. bibtex poster.aux
pdflatex poster.tex
pdflatex poster.tex
mv poster.pdf Esterer_Nicholas_dafx_2017_poster.pdf
#dvipdf thesis.dvi
