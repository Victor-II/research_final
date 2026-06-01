TEX = pdflatex
BIB = bibtex
MAIN = dissertation

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex bibliographyDizertatie.bib
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all clean distclean
