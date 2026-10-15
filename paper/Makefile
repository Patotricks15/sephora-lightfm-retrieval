# Build the technical report.
#
# Requires a LaTeX toolchain. If one is not installed locally, use the
# containerised target instead:
#
#     make docker
#
# Tables under tables/ are generated from the code; regenerate them with
# `make tables` before building so the report cannot drift from the results.

MAIN    := main
LATEX   := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX  := bibtex
PYTHON  ?= python

.PHONY: all tables docker clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(wildcard sections/*.tex) $(wildcard tables/*.tex) references.bib
	$(LATEX) $(MAIN)
	-$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

tables:
	cd .. && $(PYTHON) scripts/export_results_table.py
	cd .. && $(PYTHON) scripts/export_feature_tables.py

docker:
	docker run --rm -v "$(CURDIR)":/work -w /work texlive/texlive:latest \
	  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc \
	      $(MAIN).bbl $(MAIN).blg $(MAIN).fls $(MAIN).fdb_latexmk

distclean: clean
	rm -f $(MAIN).pdf
