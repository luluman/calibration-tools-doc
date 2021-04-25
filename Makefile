# You can set these variables from the command line.

LANG=zh

ifeq ($(LANG), en)
  SOURCEDIR     = source_en
  pdf_name      = Calibration-Tools_en.pdf
else ifeq($(LANG), zh)
  SOURCEDIR     = source_zh
  pdf_name      = Calibration-Tools_zh.pdf
else
  exit 1
endif

#echo $SOURCEDIR
#exit 1

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

pdf: latex
	@cd $(BUILDDIR)/latex && xelatex Quantization-Tools.tex
	#@mv $(BUILDDIR)/latex/*.pdf $(BUILDDIR) #&& rm -rf $(BUILDDIR)/latex
	@cd $(BUILDDIR)/latex && xelatex Quantization-Tools.tex
	@mv $(BUILDDIR)/latex/Quantization-Tools.pdf $(BUILDDIR)/"${pdf_name}" && rm -rf $(BUILDDIR)/latex

web: html
	#@python3 -m http.server --directory build/html

clean:
	@rm -rf $(BUILDDIR)
