# Usage
# -----
#
# Initialize the notebooks with `make all`.
# Any update to the `.py` or `*.ipynb` files can be
# synchronized by `make sync`. Synchronization works both ways,
# edits on .ipynb will reflect in paired .py, and vice-versa.

# Define source directory and output directory
PY_FILES := $(wildcard [ch,demo]*.py)
IPYNB_FILES := $(patsubst %.py,%.ipynb,$(PY_FILES))
PAGES_SUBDIR := ../pages/sealir_tutorials
HTML_FILES := $(patsubst %.ipynb,$(PAGES_SUBDIR)/%.html,$(IPYNB_FILES))

# Default target to build all notebooks
all: $(IPYNB_FILES)

# Synchronize paired py-ipynb files
sync:
	jupytext --sync *.ipynb

# Run auto formatter
format:
	isort -l79 --profile black .
	black -l79 .

pages: $(HTML_FILES)

$(PAGES_SUBDIR):
	mkdir -p $(PAGES_SUBDIR)

# Pattern rule to convert .py files to .ipynb
%.ipynb: %.py
	jupytext --update --to ipynb --from py:light $< -o $@

# Pattern rule to convert .ipynb files to rendered .html
$(PAGES_SUBDIR)/%.html: %.ipynb $(PAGES_SUBDIR)
	jupyter nbconvert --execute --to html --output=$@ $<

# Clean target to remove generated notebooks and rendered HTML
clean:
	rm -f $(IPYNB_FILES)
	rm -f $(HTML_FILES)
	rm -df $(PAGES_SUBDIR)

test:
	pytest -n auto tests

.PHONY: all sync dump test
