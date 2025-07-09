# Usage
# -----
#
# Initialize the notebooks with `make all`.
# Any update to the `.py` or `*.ipynb` files can be
# synchronized by `make sync`. Synchronization works both ways,
# edits on .ipynb will reflect in paired .py, and vice-versa.

# Define source directory and output directory
PY_FILES := $(wildcard ch[0-9]*.py) $(wildcard demo[0-9]*.py) index.py
IPYNB_FILES := $(patsubst %.py,%.ipynb,$(PY_FILES))
PAGES_SUBDIR := ../pages/sealir_tutorials
PRE_RENDERED_HTML_FILES := $(wildcard *.pre_rendered.html)
HTML_FILES := $(patsubst %.ipynb,$(PAGES_SUBDIR)/%.html,$(IPYNB_FILES)) $(patsubst %.pre_rendered.html,$(PAGES_SUBDIR)/%.pre_rendered.html,$(PRE_RENDERED_HTML_FILES))

# Default target to build all notebooks
all: $(IPYNB_FILES)

# Synchronize paired py-ipynb files
sync:
	jupytext --sync *.ipynb

# Run auto formatter (only tracked files)
format:
	@echo "Formatting tracked Python files..."
	@files="$$(git ls-files '*.py' 2>/dev/null || echo '')"; \
	if [ -n "$$files" ]; then \
		echo "$$files"; \
		echo "$$files" | xargs isort -l79 --profile black; \
		echo "$$files" | xargs black -l79; \
	fi

pages: $(PAGES_SUBDIR) $(HTML_FILES)

$(PAGES_SUBDIR):
	mkdir -p $(PAGES_SUBDIR)

# Pattern rule to convert .py files to .ipynb
%.ipynb: %.py
	jupytext --update --to ipynb --from py:light $< -o $@

# Pattern rule to convert .ipynb files to rendered .html
$(PAGES_SUBDIR)/%.html: %.ipynb
	jupyter nbconvert --execute --to html  --HTMLExporter.theme=dark --output=$@ $<

# Pattern rule to copy pre-rendered .html files
$(PAGES_SUBDIR)/%.pre_rendered.html: %.pre_rendered.html
	cp $< $@

# Clean target to remove generated notebooks and rendered HTML
clean:
	rm -f $(IPYNB_FILES)
	rm -f $(HTML_FILES)
	rm -df $(PAGES_SUBDIR)

test:
	pytest -n auto tests

.PHONY: all sync dump test
