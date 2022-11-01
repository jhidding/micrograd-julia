.PHONY: site clean watch watch-pandoc watch-browser-sync

theme := default
theme_dir := .entangled/templates/$(theme)

pandoc_args += -s -t html5 -f markdown+multiline_tables+simple_tables --toc --toc-depth 2
pandoc_args += --template $(theme_dir)/template.html
pandoc_args += --css theme.css
pandoc_args += --filter pandoc-eqnos --filter pandoc-fignos
pandoc_args += --mathjax
pandoc_args += --section-divs
pandoc_args += --lua-filter .entangled/scripts/hide.lua
pandoc_args += --lua-filter .entangled/scripts/annotate.lua
pandoc_args += --lua-filter .entangled/scripts/make.lua
pandoc_args += --citeproc
pandoc_args += --shift-heading-level-by=-1
pandoc_args += --metadata author="Johan Hidding (after Andrej Karpathy)" \
               --metadata subtitle="introduction to automatic differentiation and backpropagation"

pandoc_input := README.md
pandoc_output := docs/index.html

static_files := $(theme_dir)/theme.css $(theme_dir)/static
static_targets := $(static_files:$(theme_dir)/%=docs/%)
functional_deps := Makefile $(wildcard .entangled/scripts/*.lua) $(theme_dir)/template.html $(theme_dir)/syntax.theme

site: $(pandoc_output) $(static_targets) $(figure_targets)

clean:
	rm -rf docs

$(static_targets): docs/%: $(theme_dir)/%
	@mkdir -p $(@D)
	rm -rf $@
	cp -r $< $@

docs/index.html: README.md $(functional_deps)
	@mkdir -p $(@D)
	pandoc $(pandoc_args) -o $@ $<

# docs/%.html: lit/%.md $(functional_deps)
# 	@mkdir -p $(@D)
# 	pandoc $(pandoc_args) -o $@ $<

# Starts a tmux with Entangled, Browser-sync and an Inotify loop for running
# Pandoc.
watch:
	@tmux new-session make --no-print-directory watch-pandoc \; \
		split-window -v make --no-print-directory watch-browser-sync \; \
		split-window -v entangled daemon \; \
		select-layout even-vertical \;

watch-pandoc:
	@while true; do \
		inotifywait -e close_write -r .entangled Makefile README.md src; \
		make site; \
	done

watch-browser-sync:
	browser-sync start -w -s docs

