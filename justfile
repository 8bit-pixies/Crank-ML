render: && clean_lightning_logs
	@quarto render examples/classification/feature_discretization.qmd
	@cp examples/classification/feature_discretization.md docs/preprocessing/feature_discretization.md

render_paper:
	@quarto render paper/paper.qmd

lint:
	@isort . --check-only 
	@ruff check . --fix
	@black --check .
	@echo "lint finished"

format:
	@isort .
	@black .

clean_lightning_logs:
	@rm -r -f lightning_logs
	@rm -r -f examples/*/lightning_logs

archive:
	git archive HEAD -o ${PWD##*/}.zip