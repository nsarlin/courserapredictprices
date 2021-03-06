.PHONY: clean dataclean data datasample train trainsample lint sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = courserapredictprices
PYTHON_INTERPRETER = python3

DATA_TARG = data/processed/X_train.npz
DATA_SMPL_TARG = data/processed_smpl/X_train.npz
TRAIN_TARG = models/dnn.h5 models/xgb.bin
TRAIN_SMPL_TARG = models_smpl/dnn.h5 models_smpl/xgb.bin
DATA_SRC = src/data/prepare.py src/data/make_dataset.py
COMMON_SRC = src/common.py
TRAIN_SRC = src/models/train_model.py
DNN_SRC = src/models/dnn.py
XGB_SRC = src/models/xgb.py

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.requirements: .test_environment requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	touch .requirements

## Make Dataset
data: $(DATA_TARG)
datasample: $(DATA_SMPL_TARG)
$(DATA_TARG): .requirements $(DATA_SRC) $(COMMON_SRC)
	$(PYTHON_INTERPRETER) src/data/make_dataset.py -i data/interim data/raw data/processed
$(DATA_SMPL_TARG): .requirements $(DATA_SRC) $(COMMON_SRC)
	$(PYTHON_INTERPRETER) src/data/make_dataset.py --sample -i data/interim_smpl data/raw data/processed_smpl

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

dataclean:
	rm -f data/processed/X_train.npz
	rm -f data/processed_smpl/X_train.npz

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
.test_environment: test_environment.py
	$(PYTHON_INTERPRETER) test_environment.py
	touch .test_environment

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
train: $(TRAIN_TARG) $(TRAIN_SRC)
trainsample: $(TRAIN_SMPL_TARG) $(TRAIN_SRC)
models/dnn.h5: $(DATA_TARG) $(DNN_SRC) $(COMMON_SRC)
	optirun $(PYTHON_INTERPRETER) src/models/train_model.py dnn data/processed models
models_smpl/dnn.h5: $(DATA_SMPL_TARG) $(DNN_SRC) $(COMMON_SRC)
	optirun $(PYTHON_INTERPRETER) src/models/train_model.py dnn data/processed_smpl models_smpl
models/xgb.bin: $(DATA_TARG) $(XGB_SRC) $(COMMON_SRC)
	$(PYTHON_INTERPRETER) src/models/train_model.py xgb data/processed models
models_smpl/xgb.bin: $(DATA_SMPL_TARG) $(XGB_SRC) $(COMMON_SRC)
	$(PYTHON_INTERPRETER) src/models/train_model.py xgb data/processed_smpl models_smpl



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
