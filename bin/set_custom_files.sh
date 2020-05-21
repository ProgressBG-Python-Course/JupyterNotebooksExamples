#!/bin/bash

# echo $PWD

CUSTOM_CSS_FILE="$PWD/bin/custom.css"

JUPY_CSS_FILE="$PWD/.venv/lib/python3.7/site-packages/notebook/static/custom/custom.css"
if [ -f $JUPY_CSS_FILE ]; then
  # backup
  mv $JUPY_CSS_FILE "$JUPY_CSS_FILE.ori"
  # replace
  cp $CUSTOM_CSS_FILE $JUPY_CSS_FILE
else
   echo "File $JUPY_CSS_FILE does not exist."
fi



CUSTOM_JS_FILE="$PWD/bin/custom.js"

JUPY_JS_FILE="$PWD/.venv/lib/python3.7/site-packages/notebook/static/custom/custom.js"
if [ -f $JUPY_JS_FILE ]; then
  # backup
  mv $JUPY_JS_FILE "$JUPY_JS_FILE.ori"
  # replace
  cp $CUSTOM_JS_FILE $JUPY_JS_FILE
else
   echo "File $JUPY_JS_FILE does not exist."
fi