@echo off
setlocal
cd /d "%~dp0.."
if not exist "vendor\comfyui\nodes.py" (
  echo Cloning ComfyUI v0.32.0 ...
  git clone --depth 1 --branch v0.32.0 https://github.com/comfyanonymous/ComfyUI.git vendor\comfyui
)
if not exist "vendor\ultimatesd\nodes.py" (
  echo Cloning UltimateSDUpscale 627c871 ...
  git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git vendor\ultimatesd
  git -C vendor\ultimatesd checkout 627c871f14532b164331f08d0eebfbf7404161ee
)
git -C vendor\ultimatesd submodule update --init
if not exist "vendor\mickmumpitz-nodes\nodes\panorama_tools\nodes.py" (
  echo Cloning Mickmumpitz-Nodes main for panorama_tools ...
  git clone https://github.com/mickmumpitz/ComfyUI-Mickmumpitz-Nodes.git vendor\mickmumpitz-nodes
  git -C vendor\mickmumpitz-nodes checkout 4d5ff7c433884631599c3b2d82011a2cbbeea37e
)
if not exist "vendor\ostris-edit\nodes.py" (
  echo Cloning Ostris Krea2 Edit 7756566 ...
  git clone https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit.git vendor\ostris-edit
  git -C vendor\ostris-edit checkout 7756566160c4a1b24bb1bd9f0ff3ced1a83d7547
)
echo Pins: tools\vendor_pins.json
endlocal
