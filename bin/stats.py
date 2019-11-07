# Print the location of Jupyter's config directory
from jupyter_core.paths import jupyter_config_dir
jupyter_dir = jupyter_config_dir()
print(f'jupyter_dir: {jupyter_dir}')

# Print the location of custom.js
import os.path
custom_js_path = os.path.join(jupyter_dir, 'custom', 'custom.js')
print(f'custom_js_path: {custom_js_path}')

# Print the contents of custom.js, if it exists.
if os.path.isfile(custom_js_path):
    with open(custom_js_path) as f:
        print(f.read())
else:
    print("You don't have a custom.js file")