import os
import shutil
from glob import glob
from jinja2 import Template
from md import compile_markdown

# load jinja template
here = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = 'build'
with open(os.path.join(here, 'template.html'), 'r') as f:
    tmpl = Template(f.read())

# prep output directory
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)
shutil.copy(os.path.join(here, 'style.css'),
            os.path.join(OUT_DIR, 'style.css'))
shutil.copytree('assets', os.path.join(OUT_DIR, 'assets'))

# compile files
for file in glob('*.md'):
    raw = open(file, 'r').read()
    body = compile_markdown(raw)
    html = tmpl.render(body=body)
    fname = file.replace('.md', '.html')
    with open(os.path.join(OUT_DIR, fname), 'w') as f:
        f.write(html)

print('Compiled')

# make ipython notebooks TODO
# from IPython.nbformat import current as nbf
# nb = nbf.new_notebook()