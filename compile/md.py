"""
Clean up extra latex and markdown stuff, e.g. the keyword arguments to markdown figures.

This is hella messy...will clean up
"""

import re
import markdown
from markdown.util import etree
from mdx_gfm import GithubFlavoredMarkdownExtension as GFM


def compile_markdown(md):
    """
    Compiles markdown to html.
    """
    return markdown.markdown(md, extensions=[GFM(), 'markdown.extensions.footnotes', MathJaxExtension(), FigureCaptionExtension()], lazy_ol=False)


"""
The below is from: <https://github.com/jdittrich/figureAltCaption>
(Not provided as a pypi package, so reproduced here)

Generates a Caption for Figures for each Image which stands alone in a paragraph,
similar to pandoc#s handling of images/figures

--------------------------------------------

Licensed under the GPL 2 (see LICENSE.md)

Copyright 2015 - Jan Dittrich by
building upon the markdown-figures Plugin by
Copyright 2013 - [Helder Correia](http://heldercorreia.com) (GPL2)
"""
from markdown.blockprocessors import BlockProcessor
from markdown.inlinepatterns import IMAGE_LINK_RE, IMAGE_REFERENCE_RE
FIGURES = [u'^\s*'+IMAGE_LINK_RE+u'\s*$', u'^\s*'+IMAGE_REFERENCE_RE+u'\s*$'] #is: linestart, any whitespace (even none), image, any whitespace (even none), line ends.

# This is the core part of the extension
class FigureCaptionProcessor(BlockProcessor):
    FIGURES_RE = re.compile('|'.join(f for f in FIGURES))
    def test(self, parent, block): # is the block relevant
        # Wenn es ein Bild gibt und das Bild alleine im paragraph ist, und das Bild nicht schon einen figure parent hat, returne True
        isImage = bool(self.FIGURES_RE.search(block))
        isOnlyOneLine = (len(block.splitlines())== 1)
        isInFigure = (parent.tag == 'figure')

        # print(block, isImage, isOnlyOneLine, isInFigure, "T,T,F")
        if (isImage and isOnlyOneLine and not isInFigure):
            return True
        else:
            return False

    def run(self, parent, blocks): # how to process the block?
        raw_block = blocks.pop(0)
        captionText = self.FIGURES_RE.search(raw_block).group(1)

        # create figure
        figure = etree.SubElement(parent, 'figure')

        # render image in figure
        figure.text = raw_block

        # create caption
        figcaptionElem = etree.SubElement(figure, 'figcaption')
        figcaptionElem.text = captionText #no clue why the text itself turns out as html again and not raw. Anyhow, it suits me, the blockparsers annoyingly wrapped everything into <p>.


class FigureCaptionExtension(markdown.Extension):
    def extendMarkdown(self, md, md_globals):
        """ Add an instance of FigcaptionProcessor to BlockParser. """
        md.parser.blockprocessors.add('figureAltcaption',
                                      FigureCaptionProcessor(md.parser),
                                      '<ulist')


class MathJaxProcessor(BlockProcessor):
    def test(self, parent, block): # is the block relevant
        return block.startswith('$$') and block.endswith('$$')

    def run(self, parent, blocks): # how to process the block?
        block = blocks.pop(0)
        element = markdown.util.etree.SubElement(parent, 'mathjax')
        element.text = markdown.util.AtomicString(block)


class MathJaxExtension(markdown.Extension):
    def extendMarkdown(self, md, md_globals):
        md.parser.blockprocessors.add('mathjax',
                                      MathJaxProcessor(md.parser),
                                      '<ulist')