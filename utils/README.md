
### Scraping WikiArt (thanks to [@robbiebarrat](http://github.com/robbiebarrat/))

See https://www.wikiart.org/en/paintings-by-genre/ for list of genres. List follows:

**>1000 images**: portrait, landscape, genre-painting, abstract, religious-painting, cityscape, sketch-and-study, figurative, illustration, still-life, design, nude-painting-nu, mythological-painting, marina, animal-painting, flower-painting, self-portrait, installation, photo, allegorical-painting, history-painting, 

**<1000 images**: interior, literary-painting, poster, caricature, battle-painting, wildlife-painting, cloudscape, miniature, veduta, yakusha-e, calligraphy, graffiti, tessellation, capriccio, advertisement, bird-and-flower-painting, performance, bijinga, pastorale, trompe-loeil, vanitas, shan-shui, tapestry, mosaic, quadratura, panorama, architecture


See https://www.wikiart.org/en/paintings-by-styles/ for list of genres. List follows:

impressionism, realism, romanticism, expressionism, post-impressionism, surrealism, art-nouveau, baroque, symbolism, abstract-expressionism, na-ve-art-primitivism, neoclassicism, cubism, rococo, northern-renaissance, pop-art, minimalism, abstract-art, art-informel, ukiyo-e, conceptual-art, color-field-painting, high-renaissance


Example:

    python scrape_wikiart.py --genre landscape --num_pages 3 --output_dir ../datasets
    python scrape_wikiart.py --style impressionism --num_pages 3 --output_dir ../datasets
    
    
### Crop and resize folder of images

documentation needed... see argparse.

    python dataset_utils.py 
