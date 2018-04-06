

### Scraping WikiArt (thanks to [@robbiebarrat](http://github.com/robbiebarrat/))

See https://www.wikiart.org/en/paintings-by-genre/ for list of genres. List follows:

**>1000 images**: portrait, landscape, genre-painting, abstract, religious-painting, cityscape, sketch-and-study, figurative, illustration, still-life, design, nude-painting-nu, mythological-painting, marina, animal-painting, flower-painting, self-portrait, installation, photo, allegorical-painting, history-painting, 

**<1000 images**: interior, literary-painting, poster, caricature, battle-painting, wildlife-painting, cloudscape, miniature, veduta, yakusha-e, calligraphy, graffiti, tessellation, capriccio, advertisement, bird-and-flower-painting, performance, bijinga, pastorale, trompe-loeil, vanitas, shan-shui, tapestry, mosaic, quadratura, panorama, architecture

Example:

    python scrape_wikiart.py --genre landscape --num_pages 3 --output_dir ../datasets
    
    
### Crop and resize folder of images

The following example will take all images in `../datasets/landscape`, take a random square crop, where the side length is half the size of the original (`frac` parameter), resize to 128x128, and place in new directory `../datasets/landscape_small`.

    python dataset_utils.py --input_dir ../datasets/landscape --output_dir ../datasets/landscape_small --frac 0.5 --w 128 --h 128
