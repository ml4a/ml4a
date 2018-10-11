need
 - pip install --upgrade git+https://github.com/tensorpack/tensorpack.git
 - `wget http://models.tensorpack.com/HED/HED_reproduced.npz ../data/.`

### Scraping WikiArt (thanks to [@robbiebarrat](http://github.com/robbiebarrat/))

See https://www.wikiart.org/en/paintings-by-genre/ for list of genres. List follows:

**>1000 images**: portrait, landscape, genre-painting, abstract, religious-painting, cityscape, sketch-and-study, figurative, illustration, still-life, design, nude-painting-nu, mythological-painting, marina, animal-painting, flower-painting, self-portrait, installation, photo, allegorical-painting, history-painting, 

**<1000 images**: interior, literary-painting, poster, caricature, battle-painting, wildlife-painting, cloudscape, miniature, veduta, yakusha-e, calligraphy, graffiti, tessellation, capriccio, advertisement, bird-and-flower-painting, performance, bijinga, pastorale, trompe-loeil, vanitas, shan-shui, tapestry, mosaic, quadratura, panorama, architecture


See https://www.wikiart.org/en/paintings-by-styles/ for list of styles. List follows:

impressionism, realism, romanticism, expressionism, post-impressionism, surrealism, art-nouveau, baroque, symbolism, abstract-expressionism, na-ve-art-primitivism, neoclassicism, cubism, rococo, northern-renaissance, pop-art, minimalism, abstract-art, art-informel, ukiyo-e, conceptual-art, color-field-painting, high-renaissance

Example:

    python scrape_wikiart.py --genre landscape --num_pages 3 --output_dir ../datasets
    python scrape_wikiart.py --style impressionism --num_pages 3 --output_dir ../datasets
    
    
### Dataset utils

Example:

This script will take `--num_images` images from `--input_dir` (all if omitted), make `--num_augment` copies of it rotated by random angle up to max `--max_ang`, random crop of  `--frac +/- --frac_vary` %, resized to `--w` x `--h`, saved to `--output_dir`.

    python3 dataset_utils.py --input_dir ../datasets/portrait/ --output_dir ../datasets/portrait_1024 --augment 1 --num_augment 4 --action simplify --frac 0.75 --frac_vary 0.075 --max_ang 4 --w 1024 --h 1024 --split 0 --pct_train 1.0 --combine 0 --num_images 10




python3 dataset_utils.py --input_dir ../datasets/futurium/landscape_subset/ --output_dir ../datasets/futurium/test3 --augment --num_augment 1 --action simplify --frac 0.975 --frac_vary 0.025 --max_ang 0 --w 1024 --h 512 --pct_train 1.0 --num_images 5 --include_orig --split




### Dutils

#    input_src = 'trump_sub_short.mp4'
    #input_src = '../data/101_ObjectCategories/camera'

target_face_image trump2.png
save_ext = 'png'
save_mode = 'split' # 'combined', 'split', 'output_only'
    pct_test = 0.2



python3 dataset_utils.py --input_src trump_sub_short.mp4 --output_dir test66 \
    --w 1024 --h 512 --num_per 2 --frac 0.95 --frac_vary 0.05 \
    --max_ang_rot 0 --max_stretch 0 --centered \
    --action simplify --save_mode combined