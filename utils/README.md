
# Dataset-utils
    
Input options

* `--input_src` either a video (mp4) or directory of images to use as the input source
* `--max_num_images` cap number of images to use as input (if None, no cap, default None)
* `--shuffle` shuffle the input images (true) or go in order (false)
* `--min_dim` skip any image whose width or height are less (default 0)

Output options

* `--output_dir` directory to put resulting images into
* `--pct_test` fraction of output images (0-1) to reserve for testing set (default 0)
* `--save_mode` "split": separate directories for input/output, "combined": concatenate horizontally, "output_only": save only the resulting outputs
* `--save_ext` save jpg or png

Pre-processing and augmentation options

* `--w` output width (default 256)
* `--h` output height (default 256)
* `--num_per` how many copies of each input image (default 1)
* `--frac` fraction of original image to crop (default 1, whole image)
* `--frac_vary` random deviation to fraction (default 0)
* `--max_ang_rot` rotate the image randomly up to "max_ang_rot" (default 0, no stretch)
* `--max_stretch` stretch the image randomly up to "max_stretch" (default 0, no stretch)
* `--centered` take center crop (if false, take random crop, default false)

Processing the input image

* `--action` a comma-separated list of processing actions from ('quantize', 'trace', 'hed', 'segment', 'simplify', 'face')
* `--target_face_image` if doing face extraction, use this image to specify a target face (if None, then it takes first face it can find)
* `--hed_model_path` path to model file for holistic-edge-detection (HED) processing default='../data/HED_reproduced.npz'
* `--landmarks_path` path to dlib face landmarks file (default='../data/shape_predictor_68_face_landmarks.dat')


#### Install

If you use the HED processing, you need to install tensorpack and download the model.

    pip install --upgrade git+https://github.com/tensorpack/tensorpack.git
    wget http://models.tensorpack.com/HED/HED_reproduced.npz -P ../data/.
    
If you plan to do face extraction, you need the landmarks file

    curl -L -o ../data/shape_predictor_68_face_landmarks.dat.bz2 --progress-bar https://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d ../data/shape_predictor_68_face_landmarks.dat.bz2

#### Example

    python3 dataset_utils.py --input_src myMovie.mp4 --output_dir myResults \
        --w 1024 --h 512 --num_per 2 --frac 0.95 --frac_vary 0.05 \
        --max_ang_rot 0 --max_stretch 0 --centered \
        --action simplify --save_mode combined


# Scraping WikiArt 

See https://www.wikiart.org/en/paintings-by-genre/ for list of genres. List follows:

portrait, landscape, genre-painting, abstract, religious-painting, cityscape, sketch-and-study, figurative, illustration, still-life, design, nude-painting-nu, mythological-painting, marina, animal-painting, flower-painting, self-portrait, installation, photo, allegorical-painting, history-painting, interior, literary-painting, poster, caricature, battle-painting, wildlife-painting, cloudscape, miniature, veduta, yakusha-e, calligraphy, graffiti, tessellation, capriccio, advertisement, bird-and-flower-painting, performance, bijinga, pastorale, trompe-loeil, vanitas, shan-shui, tapestry, mosaic, quadratura, panorama, architecture

See https://www.wikiart.org/en/paintings-by-styles/ for list of styles. List follows:

impressionism, realism, romanticism, expressionism, post-impressionism, surrealism, art-nouveau, baroque, symbolism, abstract-expressionism, na-ve-art-primitivism, neoclassicism, cubism, rococo, northern-renaissance, pop-art, minimalism, abstract-art, art-informel, ukiyo-e, conceptual-art, color-field-painting, high-renaissance

You can only choose genre or style at the moment, not both. 

Example:

    python scrape_wikiart.py --genre landscape --num_pages 3 --output_dir ../datasets
    python scrape_wikiart.py --style impressionism --num_pages 3 --output_dir ../datasets
