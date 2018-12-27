
# for most image processing notebooks, get 101_Object_Categories
echo "Downloading 101_Object_Categories for image notebooks"
curl -L -o 101_ObjectCategories.tar.gz --progress-bar http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
tar -xzf 101_ObjectCategories.tar.gz
rm 101_ObjectCategories.tar.gz

# for eigenfaces, get labeled faces in the wild
echo "Downloading LFW dataset for faces in the wild"
curl -L -o lfw-funneled.tgz --progress-bar http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
tar -xzf lfw-funneled.tgz 
rm lfw-funneled.tgz 

# drum samples
echo "Downloading drum samples for audio t-SNE notebook"
curl -L -o drums.rar --progress-bar http://ivcloud.de/index.php/s/QyDXk1EDYDTVYkF/download
unrar x -o+ drums.rar
rm drums.rar

# drum samples
echo "Downloading Reuters dataset for text retrieval notebook"
curl -L -o Reuters21578-Apte-90Cat.tar.gz --progress-bar wget http://disi.unitn.it/moschitti/corpora/Reuters21578-Apte-90Cat.tar.gz
tar -xzf Reuters21578-Apte-90Cat.tar.gz
rm Reuters21578-Apte-90Cat.tar.gz
