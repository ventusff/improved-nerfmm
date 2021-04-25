# modified from https://github.com/bmild/nerf/blob/master/download_example_data.sh
mkdir -p data
cd data
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip nerf_example_data.zip
cd ..