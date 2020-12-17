### Hardware requirements
# OS : Ubuntu 18.04
# GPU memory : >= 8G
# CPU :  >= 4-cores (>8 better)
# RAM : >= 16G (>32 better)

sudo apt-get install p7zip-full



conda create --name research python=3.6.10
conda activate research
pip install -r requirements-research.txt 


conda create --name tf2 python=3.7.7
conda activate tf2
pip install -r requirements-tf2.txt 



# download dataset video
cd datasets

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1EUzlw3-b5H_otRLudNj0SrscQsXt_wle' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EUzlw3-b5H_otRLudNj0SrscQsXt_wle" -O tmp.zip && rm -rf /tmp/cookies.txt
7z x tmp.zip
cd ../

# download CycleGAN model
cd models

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YrAB4juUXzrUg4uY5PxH8cvLR9lX7mwN -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YrAB4juUXzrUg4uY5PxH8cvLR9lX7mwN" -O tmp.zip && rm -rf /tmp/cookies.txt
7z x tmp.zip
cd ../
