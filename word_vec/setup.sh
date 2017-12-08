mkdir tmp
cd  tmp
mkdir text
#wget https://dumps.wikimedia.org/simplewiki/20171120/simplewiki-20171120-pages-articles.xml.bz2 
#sudo apt install dtrx
#dtrx simplewiki-20171120-pages-articles.xml.bz2
#get Wiki Extractor
git clone https://github.com/attardi/wikiextractor
cd wikiextractor
python WikiExtractor.py ../simplewiki-20171120-pages-articles.xml -o ../text/
	

