all: build

build:
	jupyter-book build deeprl

export:
	git add deeprl/_build/html
	git commit -a -m "Rebuilding site `date`"
	git push origin master
	ghp-import -n -p -f deeprl/_build/html
	scp .htaccess deeprl/_build/html/
	rsync -avze ssh --progress --delete ./deeprl/_build/html/ vitay@login.tu-chemnitz.de:/afs/tu-chemnitz.de/www/root/informatik/KI/edu/deeprl/notes/

clean:
	rm -rf ./deeprl/_build
