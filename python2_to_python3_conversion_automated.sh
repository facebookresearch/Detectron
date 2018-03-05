# https://stackoverflow.com/questions/1585170/how-to-find-and-replace-all-occurrences-of-a-string-recursively-in-a-directory-t
find . -type f | xargs sed -i 's/#!\/usr\/bin\/env python2/#!\/usr\/bin\/env python/g'
find . -type f | xargs sed -i 's/python2 /python /g'

# use 2to3 to do the rest
2to3 -wn .
