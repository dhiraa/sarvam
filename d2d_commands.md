```commandline
git add .
git commit -m "update"
git push origin HEAD
```



-------------------------------------------------------------------------------------------------------
**Sub modules notes:**

```commandline
#add submodule and define the master branch as the one you want to track  
git submodule add -b master [URL to Git repo]     
git submodule init

#update your submodule --remote fetches new commits in the submodules 
# and updates the working tree to the commit described by the branch  
# pull all changes for the submodules
git submodule update --remote
 ---or---
# pull all changes in the repo including changes in the submodules
git pull --recurse-submodules


# update submodule in the master branch
# skip this if you use --recurse-submodules
# and have the master branch checked out
cd [submodule directory]
git checkout master
git pull

# commit the change in main repo
# to use the latest commit in master of the submodule
cd ..
git add [submodule directory]
git commit -m "move submodule to latest commit in master"

# share your changes
git push
``` 

```
ipython notebook --no-browser --port=8889
ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host
```

