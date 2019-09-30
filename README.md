CUDA Number Algorithms
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

Davis Polito 
*  [https://github.com/davispolito/Project3-CUDA-Path-Tracer/blob/master]()
* Tested on: Windows 10, i7-8750H @ 2.20GHz 16GB, GTX 1060       
#Stream Compaction

![Graph Showing effect of stream compaction](/img/stream.png)
This graph shows that as we remove the number of walls, making the scene more open, stream compaction is able to increase the speed of our project by drastically reducing the number of paths we must check intersection for. 

#Material Sorting
Without sorting
![Without Sorting](/img/nosort.png)
with sorting
![With Sorting](/img/sort.PNG)
As you can see the obvious differences between the issuing efficiency of with sorting verses without sorting are in memory dependency. We've reduced the latency significantly since the gpu can do sequential memory accesses. 

#Object Loading

![Cube from object](/img/objcube.png)
This is a reflective cube generated using object files. 
![Wahoo from Object](/img/wahoo.png)
![Wahoo from Object](/img/bighoo.png)

#Blooper reel
These two bloopers occured due to stream compacting before shading and computing intersections
![Stream Compaction too Early](/img/bloop2.png)

![Stream Compaction too Early](/img/bloop3.png)
This blooper occured due to poor seeding for the random number generation
![Random Number seeding in Error](/img/bloop1.png)
