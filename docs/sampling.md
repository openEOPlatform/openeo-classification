# Sampling 

## Creodias runs

First run:
- indexreduction 7
- tilesize 64
- temporal by day
- no alignment between Sentinel 1 and 2 data cubes

These settings resulted in: 
- huge numbers of partitions with respect to actual dataset.
- Large partitions combined with low memory resulted in a lot of 'index' shuffle size, with a size of 1.1MB each
- Executors went out of disk space, using up to 90GB for shuffle storage
- Long run times (4-5 hours) for relatively few samples


### Run 26/03
With Goofys
6 hours, then failure on IO errors

Maybe too high parallellism

Still ~32k partitions in Sentinel-2 preprocessing 

### Run 29/03
Partitioning by spatial key resulted in hanging jobs, deadlock like, but in the native gdal code.
Partitioning again into smaller partitions fixed this, ran at 9 seconds per chunk, 1.8 hours total.