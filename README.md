# Network Analysis
Three dimensional network analysis using stacked two dimensional images.

Original work was done by [Qiber3D](https://github.com/theia-dev/Qiber3D).

## Workflow
1. Go to test directory and create a separate folder to test with.
```
cd test
mkdir mytest
cd mytest
```

2. Log into a cpu node (optional)
```
srun -p cpu -n 1 --pty bash
```

3. Run Python script
```
python ../../python/srun.py --plot True
```
