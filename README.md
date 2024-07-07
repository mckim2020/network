# network
Three dimensional network analysis using stacked two dimensional images.

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