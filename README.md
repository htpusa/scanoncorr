# Sparse canonical correlation analysis

`scanoncorr` performs sparse canonical correlation analysis in MATLAB. The 
algorithm is based on the alternating projected gradient approach presented
in [1]. Sparsity is induced using L1-norm constraints on the canonical 
coefficient vectors.

## Quick start

"Installing" a MATLAB package is easy: just clone this repository and add
it to your MATLAB path:

```
addpath('path_to/scanoncorr') 
```

Set up some data
```
load carbig;
data = [Displacement Horsepower Weight Acceleration MPG];
nans = sum(isnan(data),2) > 0;
X = data(~nans,1:3); Y = data(~nans,4:5);
```

Choose sparsity parameters and calculate canonical coefficients
```
cx = 0.1;
cy = 0.1;
[A B r U V] = scanoncorr(X,Y,cx,cy);
```

Visualise the results
```
figure
    subplot(2,2,1:2)
    gscatter(U,V,Cylinders(~nans))
    subplot(2,2,3)
    bar(A)
    xticklabels(["Displacement","Horsepower","Weight"])
    subplot(2,2,4)
    bar(B)
    xticklabels(["Acceleration","MPG"])
````

## More detailed instructions

Load a more illustrative data set
```
load scanoncorr_example
X = data.X; Y = data.Y;
```

### Multiple canonical vectors

You can find multiple canonical vectors using the option 'D'
```
[A B] = scanoncorr(X,Y,cx,cy,'D',2);
```

### Choosing the initialisation method

A and B have to be seeded at the start of the algorithm. By default this is done using singular vectors of the cross-covariance matrix. Another option
is to try several random starts and pick the result that achieves the best value for the objective
```
[A B] = scanoncorr(X,Y,cx,cy,'init','random');
```
The two options can also be combined. Here we try first the singular vectors and then 10 random starts
```
[A B] = scanoncorr(X,Y,cx,cy,'rStarts',10);
```
Note that the default 'svd' option using singular vectors usually performs the best.

### Optimising the hyperparameters
The function `optimiseScanoncorrParameters` can be used to find the best values for the regularisation parameters cx and cy, and to pick the initialisation method. It performs cross-validation over a grid of cx and cy values and picks the combination of parameters that performs the best on average.
```
[optInit,optCx,optCy,results] = optimiseScanoncorrParameters(X,Y)
```
This function produces two figures, one for each initialisation method, displaying the average correlation in the test set, as well as the approximate cardinality of A and B.

## References

[1] Uurtio, Viivi, Sahely Bhadra, and Juho Rousu. "Large-scale sparse 
    kernel canonical correlation analysis." International Conference on
    Machine Learning. PMLR, 2019.
