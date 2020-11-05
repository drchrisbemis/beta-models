# beta-models
Linear modeling examples

Proof of concept of building out some linear modeling framework.  By no means complete, but indicative.

Perhaps uncommon packages which are required:
yfinance (installed using pip)
requests (installed using conda)
quandl (installed using conda)

All code may be seen in action via 
  main_script.py
 which may be simply run.

The project begins with pulling a cross-section of tickers from Quandl.  I have shared my api key.  Please don't use it outside this example.

After a cross-section has been established -- here, about 3400 names -- market cap information is pulled, again from Quandl. This is low-frequency (as in filing-frequency), but provides some indication of market cap for this project.  A fuller treatment is obvious.

The project asked to slice across different market cap segments.  Market cap segmentation is performed on the cross-section of market caps based on prescribed percentiles.  With labels in hand, some sample is taken across Large, Mid, and Small cap.

Daily pricing information is next pulled from Yahoo for the sample of names.  A start date of 1/1/2018 is arbitrarily chosen. 

A market reference of SPY is fixed as well, and pricing is pulled from the same source across the same dates.

Return cacluations are performed assuming no date gaps. This would be a necessary refinement, but I did not see it as accretive to the current project's outline.

Some manipulation of data is necessary (or expedient, either way), and to show performance in-line, a even smaller sample is developed.  This is only done to show example performance in-line; the code is agnostic to the size of cross-sections or number of dates needed for a backfill (within reason).  This qualifier on size could be mitigated or removed if the work incorporated a database instead of extensive use of dataframes.

Two examples of linear modeling are given in the next section of the code.  First, a barbones least squares implementation is provided using just np.linalg.lstsq.  The next uses the same least squares engine, but also includes more diagnostics as well as an inidicative outlier screening based on (internal) studentized residuals.  This is not an indication of this being a preferred outlier detection methodology; it is not.  It seemed appropriate for this project, however.

In the fuller implementation, several statistical quantities are calculated from model residuals without the use of additional packages, save a necessary call to scipy.stats to obtain student t critical values.

Both modeling calculation functions are wrapped in single day and backfill scripts in separate examples in the main script.  Some attempt to show an iterative and flexible approach was made as it seemed to be indicated by the original spec.  Too, there was some question of improving speed over the baseline np.linalg.lstsq.  I could not.  I understand that solving the normal equations will be slower.  I also tried using the closed form solution of linear regression with intercept and single predictor.  This, too, was slower than the LU implementation I believe np.linalg.lstsq uses.

A full backfill of 300 names across almost 600 dates takes about 860 s on my machine.  I did not parallelize any of the loops, but this could be done.  I have not done this in the past and wanted to focus elsewhere.  The time requirements here are well within scope for the time scales I have considered previously; I understand this is not the case there.

While it is not necessary to run a full backfill, it could be done if desired.  I saved a .pkl of the results as full_beta_backfill.pkl which is loaded within the main script to a dataframe.

Next, examples of visualizing the data are given, including time series plots of linear estimates and their confidence intervals through time for a given name, as well as R2, realized volatility, and idiosyncratic volatility based on the model being considered.  The literature is often interested in the cross-section of idiosyncratic volatility.  As well, there is a perverse relationship with the cross-section of betas and ex-post performance.

Some simple diagnostics are also shown for a single day's fit, with outliers indicated in a scatter plot as well as expectation lines with mean prediction confidence intervals.  Studentized residuals are also shown, sharing the same axes as a soft indication of potential unexplained features.

Finally, an example using an alternative regression methodology is shown.  I chose to keep this relatively barebones, omitting checks shwon in some of the work above as I simply wanted to share a machine learning pipeline example using an interesting regressor; here, the Huber Regressor.  Put simply, the loss function considers L2 losses up to a band, and then L1 losses outside.  The benefit being that it is robust to outliers.  Too, the problem may be formulated as a QP problem, so it is performant.  

The question of 'fitting the band' suggested just above is handled via k-fold cross-validation and a parameter grid outlined in the function I wrote.

A plot of the output of the model is given, and the band chosen is indicated vis a vis the outliers as determined by the model and highlighted in the shared figure.
