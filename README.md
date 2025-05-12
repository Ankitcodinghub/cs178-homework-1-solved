# cs178-homework-1-solved
**TO GET THIS SOLUTION VISIT:** [CS178 Homework 1 Solved](https://www.ankitcodinghub.com/product/cs178-homework-1-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;90975&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS178 Homework 1 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
<div class="page" title="Page 1">
<table>
<tbody>
<tr>
<td></td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
This homework (and many subsequent ones) involves data analysis, and discussion of methods and results, using Python. You must submit a single PDF file that contains all answers, including any text needed to describe your results, the complete code snippets used to answer each problem, any figures that were generated, and scans of any (clearly readable) work on paper that you want the graders to consider. It is important that you include enough detail that we know how you solved each problem, since otherwise we will be unable to grade it.

We recommend that you use Jupyter/iPython notebooks to write your report. It will help you not only ensure all of the code for the solutions is included, but also provide an easy way to export your results to a PDF file.1 We recommend liberal use of Markdown cells to create headers for each problem and sub-problem, explaining your implementation/answers, and including any mathematical equations. For parts of the homework you do on paper, scan it in such that it is legible (there are a number of free Android/iOS scanning apps, if you do not have access to a scanner), and include it as an image in the iPython notebook. If you have any questions about using iPython, ask us on Piazza. If you decide not to use iPython notebooks, and instead create your PDF file with Word or LaTeX, make sure all of the answers can be generated from the code snippets included in the document.

TL;DR: (1) submit a single, standalone PDF report, with all code; (2) I recommed iPython notebooks. Problem 0: Get Connected (0 points, but it will make the course easier!)

Please visit our class forum on Piazza: piazza.com/uci/fall2020/cs178. Piazza will be the place to post your questions and discussions, rather than by email to the instructor or TA. Often, other students have the same or similar questions, and will be helped by seeing the online discussion.

Problem 1: Python &amp; Data Exploration

In this problem, we will compute some basic statistics and create visualizations of an example data set. First, download the zip file for Homework 1, which contains some course code (the mltools directory) and a dataset of New York area real estate sales, “nyc_housing”. Load the data into Python:

1 2 3 4 5 6

</div>
</div>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
import numpy as np

import matplotlib.pyplot as plt

nych = np.genfromtxt(“data/nyc_housing.txt”,delimiter=None) # load the text file

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Y = nych[:,-1] # target value (NYC borough) is the last column X = nych[:,0:-1] # features are the other columns

</div>
</div>
</td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
These data are from the “NYC Open Data” initiative, and consist of three real-valued features and a class value Y representing in which of three boroughs the house or apartment was located (Manhattan, the Bronx, or Staten Island).

1. Use X.shape to get the number of features and the number of data points. Report both numbers, mentioning which number is which. (5 points)

<ol start="2">
<li>For each feature, plot a histogram ( plt.hist ) of the data values. (5 points)</li>
<li>Compute the mean &amp; standard deviation of the data points for each feature ( np.mean , np.std ). (5 points)</li>
<li>For each pair of features (1,2), (1,3), and (2,3), plot a scatterplot (see plt.plot or plt.scatter ) of the feature values, colored according to their target value (class). (For example, plot all data points with y = 0 as blue, y = 1 as green, and y = 2 as red.) (5 points)</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
1For example, by doing a Print Preview in Chrome and printing it to a PDF. Please also remember to check the resulting PDF before submitting.

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Homework 1 UC Irvine 1/ 4

</div>
</div>
</td>
</tr>
</tbody>
</table>
</div>
<div class="page" title="Page 2">
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
CS 178: Machine Learning &amp; Data Mining Fall 2020

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Problem 2: k-nearest-neighbor predictions (25 points)

In this problem, you will continue to use the NYC Housing data and create a k-nearest-neighbor (kNN) classifier using the provided knnClassify python class. While completing this problem, please explore the implementation to become familiar with how it works.

First, we will shuffle and split the data into training and validation subsets:

</div>
</div>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
nych = np.genfromtxt(“data/nyc_housing.txt”,delimiter=None) # load the data Y = nych[:,-1]

X = nych[:,0:-1]

# Note: indexing with “:” indicates all values (in this case, all rows);

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
# indexing with a value (“0”, “1”, “-1”, etc.) extracts only that value (here, columns); # indexing rows/columns with a range (“1:-1”) extracts any row/column in that range.

import mltools as ml

# We’ll use some data manipulation routines in the provided class code

# Make sure the “mltools” directory is in a directory on your Python path, e.g.,

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
# export PYTHONPATH=$\$${PYTHONPATH}:/path/to/parent/dir # or add it to your path inside Python:

<ul>
<li># &nbsp;import sys</li>
<li># &nbsp;sys.path.append(‘/path/to/parent/dir/’);</li>
</ul>
</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
np.random.seed(0) # set the random number seed

X,Y = ml.shuffleData(X,Y); # shuffle data randomly

# (This is a good idea in case your data are ordered in some systematic way.)

Xtr,Xva,Ytr,Yva = ml.splitData(X,Y, 0.75); # split data into 75/25 train/validation

</div>
</div>
</td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
1 2 3 4 5 6 7 8 9

<pre>10
11
12
13
14
15
16
17
18
19
20
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
Make sure to set the random number seed to 0 before calling shuffleData as in the example above (and in general, for every assignment). This ensures consistent behavior each time the code is run.

Learner Objects Our learners (the parameterized functions that do the prediction) will be defined as python objects, derived from either an abstract classifier or abstract regressor class. The abstract base classes have a few useful functions, such as computing error rates or other measures of quality. More importantly, the learners will all follow a generic behavioral pattern, allowing us to train the function on one data set (i.e., set the parameters of the model to perform well on those data), and then make predictions on another data set.

</div>
</div>
<div class="layoutArea">
<div class="column">
You can now build and train a kNN classifier on Xtr,Ytr and make predictions on some data Xva with it:

</div>
</div>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
knn = ml.knn.knnClassify() # create the object and train it

knn.train(Xtr, Ytr, K) # where K is an integer, e.g. 1 for nearest neighbor prediction

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
YvaHat = knn.predict(Xva) # get estimates of y for each data point in Xva

# Alternatively, the constructor provides a shortcut to “train”:

<pre>knn = ml.knn.knnClassify( Xtr, Ytr, K );
YvaHat = predict( knn, Xva );
</pre>
</div>
</div>
</td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
1 2 3 4 5 6 7

1

</div>
<div class="column">
If your data are 2D, you can visualize the data set and a classifier’s decision regions using the function

ml.plotClassify2D( knn, Xtr, Ytr ); # make 2D classification plot with data (Xtr,Ytr)

This function plots the training data and colored points as per their labels, then calls knn ’s predict function on a densely spaced grid of points in the 2D space, and uses this to produce the background color. Calling the function with knn=None will plot only the data.

1. Modify the code listed above to use only the first two features of X (e.g., let X be only the first two columns of nych , instead of the first three), and visualize (plot) the classification boundary for varying values of K = [1, 5, 10, 50] using plotClassify2D . (10 points)

2. Again using only the first two features, compute the error rate (number of misclassifications) on both the training and validation data as a function of K = [1, 2, 5, 10, 50, 100, 200]. You can do this most easily with a for-loop:

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Homework 1 UC Irvine 2/ 4

</div>
</div>
</td>
</tr>
</tbody>
</table>
</div>
<div class="page" title="Page 3">
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
CS 178: Machine Learning &amp; Data Mining Fall 2020

</div>
</div>
</td>
</tr>
<tr>
<td>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
K=[1,2,5,10,50,100,200];

errTrain = [None]*len(K) # (preallocate storage for training error) for i,k in enumerate(K):

learner = ml.knn.knnClassify(… # TODO: complete code to train model

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Yhat = learner.predict(… # TODO: predict results on training data

errTrain[i] = … # TODO: count what fraction of predictions are wrong #TODO: repeat prediction / error evaluation for validation data

plt.semilogx(… #TODO: average and plot results on semi-log scale

</div>
</div>
</td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
1 2 3 4 5 6 7 8 9

3. Create the same error rate plots as the previous part, but with all the features in the dataset. Are the plots very different? Is your recommendation for the best K different? (5 points)

Problem 3: Naïve Bayes Classifiers (35 points)

In order to reduce my email load, I decide to implement a machine learning algorithm to decide whether or not I should read an email, or simply file it away instead. To train my model, I obtain the following data set of binary-valued features about each email, including whether I know the author or not, whether the email is long or short, and whether it has any of several key words, along with my final decision about whether to read it ( y = +1 for “read”, y = −1 for “discard”).

x1 x2 x3 x4 x5 y know author? is long? has ‘research’ has ‘grade’ has ‘lottery’ ⇒ read?

0 0 1 1 0 -1 1 1 0 1 0 -1 0 1 1 1 1 -1 1 1 1 1 0 -1 0 1 0 0 0 -1 101111 001001 100001 101101 1 1 1 1 1 -1

I decide to try a naïve Bayes classifier to make my decisions and compute my uncertainty. In the case of any ties where both classes have equal probability, we will prefer to predict class +1.

<ol>
<li>Compute all the probabilities necessary for a naïve Bayes classifier, i.e., the class probability p(y) and all the individual feature probabilities p(xi|y), for each class y and feature xi. (7 points)</li>
<li>Whichclasswouldbepredictedforx=(00000)? Whataboutforx=(11010)? (7points)</li>
<li>Compute the posterior probability that y = +1 given the observation x = (0 0 0 0 0). Also compute theposterior probability that y = +1 given the observation x = (1 1 0 1 0). (7 points)</li>
<li>Why should we probably not use a “joint” Bayes classifier (using the joint probability of the features x, asopposed to the conditional independencies assumed by naïve Bayes) for these data? (7 points)</li>
<li>Suppose that before we make our predictions, we lose access to my address book, so that we cannot tell whether the email author is known. Do we need to re-train the model to classify based solely on the other four features? If so, how? If not, what changes about how our trained parameters are used? Hint: what parameters do I need for a naïve Bayes model over only features x2, . . . , x5? Do I need to re-calculate any new parameter values in our new setting? What, if anything, changes about the parameters or the way they are used? (7 points)</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
Plot the resulting error rate functions using a semi-log plot ( semilogx ), with training error in red and validation error in green. Based on these plots, what value of K would you recommend? (10 points)

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Homework 1 UC Irvine 3/ 4

</div>
</div>
</td>
</tr>
</tbody>
</table>
</div>
<div class="page" title="Page 4">
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
CS 178: Machine Learning &amp; Data Mining Fall 2020

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Problem 4: Gaussian Bayes Classifiers (15 points)

Now, using the NYC Housing data, we will explore a classifier based on Bayes rule. Again, we’ll use only the first two features of NYC Housing, shuffled and split in to training and validation sets as before.

<ol>
<li>Splitting your training data by class, compute the empirical mean vector and covariance matrix of the data in each class. (You can use mean and cov for this.) (5 points)</li>
<li>Plot a scatterplot of the data, coloring each data point by its class, and use plotGauss2D to plot contours on your scatterplot for each class, i.e., plot a Gaussian contour for each class using its empirical parameters, in the same color you used for those data points. 5 points</li>
<li>Visualize the classifier and its boundaries that result from applying Bayes rule, using</li>
</ol>
1 2

Problem 5: Statement of Collaboration (5 points)

It is mandatory to include a Statement of Collaboration in each submission, that follows the guidelines below. Include the names of everyone involved in the discussions (especially in-person ones), and what was discussed.

All students are required to follow the academic honesty guidelines posted on the course website. For programming assignments in particular, I encourage students to organize (perhaps using Piazza) to discuss the task descriptions, requirements, possible bugs in the support code, and the relevant technical content before they start working on it. However, you should not discuss the specific solutions, and as a guiding principle, you are not allowed to take anything written or drawn away from these discussions (no photographs of the blackboard, written notes, referring to Piazza, etc.). Especially after you have started working on the assignment, try to restrict the discussion to Piazza as much as possible, so that there is no doubt as to the extent of your collaboration.

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>bc = ml.bayes.gaussClassify( Xtr, Ytr );
ml.plotClassify2D(bc, Xtr, Ytr);
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
Also compute the empirical error rate (number of misclassified points) on the training and validation data.

(5 points)

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Homework 1 UC Irvine 4/ 4

</div>
</div>
</td>
</tr>
</tbody>
</table>
</div>
