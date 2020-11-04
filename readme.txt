

Dependencies include keras, numpy, scikit-learn, pandas, biopython, and tensorflow

The work here is based off of DeeplyEssential as published by UCR and modifications were made to get the program functioning as well as optimizing

One runs this network by typing ./main.py '' '' orthoMCL.txt dataset.txt -c results

Where the first two empty/null fields is the path to the DEG dataset, the second last field specifies the use of both gram positive and gram negative bacterial species and the last argument is used to specify the name of the output files.

The dataset used for this can be obtained from http://www.essentialgene.org/ and is the essential and non-essential bacterial species dataset.
