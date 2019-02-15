# CONSUMER-SEGMENTATION-USING-APRIORI-ALGORITHM
PROBLEM STATEMENT: - 

Segmenting potential customers to target them in order to offer them relevant products. 



PROBLEM DESCRIPTION: - 

Customers often get confused seeing the variety of products they are surrounded by. 
Even the marketers waste a lot of time and money on convincing and approaching customers who are less interested or are not in need of that product. 
So in order to solve both the issues, we will try to come up with a solution using data mining and hence will get a better idea about interests and wants of individual customers. 
Having a better understanding of customers will help the marketers to communicate with the targeted people more effectively, giving them the best insights about their favorable products.

DATASET DESCRIPTION:

Abstract: This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.	
Data Set Characteristics:  	Multivariate, Sequential, Time-Series	
Number of Instances:	541909	
Area:	Business
Attribute Characteristics:	Integer, Real	
Number of Attributes:	8	
Date Donated	2015-11-06
Associated Tasks:	Classification, Clustering	
Missing Values?	N/A	
Number of Web Hits:	253549


Source:
Dr Daqing Chen, Director: Public Analytics group. chend'@' lsbu.ac.uk, School of Engineering, London South Bank University, London SE1 0AA, UK. 
Attribute Information:
InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation. 
StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product. 
Description: Product (item) name. Nominal. 
Quantity: The quantities of each product (item) per transaction. Numeric. 
InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated. 
UnitPrice: Unit price. Numeric, Product price per unit in sterling. 
CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer. 
Country: Country name. Nominal, the name of the country where each customer resides. 

ALGORITHM:
For Customer Profiling and Segmentation, we will be using Association Data Mining rule(Apriori) and also K-means Clustering

Apriori Algorithm:

Apriori algorithm is used for finding frequent itemsets in a dataset for Boolean association rule. Name of algorithm is Apriori is because it uses prior knowledge of frequent item set properties. We apply a iterative approach or level-wise search where k-frequent itemsets are used to find k+1 itemsets.
To improve the efficiency of level-wise generation of frequent itemsets an important property is used called Apriori property which helps by reducing the search space.
Apriori Property –
All nonempty subset of frequent itemset must be frequent. The key concept of Apriori algorithm is its anti-monotonicity of support measure. Apriori assumes that
All subsets of a frequent itemset must be frequent(Aprioripropertry).
If a itemset is infrequent all its supersets will be infrequent. 
