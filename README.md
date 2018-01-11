# Learning-Multi-grained-Aspect-Target-Sequence-for-Chinese-Sentiment-Analysis

## Requirements:

1. Python 2.7.12
2. Numpy 1.12.0
3. Tensorflow 0.12.1
4. Keras 1.2.1

### There are three sets of experiments in this catalogue: 1) Chinese word level experiments; 2) English experiments; and 3) Fusion experiments. 
Detailed instructions of how to run the codes are listed below.


## Chinese word level experiments:


### 1. For files of 'ATSM-S-word_4chn.py', 'v1.py', 'v2.py', 'v3.py', 'v4.py' and 'v5.py':

	  Need to change the three dataset input files for each dataset. The three files are: dataset embeddings, dataset labels and dataset aspect information.
	  Specifically, they are:

	  Notebook: 623embedding.py, 623label.txt, 623id.py
	  
	  Car:      1172embedding.py, 1172label.txt, 1172id.py
	  
	  Camera:   2231embedding.py, 2231label.txt, 2231id.py
	  
	  Phone:    2556embedding.py, 2556label.txt, 2556id.py
	  
	  All:      allembedding.py, alllabel.txt, allid.py
	  

### 2. For 'bilstm\lstm_4chn.py':

	Only need to change two dataset input files for each dataset, namely dataset embeddings and dataset labels.

	  Notebook: 623embedding.py, 623label.txt
	  
	  Car: 	    1172embedding.py, 1172label.txt
	  
	  Camera:   2231embedding.py, 2231label.txt
	  
	  Phone:    2556embedding.py, 2556label.txt
	  
	  All:      allembedding.py, alllabel.txt	

### 3. For 'memnet_4chn.py':

	  Need to change four dataset input files for each dataset, as below:

	  Notebook: 623embedding.py, 623label.txt, cont623embedding.py,623aspectid.py
	  
	  Car: 	    1172embedding.py, 1172label.txt, cont1172embedding.py, 1172aspectid.py
	  
	  Camera:   2231embedding.py, 2231label.txt, cont2231embedding.py, 2231aspectid.py
	  
	  Phone:    2556embedding.py, 2556label.txt, cont2556embedding.py, 2556aspectid.py
	  
	  All:      allembedding.py, alllabel.txt, contallembedding.py, allaspectid.py


### 4. For 'tdlstm_4chn.py':

	  Need to update four dataset input files for each dataset, as below:

	  Notebook: 623embedding.py, 623label.txt, 623ind_forward.py, 623ind_backward.py
	  
	  Car: 	    1172embedding.py, 1172label.txt, 1172ind_forward.py, 1172ind_backward.py
	  
	  Camera:   2231embedding.py, 2231label.txt, 2231ind_forward.py, 2231ind_backward.py
	  
	  Phone:    2556embedding.py, 2556label.txt, 2556ind_forward.py, 2556ind_backward.py
	  
  	  All:      allembedding.py, alllabel.txt, allind_forward.py, allind_backward.py


### 5. For 'tclstm_4chn.py':

	  Need to update five dataset input files for each dataset, as below:

	  Notebook: 623embedding.py, 623label.txt,623aspectid.py, 623ind_forward.py, 623ind_backward.py
	  
	  Car: 	1172embedding.py, 1172label.txt, 1172aspectid.py, 1172ind_forward.py, 1172ind_backward.py
	  
	  Camera: 	2231embedding.py, 2231label.txt, 2231aspectid.py, 2231ind_forward.py, 2231ind_backward.py
	  
	  Phone: 2556embedding.py, 2556label.txt, 2556aspectid.py, 2556ind_forward.py, 2556ind_backward.py
	  
	  All:  allembedding.py, alllabel.txt, allaspectid.py, allind_forward.py, allind_backward.py




## English experiments:


### 1. ATSM-S-eng.py: 

			Run on laptop domain:

				1. Uncomment line 103- 149 in order to load laptop domain data, comment line 152- 202 to block restaurant domain data.
				2. Pick up the first element (namely, [0]) on line 110, 115, 207 and 208 to run on the single word aspect subset.
				3. Pick up the second element (namely, [1]) on line 110, 115, 207 and 208 to run on the multi-word aspect subset.
				4. Record the 'Averaged testing accuracy', 'Macro F1 score' and 'Testing length' for later calculation.

			Run on restaurant domain:

				1. Uncomment line 152- 202 in order to load restaurant domain data, comment line 103- 149 to block laptop domain data.
				2. Pick up the first element (namely, [0]) on line 110, 115, 207 and 208 to run on the single word aspect subset.
				3. Pick up the second element (namely, [1]) on line 110, 115, 207 and 208 to run on the multi-word aspect subset.
				4. Record the 'Averaged testing accuracy', 'Macro F1 score' and 'Testing length' for later calculation.

			Calculation:

				Compute the weighted accuracy based on 'Averaged testing accuracy' and 'Testing length' from two domains.
				Compute the macro f1 score based on 'Macro F1 score' from two domains.


### 2. bilstm\lstm_4eng.py: 

			Run on laptop domain:

				1. Uncomment line 70- 105 in order to load laptop domain data, comment line 109- 147 to block restaurant domain data.
				2. Pick up the first element (namely, [0]) on line 77, 82, 154 and 155 to run on the single word aspect subset.
				3. Pick up the second element (namely, [1]) on line 77, 82, 154 and 155 to run on the multi-word aspect subset.
				4. Record the 'Averaged testing accuracy', 'Macro F1 score' and 'Testing length' for later calculation.

			Run on restaurant domain:

				1. Uncomment line 109- 147 in order to load restaurant domain data, comment line 70- 105 to block laptop domain data.
				2. Pick up the first element (namely, [0]) on line 77, 82, 154 and 155 to run on the single word aspect subset.
				3. Pick up the second element (namely, [1]) on line 77, 82, 154 and 155 to run on the multi-word aspect subset.
				4. Record the 'Averaged testing accuracy', 'Macro F1 score' and 'Testing length' for later calculation.

			Calculation:

				Compute the weighted accuracy based on 'Averaged testing accuracy' and 'Testing length' from two domains.
				Compute the macro f1 score based on 'Macro F1 score' from two domains.


### 3. memnet_4eng.py:

			Run on single word aspect subset:

				1.Run on restaurant domain: Uncomment line 136-140. Comment line 141-156.
				2.Run on laptop domain: Uncommnet line 136, 137, 143 and 144. Comment line 139, 140 and 149-156.

			Run on multi-word aspect subset:

				1.Run on restaurant domain: Uncomment line 149-155. Comment line 136-146, 157 and 158.
				2.Run on laptop domain: Uncomment line 151, 152, 157 and 158. Commnet line 136-146, 154 and 155.

			Calculation:

				For each of the four cases above, record 'Averaged testing accuracy', 'Macro F1 score' and 'Testing length'. Then compute    weighted accuracy and macro F1.


### 4. tc_4eng.py:

			Data loading area: line 125-150.

			Run on laptop domain:

				1. Load data files of '2951l_aspectid.py', '2951l_emb.py', 'l_label.txt', 'lind_forward.py' and 'lind_backward.py' on line 120, 157, 162, 172 and 175 respectively.

				2. Run on single word subset: Uncommnet line 125-128, 134 and 135. Comment rest of the 'Data loading area'.

				3. Run on multi-word aspect subset: Uncomment line 140-143, 149 and 150. Comment rest of the 'Data loading area'.

			Run on restaurant domain:

				1. Load data files of '4722r_aspectid.py', '4722r_emb.py', 'r_label.txt', 'rind_forward.py' and 'rind_backward.py' on line 120, 157, 162, 172 and 175 respectively.

				2. Run on single word subset: Uncomment line 125-128, 131 and 132. Comment rest of the 'Data loading area'.

				3. Run on multi-word aspect subset: Uncomment line 140-143, 146 and 147. Comment rest of the 'Data loading area'.

			Calculation:

				For each of the four cases above, record 'Averaged testing accuracy', 'Macro F1 score' and 'Testing length'. Then compute weighted accuracy and macro F1.	
				


### 5. td_4eng.py:

			Data loading area: line 83-108.

			Run on laptop domain:

				1. Load data files of '2951l_aspectid.py', '2951l_emb.py', 'l_label.txt', 'lind_forward.py' and 'lind_backward.py' on line 78, 114, 119, 129 and 132 respectively.

				2. Run on single word subset: Uncommnet line 83-86, 92 and 93. Comment rest of the 'Data loading area'.

				3. Run on multi-word aspect subset: Uncomment line 98-101, 107 and 108. Comment rest of the 'Data loading area'.

			Run on restaurant domain:

				1. Load data files of '4722r_aspectid.py', '4722r_emb.py', 'r_label.txt', 'rind_forward.py' and 'rind_backward.py' on line 78, 114, 119, 129 and 132 respectively.

				2. Run on single word subset: Uncomment line 83-86, 89 and 90. Comment rest of the 'Data loading area'.

				3. Run on multi-word aspect subset: Uncomment line 98-101, 104 and 105. Comment rest of the 'Data loading area'.

			Calculation:

				For each of the four cases above, record 'Averaged testing accuracy', 'Macro F1 score' and 'Testing length'. Then compute weighted accuracy and macro F1.			



## Fusion experiments:


For every of the five datasets, update three dataset input files for each required granularities, based on combinations. For example:


	Notebook:

		Word: ../word/623embedding.py, ../word/623label.txt, ../word/623id.py
		Character: ../character/623embedding.py, ../character/623label.txt, ../character/623id.py
		Radical: ../radical/623embedding.py, ../radical/623label.txt, ../radical/623id.py


	Car:

		Word: ../word/1172embedding.py, ../word/1172label.txt, ../word/1172id.py
		Character: ../character/1172embedding.py, ../character/1172label.txt, ../character/1172id.py
		Radical: ../radical/1172embedding.py, ../radical/1172label.txt, ../radical/1172id.py

	Camera:

		Word: ../word/2231embedding.py, ../word/2231label.txt, ../word/2231id.py
		Character: ../character/2231embedding.py, ../character/2231label.txt, ../character/2231id.py
		Radical: ../radical/2231embedding.py, ../radical/2231label.txt, ../radical/2231id.py

	Phone:

		Word: ../word/2556embedding.py, ../word/2556label.txt, ../word/2556id.py
		Character: ../character/2556embedding.py, ../character/2556label.txt, ../character/2556id.py
		Radical: ../radical/2556embedding.py, ../radical/2556label.txt, ../radical/2556id.py

	All:

		Word: ../word/allembedding.py, ../word/alllabel.txt, ../word/allid.py
		Character: ../character/allembedding.py, ../character/alllabel.txt, ../character/allid.py
		Radical: ../radical/allembedding.py, ../radical/alllabel.txt, ../radical/allid.py


