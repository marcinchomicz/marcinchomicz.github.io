<html lang="pl"><head><meta charset="UTF-8"><style type="text/css">
.newStyle1 {
	font-family: "Lucida Sans", "Lucida Sans Regular", "Lucida Grande", "Lucida Sans Unicode", Geneva, Verdana, sans-serif;
	list-style-type: lower-greek;
}
.auto-style1 {
	text-align: left;
	font-size: large;
	font-style: bold;
	border: 0px solid #00FF00;
	
}
.auto-style1A {
	text-align: left;
	font-size: small;
}	
.auto-style2 {
	border: 0px solid #00FF00;
}
.auto-style4 {
	text-align: left;
}
.auto-style5 {
	border-width: 1px;
}
.auto-style6 {
	font-size: small;
}
.auto-style7 {
	text-align: left;
	font-size: small;
}
.auto-style7A {
	text-align: left;
	font-size: small;
}	
</style>
</head>
<p>The case studies included here were prepared either to verify some ideas, compare the results of various computation variants, or test the analytical process. Most of the notebooks are based on publicly available datasets, so the results can be easily compared to the solutions from Kaggle or other publicly available data science portals, nevertheless, the goal of the case studies was not to achieve to best possible performance.</p>
<body class="newStyle1" style="background-color: #FFFFFF">

<table cellspacing="1" class="auto-style2" style="width: 1024px">
	<tr>
		<th class="auto-style1" style="width: 40%"><span lang="pl">Hardware and frameworks evaluation</span></th>
		<th class="auto-style1" style="width: 60%"><span lang="pl"></span></th>
	</tr>
	<tr><td class="auto-style6"><span lang="pl">
		<a href="https://marcinchomicz.github.io/GPU%20vs%20CPU%20benchmarks/cuml_benchmarks-MCH.html">
			Classic ML algorithms on CUDA devices</a>
		<a href="https://marcinchomicz.github.io/GPU%20vs%20CPU%20benchmarks/cuml_benchmarks-MCH.ipynb">
				<img class="auto-style5" height="10" src="logo_svg.png" width="37"></a></span>
		</td>
		<td class="auto-style7A">
			<span lang="pl"><p>Several tests of classic ML algorithms performance using in two implementations: well known Scikit-learn and Nvidia CUML framework. 
				Comparing the performance on 32 cores CPU: <a href="https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+Threadripper+3970X&id=3623">Ryzen Threadripper 3970</a>
				and CUDA GPU: <a href="https://www.videocardbenchmark.net/gpu.php?gpu=GeForce+RTX+2080+Ti&id=3991">Nvidia RTX 2080Ti</a>.</p></span></td>
	</tr>	
	<tr>
		<th class="auto-style1" style="width: 40%"><span lang="pl">Natural Language Processing</span></th>
		<th class="auto-style1" style="width: 60%"><span lang="pl"></span></th>
	</tr>
	<tr>
		<th class="auto-style1A" style="width: 10%"><span lang="pl">Binary classification</span></th>
		<th class="auto-style1A" style="width: 50%"><span lang="pl"></span></th>
		<th class="auto-style1A" style="width: 40%"><span lang="pl"></span></th>
	</tr>
	<tr><td class="auto-style6"><span lang="pl">
		<a href="https://marcinchomicz.github.io/Natural%20language%20processing/IMDB_sentiment_analysis/IMDB%20sentiment%20analysis%20CS.html">
			IMDB Sentiment - ML classifiers and BOW representation</a>
		<a href="https://marcinchomicz.github.io/Natural%20language%20processin/IMDB_sentiment_analysis/IMDB%20sentiment%20analysis%20CS.ipynb">
				<img class="auto-style5" height="10" src="logo_svg.png" width="37"></a></span>
		</td>
		<td class="auto-style7A">
			<span lang="pl"><p>Tf-idf and count based representation classified using Hyperopt optimized traditional classifiers: LinearSVC, BernoulliNB and Logistic regression</p></span></td>
	</tr>	
	<tr>
		<td class="auto-style6"><span lang="pl">
			<a href="https://marcinchomicz.github.io/Natural%20language%20processing/IMDB_sentiment_analysis/IMDB%20Gensim%20Doc2Vec%20sentiment.html">
				IMDB Sentiment - Gensim Doc2Vec</a>
			<a href="https://marcinchomicz.github.io/Natural%20language%20processing/IMDB_sentiment_analysis/IMDB%20Gensim%20Doc2Vec%20sentiment.ipynb">
				<img class="auto-style5" height="10" src="logo_svg.png" width="37"></a></span>
		</td>
		<td class="auto-style7A">
			<span lang="pl"><p>Gensim paragraph embedding representation used as input for logistic regression classifier. Hyperparameters of both, vectorization and classification tuned with Hyperopt.</p></span>
		</td>
	</tr>
	<tr>
		<td class="auto-style6"><span lang="pl">
			<a href="https://marcinchomicz.github.io/Natural%20language%20processing/IMDB_sentiment_analysis/IMDB_CNN_averaged.html">
				IMDB Sentiment - word embedding with averaged CNN</a>
			<a href="https://marcinchomicz.github.io/Natural%20language%20processing/IMDB_sentiment_analysis/IMDB_CNN_averaged.ipynb">
				<img class="auto-style5" height="10" src="logo_svg.png" width="37"></a></span>
		</td>
		<td class="auto-style7A">
			<span lang="pl"><p>Five CNN variants used for classification on word embeddings. Hyperparamteters tuned with Bayesian algorithm.</p></span>
		</td>		
	</tr>	
</table>

</body>

</html>
