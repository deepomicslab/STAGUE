The baseline deep learning (DL) methods that learn cell embeddings, including
[DeepLinc](https://github.com/xryanglab/DeepLinc),
[CCST](https://github.com/xiaoyeye/CCST),
[STAGATE](https://github.com/zhanglabtools/STAGATE),
[SpaceFlow](https://github.com/hongleir/SpaceFlow?tab=readme-ov-file),
[SiGra](https://github.com/QSong-github/SiGra),
[SCAN-IT](https://github.com/zcang/SCAN-IT),
[SEDR](https://sedr.readthedocs.io/en/latest/Tutorial1_Clustering.html),
and [GraphST](https://github.com/JinmiaoChenLab/GraphST),
employed a variety of unsupervised clustering algorithms to cluster the learned embeddings, 
such as k-means, k-means++, mclust, louvain, and leiden.
STAGUE provides the flexibility of choosing different clustering algorithms.
In the benchmarking study, we observed that various datasets exhibit preferences for different clustering algorithms, 
with k-means consistently showing balanced performance and high efficiency. 
We standardized the benchmarking workflow for STAGUE and other DL-based baselines 
to ensure a fair comparison of the learned embeddings' representational capability 
and mitigate the variability introduced by the choice of clustering algorithms:

   1. The k-means algorithm is executed five times on the latent embeddings from each epoch, using random seeds `{0,1,2,3,4}`,
   and the average Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI) values are calculated.
   The best ARI and its corresponding AMI across all epochs are recorded as the measure of performance for a single run.
   2. Each method is run five times using random seeds `{0,1,2,3,4}`, adhering to the procedure described in point 1.
   For the baseline methods, their default training epochs are utilized.
   Finally, the average performance across the five runs is reported.

The benchmark scripts for STAGUE are provided in the current directory.
Refer to the benchmark scripts of [GraphST](https://drive.google.com/file/d/1qotfLf1P1mfy_8YO0eu_88gFwMYSkI2a/view?usp=sharing) as an example for the baseline methods.
