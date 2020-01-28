using SNNP.MLP;
using System.Collections.Generic;

namespace SNNP.kMeans
{
    public class KMeans
    {
        public int k;
        public int iterations;
        public double[,] dataset;

        private int rows;
        private int cols;

        private Cluster[] clusters;

        private Dictionary<int, int> lastClosest;
        private Dictionary<int, int> stopCriteria;

        /// <summary>
        /// Create a vanilla KMeans algorithm.
        /// </summary>
        /// <param name="data">Data to be clustered.</param>
        /// <param name="nClusters">Number of clusters.</param>
        /// <param name="minIterations">Amount of minimal iterations to find the clusters.</param>
        public KMeans(double[,] data, int nClusters, int minIterations = 2)
        {
            k = nClusters;
            dataset = data;
            iterations = minIterations;
            rows = data.GetLength(0);
            cols = data.GetLength(1);

            SetClustersFromDataset(dataset);

            bool stop;
            int nIter = 0;

            do
            {
                nIter++;

                for (int j = 0; j < clusters.Length; j++)
                    clusters[j].points.Clear();

                GetClosestClusters();

                for (int j = 0; j < k; j++)
                    clusters[j].RecalculatePosition();

                stop = true;

                if (stopCriteria == null)
                {
                    stopCriteria = lastClosest;
                    continue;
                }

                // Cheking if the 'lastClosest' is equal to 'stopCriteria'

                foreach (var a in lastClosest.Keys)
                    // If any key is different, don't stop
                    if (lastClosest[a] != stopCriteria[a])
                        stop = false;

            } while (!stop || nIter <= iterations);
        }

        /// <summary>
        /// Adding points from the dataset to their closest cluster.
        /// </summary>
        private void GetClosestClusters()
        {
            // int - Index of the line in the dataset; int - Cluster's index
            Dictionary<int, int> closest = new Dictionary<int, int>();

            for (int i = 0; i < rows; i++)
            {
                // Finding the closest cluster of every line in the dataset

                // Closest index
                int recordIndex = 0;
                // Smaller Euclidean distance
                double recordDistance = double.MaxValue;

                for (int j = 0; j < clusters.Length; j++)
                {
                    double[] position = new double[cols];

                    for (int m = 0; m < position.Length; m++)
                        position[m] = dataset[i, m];

                    double distance = Utility.GetDistanceSqr(position, clusters[j].position);

                    if (distance < recordDistance)
                    {
                        recordIndex = j;
                        recordDistance = distance;
                    }
                }

                if (closest.ContainsKey(recordIndex))
                    closest[recordIndex]++;
                else
                    closest.Add(recordIndex, 1);

                double[] point = new double[cols];

                for (int m = 0; m < point.Length; m++)
                    point[m] = dataset[i, m];

                clusters[recordIndex].points.Add(point);
            }

            lastClosest = closest;
        }

        /// <summary>
        /// Set the initial clusters from the dataset ramdomly. It never gets the same feature twice as long as there are enough data in the dataset.
        /// </summary>
        /// <param name="dataset">Dataset to get the initial clusters.</param>
        private void SetClustersFromDataset(double[,] dataset)
        {
            clusters = new Cluster[k];
            // Already found indices
            int[] iCluster = new int[k];

            for (int i = 0; i < k; i++)
                iCluster[i] = -1;

            for (int i = 0; i < k; i++)
            {
                // Check if a line in the dataset was found in the initial clusters
                bool clusterFound = false;
                // Random index within the dataset
                int randomIndex = Utility.Next(0, cols);

                // Check if the random index already exists in the cluster's array of indexes
                for (int j = 0; j < iCluster.Length; j++)
                    // If the cluster already exists
                    if (iCluster[j] == randomIndex)
                    {
                        clusterFound = true;
                        break;
                    }

                if (!clusterFound)
                {
                    iCluster[i] = randomIndex;
                    clusters[i] = new Cluster(new double[cols]);

                    for (int j = 0; j < cols; j++)
                        clusters[i].position[j] = dataset[randomIndex, j];
                }
                else
                    i--;
            }
        }
    }
}