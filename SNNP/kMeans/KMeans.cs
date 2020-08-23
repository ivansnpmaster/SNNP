using System;
using System.Collections.Generic;

namespace SNNP.kMeans
{
    [Serializable()]
    public class KMeans
    {
        public int k;
        public int iterations;
        public double[,] dataset;

        private int rows;
        private int cols;

        private Cluster[] clusters;

        private Dictionary<int, int> lastClosest;
        private Dictionary<int, int> stopCriteria = null;

        /// <summary>
        /// Create a vanilla KMeans algorithm.
        /// </summary>
        /// <param name="data">Data to be clustered.</param>
        /// <param name="nClusters">Number of clusters.</param>
        /// <param name="minIterations">Amount of minimal iterations to find the clusters.</param>
        public KMeans(double[,] data, int nClusters, int minIterations = 2, bool saveOutput = false)
        {
            k = nClusters;
            dataset = data;
            rows = data.GetLength(0);
            cols = data.GetLength(1);

            SetClustersFromDataset(dataset);

            bool stop;
            int nIter = 0;

            do
            {
                nIter++;

                for (int i = 0; i < clusters.Length; i++)
                    clusters[i].points.Clear();

                GetClosestClusters();

                for (int i = 0; i < k; i++)
                    clusters[i].RecalculatePosition();

                stop = true;

                if (stopCriteria == null)
                    stopCriteria = lastClosest;
                else
                {
                    for (int i = lastClosest.Count; i-- > 0;)
                        if (lastClosest[i] != stopCriteria[i])
                        {
                            stop = false;
                            break;
                        }

                    if (!stop)
                        stopCriteria = lastClosest;
                }

            } while (!stop || nIter < minIterations);

            iterations = nIter;

            if (saveOutput)
            {
                // Save class file
                Utility.Save<KMeans>(AppContext.BaseDirectory, this);

                // +1 for the classification
                object[,] exportData = new object[rows, cols + 1];

                for (int i = rows; i-- > 0;)
                {
                    for (int j = cols; j-- > 0;)
                        exportData[i, j] = data[i, j];

                    exportData[i, cols] = stopCriteria[i];
                }

                Utility.ExportCSV(exportData);
            }
        }

        /// <summary>
        /// Adding points from the dataset to their closest cluster.
        /// </summary>
        private void GetClosestClusters()
        {
            // int - Index of the line in the dataset; int - Cluster's index
            var closest = new Dictionary<int, int>();

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

                    double distance = Mathf.GetDistanceSqr(position, clusters[j].position);

                    if (distance < recordDistance)
                    {
                        recordIndex = j;
                        recordDistance = distance;
                    }
                }

                closest.Add(i, recordIndex);

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
                int randomIndex = Mathf.Next(0, rows);

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

                    double[] point = new double[cols];

                    for (int j = 0; j < cols; j++)
                        point[j] = dataset[randomIndex, j];

                    clusters[i] = new Cluster(point);
                }
                else
                    i--;
            }
        }
    }
}