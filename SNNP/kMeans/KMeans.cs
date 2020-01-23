using SNNP.MLP;
using System.Collections.Generic;

namespace SNNP.kMeans
{
    public class KMeans
    {
        public int k;
        public double[,] dataset;
        public int iterations;
        private Cluster[] clusters;

        private Dictionary<int, int> lastClosest;
        private Dictionary<int, int> stopCriteria;

        public KMeans(double[,] _dataset, int _k, int _iterations)
        {
            k = _k;
            dataset = _dataset;
            iterations = _iterations;

            SetClustersFromDataset(dataset);

            for (int i = 0; i < iterations; i++)
            {
                for (int j = 0; j < clusters.Length; j++)
                    clusters[j].points.Clear();

                GetClosestClusters();

                for (int j = 0; j < k; j++)
                    clusters[j].RecalculatePosition();

                bool stop = true;

                if (stopCriteria == null)
                {
                    stopCriteria = lastClosest;
                    continue;
                }

                // Verificar se o lastClosest é igual ao stopCriteria

                foreach (var a in lastClosest.Keys)
                    // Se qualquer key da iteração t for diferente da iteração t-1
                    if (lastClosest[a] != stopCriteria[a])
                        stop = false;

                if (stop)
                    break;
            }
        }

        private void GetClosestClusters()
        {
            // int - index of the line in the dataset; int - cluster's index
            Dictionary<int, int> closest = new Dictionary<int, int>();

            for (int i = 0; i < dataset.GetLength(0); i++)
            {
                int recordIndex = 0;
                double recordDistance = 0;

                for (int j = 0; j < clusters.Length; j++)
                {
                    double[] position = new double[dataset.GetLength(1)];

                    for (int m = 0; m < position.Length; m++)
                        position[m] = dataset[i, m];

                    double distance = Utility.GetDistance(position, clusters[j].position);

                    if (distance < recordDistance)
                    {
                        recordIndex = j;
                        recordDistance = distance;
                    }
                }

                closest[recordIndex]++;

                double[] point = new double[dataset.GetLength(1)];

                for (int m = 0; m < point.Length; m++)
                    point[m] = dataset[i, m];

                clusters[recordIndex].points.Add(point);
            }

            lastClosest = closest;
        }

        private void SetClustersFromDataset(double[,] dataset)
        {
            clusters = new Cluster[k];
            // Índices já encontrados
            int[] iCluster = new int[k];

            for (int i = 0; i < k; i++)
            {
                // Checar se uma linha do dataset foi encontrada nos clusteres iniciais
                bool clusterFound = false;
                // Um índice randômico do dataset
                int randomIndex = Utility.Next(0, dataset.GetLength(1));

                // Checando se já existe o índice randômico no array de índices de clusteres
                for (int j = 0; j < iCluster.Length; j++)
                    // Se existe um cluster já adicionado
                    if (iCluster[j] == randomIndex)
                    {
                        clusterFound = true;
                        break;
                    }

                if (!clusterFound)
                {
                    iCluster[i] = randomIndex;
                    clusters[i].position = new double[dataset.GetLength(1)];

                    for (int j = 0; j < dataset.GetLength(1); j++)
                        clusters[i].position[j] = dataset[randomIndex, j];
                }
                else
                    i--;
            }
        }
    }
}