using SNNP.MLP;
using System.Collections.Generic;

namespace SNNP.kMeans
{
    public class KMeans
    {
        public int _k;
        public double[,] _dataset;
        public int iterations;
        private Cluster[] clusters;

        private Dictionary<int, int> lastClosest;
        private Dictionary<int, int> stopCriteria;

        public KMeans(double[,] dataset, int k, int minIterations = 2)
        {
            _k = k;
            _dataset = dataset;
            iterations = minIterations;

            SetClustersFromDataset(_dataset);

            bool stop;
            int nIter = 0;

            do
            {
                nIter++;

                for (int j = 0; j < clusters.Length; j++)
                    clusters[j].points.Clear();

                GetClosestClusters();

                for (int j = 0; j < _k; j++)
                    clusters[j].RecalculatePosition();

                stop = true;

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

            } while (!stop || nIter <= iterations);
        }

        private void GetClosestClusters()
        {
            // int - index of the line in the dataset; int - cluster's index
            Dictionary<int, int> closest = new Dictionary<int, int>();

            for (int i = 0; i < _dataset.GetLength(0); i++)
            {
                // Encontrar o cluster mais próximo de cada linha do dataset

                // Índice do cluster mais próximo
                int recordIndex = 0;
                // Menor distância
                double recordDistance = double.MaxValue;

                for (int j = 0; j < clusters.Length; j++)
                {
                    double[] position = new double[_dataset.GetLength(1)];

                    for (int m = 0; m < position.Length; m++)
                        position[m] = _dataset[i, m];

                    double distance = Utility.GetDistance(position, clusters[j].position);

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

                double[] point = new double[_dataset.GetLength(1)];

                for (int m = 0; m < point.Length; m++)
                    point[m] = _dataset[i, m];

                clusters[recordIndex].points.Add(point);
            }

            lastClosest = closest;
        }

        private void SetClustersFromDataset(double[,] dataset)
        {
            clusters = new Cluster[_k];
            // Índices já encontrados
            int[] iCluster = new int[_k];

            for (int i = 0; i < _k; i++)
                iCluster[i] = -1;

            for (int i = 0; i < _k; i++)
            {
                //Console.WriteLine(string.Format("Setting cluster k = {0}", i + 1));

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
                    clusters[i] = new Cluster(new double[dataset.GetLength(1)]);

                    for (int j = 0; j < dataset.GetLength(1); j++)
                        clusters[i].position[j] = dataset[randomIndex, j];
                }
                else
                    i--;
            }
        }
    }
}